"""阶段十验收测试：全链路联调 + 地狱测试。

验收标准（V5.0 路线图 10.1）：
  T10.1: 完整单品链路 — 寻址→VLM→归一化→熔断→批量入库
  T10.2: 弹窗地狱 — 遮挡→挂起→恢复续跑
  T10.3: 改版熔断地狱 — 20 帧错位→kill-switch
  T10.4: 断点续跑 — Ctrl+C→checkpoint→重启续跑
  T10.5: 跨天运行 — 凌晨运行日期正确归属
  T10.6: 内存压力 — 100 单品有界队列≤10
"""
import json as _json
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.time_normalizer import TimeNormalizer
from src.core.early_stop import EarlyStopEngine, StopSignal
from src.core.temporal_state import TemporalStateMachine
from src.ocr.entry_locator import EntryLocator, EntryPoint
from src.ocr.vlm_extractor import VLMExtractor, SalesSnapshot, OrderItem
from src.pipeline.micro_batch import MicroBatchBuffer
from src.pipeline.queue_manager import ImageQueue
from src.ui_guard.overlay_detector import OverlayDetector
from src.ui_guard.layout_guard import LayoutGuard
from src.core.session import SessionManager


# ────────────────────────────────────────────────────────────
# T10.1: 完整单品链路
# ────────────────────────────────────────────────────────────

class TestFullPipeline:
    """T10.1 — 寻址→VLM提取→归一化→熔断→入库 全链路"""

    def test_full_single_product_pipeline(self):
        """一个单品从进店到订单入库的完整路径。"""
        # Step 1: 动态入口寻址
        mock_ocr = MagicMock(return_value=[
            {"text": "超300人加购", "bbox": [[100, 200], [250, 200], [250, 230], [100, 230]],
             "confidence": 0.95},
        ])
        locator = EntryLocator(ocr_func=mock_ocr)
        import numpy as np
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        entry = locator.locate(frame)
        assert entry is not None

        # Step 2: Qwen-VL 多模态提取
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "response": _json.dumps({
                    "summary": {"cart_adds": "超300人加购", "growth": "2x"},
                    "orders": [
                        {"buyer": "A**", "is_repeat": False,
                         "sku": "课程A", "time_str": "3小时前"},
                        {"buyer": "B**", "is_repeat": True,
                         "sku": "课程B", "time_str": "1天前"},
                    ],
                }),
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            extractor = VLMExtractor()
            snapshot = extractor.extract(b"fake_image")
            assert snapshot is not None
            assert len(snapshot.orders) == 2

        # Step 3: 时间归一化 + 24h 熔断
        normalizer = TimeNormalizer(
            task_start_epoch_ms=datetime(2026, 5, 28, 14, 0, 0).timestamp() * 1000,
        )
        engine = EarlyStopEngine(normalizer=normalizer)

        orders_raw = [
            {"buyer": o.buyer, "sku": o.sku, "time_str": o.time_str}
            for o in snapshot.orders
        ]
        today_orders, signal = engine.process_orders(orders_raw)

        # 只有第 1 条今日订单入库，第 2 条触发熔断
        assert len(today_orders) == 1
        assert signal == StopSignal.STOP_SCROLL_AND_CLOSE

        # Step 4: 微批次缓冲 + 入库
        flushed = []
        buf = MicroBatchBuffer(
            batch_size=5,
            flush_callback=lambda orders: flushed.extend(orders) or len(orders),
        )

        for order in today_orders:
            buf.insert(order)

        # 未达阈值，手动刷盘
        count = buf.emergency_flush()
        assert count == 1
        assert flushed[0]["buyer"] == "A**"


# ────────────────────────────────────────────────────────────
# T10.2: 弹窗地狱
# ────────────────────────────────────────────────────────────

class TestOverlayHell:
    """T10.2 — 弹窗遮挡→挂起→恢复续跑"""

    def test_overlay_detected_pauses_system(self):
        """弹窗检测命中后系统挂起，恢复后清空队列续跑。"""
        od = OverlayDetector()
        q = ImageQueue(max_size=5, low_watermark=3)

        # 填一些帧
        for i in range(3):
            q.put(f"frame_{i}")

        # 模拟弹窗
        for _ in range(3):
            od.check(["低电量", "充电", "确定"], card_regions_valid=False)

        assert od.is_blocked

        # 系统挂起：清空队列
        if od.is_blocked:
            q.block_producer()
            while not q.is_empty:
                q.get(timeout=0.1)

        assert q.is_empty

        # 模拟恢复
        od.reset()
        q.resume_producer()

        assert not od.is_blocked
        assert not q._producer_paused.is_set()


# ────────────────────────────────────────────────────────────
# T10.3: 改版熔断地狱
# ────────────────────────────────────────────────────────────

class TestUIShiftHell:
    """T10.3 — 20 帧错位→kill-switch 毫秒触发"""

    def test_killswitch_millisecond_trigger(self):
        """改版熔断在 20 帧内触发。"""
        guard = LayoutGuard()
        guard.calibrate([{"w": 540, "h": 580}] * 5)

        start = time.perf_counter()
        for i in range(20):
            ok = guard.check({"w": 200, "h": 800}, sales_value=100)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert guard.is_killed
        assert elapsed_ms < 50, f"熔断应在毫秒级触发: {elapsed_ms:.1f}ms"


# ────────────────────────────────────────────────────────────
# T10.4: 断点续跑
# ────────────────────────────────────────────────────────────

class TestCheckpoint:
    """T10.4 — Ctrl+C→checkpoint.json→重启续跑"""

    def test_checkpoint_save_and_restore(self, tmp_path):
        """断点保存和恢复。"""
        state_file = tmp_path / "state.json"
        mgr = SessionManager(state_file=str(state_file))

        # 模拟采集到一半
        mgr.new_batch(store_id="store_001")
        mgr.update_progress(progress=42, virtual_id="prod_abc123")

        before = mgr.state

        # 模拟程序重启
        mgr2 = SessionManager(state_file=str(state_file))
        after = mgr2.state

        assert after.current_store_id == before.current_store_id
        assert after.current_store_progress == 42
        assert after.last_successful_virtual_id == "prod_abc123"


# ────────────────────────────────────────────────────────────
# T10.5: 跨天运行
# ────────────────────────────────────────────────────────────

class TestMidnightCrossover:
    """T10.5 — 凌晨运行日期正确归属"""

    def test_midnight_time_normalization(self):
        """凌晨 00:30 启动，"2小时前"归属昨天。"""
        anchor = datetime(2026, 5, 28, 0, 30, 0)
        now_ms = anchor.timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        # "2小时前" = 5/27 22:30 → 应该归昨天
        date_str, should_stop = normalizer.normalize("2小时前")
        assert date_str == "2026-05-27"

        # "刚刚" = 5/28 00:30 → 今天
        date_str2, _ = normalizer.normalize("刚刚")
        assert date_str2 == "2026-05-28"

    def test_temporal_state_machine_cross_midnight(self):
        """跨天状态机检测。"""
        tsm = TemporalStateMachine()
        yesterday = datetime.now() - timedelta(days=1)
        tsm._task_start_epoch_ms = yesterday.timestamp() * 1000
        tsm._batch_id = "test_batch"
        tsm._normalizer = TimeNormalizer(task_start_epoch_ms=yesterday.timestamp() * 1000)
        tsm._active = True

        assert tsm.is_cross_midnight()
        assert tsm.get_cross_midnight_offset_days() >= 1


# ────────────────────────────────────────────────────────────
# T10.6: 内存压力
# ────────────────────────────────────────────────────────────

class TestMemoryPressure:
    """T10.6 — 100 单品有界队列 ≤ 10，无 OOM"""

    def test_queue_bounded_under_100_items(self):
        """持续输入 100 帧，队列永不超过 10。"""
        q = ImageQueue(max_size=10, low_watermark=7, producer_timeout=0.5)

        max_seen = 0
        for i in range(100):
            success = q.put(f"frame_{i}")
            sz = q.qsize
            if sz > max_seen:
                max_seen = sz
            if not success:
                # 队列满，取出一些模拟消费
                q.get(timeout=0.1)

        assert max_seen <= 10, f"队列最大容量: {max_seen}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
