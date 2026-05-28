"""阶段七验收测试：时间归一化 + 24h 熔断 + 跨天状态机。

验收标准（V5.0 路线图 7.2）：
  T7.1: "15小时前" → 归一化日期 = 锚点 - 15h
  T7.2: "3分钟前" → 归一化日期 = today
  T7.3: "1天前" 命中 → STOP 信号
  T7.4: 混合时间戳 → 今日订单完整，跨天截断
  T7.5: 23:30 启动 → "2小时前" 归属昨天
  T7.6: batch_id 隔离 → 跨天运行数据对齐
"""
from datetime import datetime, timedelta

import pytest

from src.core.time_normalizer import TimeNormalizer
from src.core.early_stop import EarlyStopEngine, StopSignal
from src.core.temporal_state import TemporalStateMachine


# ────────────────────────────────────────────────────────────
# T7.1 + T7.2: 时间归一化
# ────────────────────────────────────────────────────────────

class TestTimeNormalizer:
    """T7.1/T7.2 — 相对时间归一化为绝对日期"""

    def test_hours_ago_normalized(self):
        """T7.1: "15小时前" → 日期 = 锚点 - 15h。"""
        now_ms = datetime(2026, 5, 28, 14, 0, 0).timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        date_str, should_stop = normalizer.normalize("15小时前")
        assert date_str == "2026-05-27"  # 前一天的 23:00
        assert should_stop is False

    def test_minutes_ago_normalized(self):
        """T7.2: "3分钟前" → 日期 = today。"""
        now_ms = datetime(2026, 5, 28, 14, 0, 0).timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        date_str, should_stop = normalizer.normalize("3分钟前")
        assert date_str == "2026-05-28"
        assert should_stop is False

    def test_just_now_normalized(self):
        """"刚刚" → today。"""
        now_ms = datetime(2026, 5, 28, 14, 0, 0).timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        date_str, _ = normalizer.normalize("刚刚")
        assert date_str == "2026-05-28"

    def test_empty_time_str_returns_today(self):
        """空字符串 → today。"""
        normalizer = TimeNormalizer()
        date_str, should_stop = normalizer.normalize("")
        assert date_str == normalizer.today_str
        assert should_stop is False

    def test_days_ago_triggers_stop(self):
        """T7.3: "1天前" → should_stop=True。"""
        now_ms = datetime(2026, 5, 28, 14, 0, 0).timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        date_str, should_stop = normalizer.normalize("1天前")
        assert date_str == "2026-05-27"
        assert should_stop is True

    def test_multi_days_ago_all_stop(self):
        """"2天前" / "3天前" → 均触发熔断。"""
        now_ms = datetime(2026, 5, 28, 14, 0, 0).timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        for days, expected_date in [(1, "2026-05-27"), (2, "2026-05-26"), (3, "2026-05-25")]:
            date_str, should_stop = normalizer.normalize(f"{days}天前")
            assert date_str == expected_date
            assert should_stop is True

    def test_today_time_normalized(self):
        """"今天 14:30" → today。"""
        now_ms = datetime(2026, 5, 28, 14, 0, 0).timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        date_str, should_stop = normalizer.normalize("今天 14:30")
        assert date_str == "2026-05-28"
        assert should_stop is False

    # ── T7.5: 跨天场景 ───────────────────────────────────────

    def test_midnight_crossover_correct_date(self):
        """T7.5: 凌晨 00:30 启动，"2小时前"订单归属昨天。"""
        # 模拟 2026-05-28 00:30 启动
        anchor = datetime(2026, 5, 28, 0, 30, 0)
        now_ms = anchor.timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        date_str, should_stop = normalizer.normalize("2小时前")
        # 锚点 - 2h = 5/27 22:30 → 应归属 2026-05-27
        assert date_str == "2026-05-27", f"跨天订单应归属昨天，实际: {date_str}"
        assert should_stop is False

    def test_midnight_days_ago_still_stops(self):
        """跨天场景下 "1天前" 仍触发熔断。"""
        anchor = datetime(2026, 5, 28, 0, 30, 0)
        now_ms = anchor.timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)

        date_str, should_stop = normalizer.normalize("1天前")
        assert date_str == "2026-05-27"
        assert should_stop is True


# ────────────────────────────────────────────────────────────
# T7.3 + T7.4: 24h 熔断引擎
# ────────────────────────────────────────────────────────────

class TestEarlyStopEngine:
    """T7.3/T7.4 — 24h 熔断截断"""

    @pytest.fixture
    def engine(self):
        """创建固定锚点的引擎（2026-05-28 14:00）。"""
        anchor_ms = datetime(2026, 5, 28, 14, 0, 0).timestamp() * 1000
        normalizer = TimeNormalizer(task_start_epoch_ms=anchor_ms)
        return EarlyStopEngine(normalizer=normalizer)

    def test_all_today_no_stop(self, engine):
        """全部今日订单 → 不触发熔断。"""
        orders = [
            {"buyer": "A**", "sku": "SKU1", "time_str": "15小时前"},
            {"buyer": "B**", "sku": "SKU2", "time_str": "3小时前"},
            {"buyer": "C**", "sku": "SKU3", "time_str": "刚刚"},
        ]

        today_orders, signal = engine.process_orders(orders)
        assert len(today_orders) == 3
        assert signal == StopSignal.NONE

    def test_mixed_timestamps_stops_at_first_days_ago(self, engine):
        """T7.4: 混合时间戳 → 今日订单完整，首条跨天截断。"""
        orders = [
            {"buyer": "A**", "sku": "SKU1", "time_str": "3小时前"},
            {"buyer": "B**", "sku": "SKU2", "time_str": "1小时前"},
            {"buyer": "C**", "sku": "SKU3", "time_str": "刚刚"},
            {"buyer": "D**", "sku": "SKU4", "time_str": "1天前"},     # ← 熔断点
            {"buyer": "E**", "sku": "SKU5", "time_str": "2天前"},     # 被截断
            {"buyer": "F**", "sku": "SKU6", "time_str": "3天前"},     # 被截断
        ]

        today_orders, signal = engine.process_orders(orders)

        # 前 3 条今日订单入库
        assert len(today_orders) == 3
        assert today_orders[0]["order_date"] == "2026-05-28"  # "3小时前" = 今天 11:00
        assert today_orders[1]["order_date"] == "2026-05-28"  # "1小时前" = 今天 13:00
        assert today_orders[2]["order_date"] == "2026-05-28"  # "刚刚"

        # 触发熔断
        assert signal == StopSignal.STOP_SCROLL_AND_CLOSE

    def test_should_stop_quick_check(self, engine):
        """should_stop 快速预检正确。"""
        assert engine.should_stop("刚刚") is False
        assert engine.should_stop("3小时前") is False
        assert engine.should_stop("1天前") is True
        assert engine.should_stop("2天前") is True

    def test_reset_clears_state(self, engine):
        """reset 清除批次状态。"""
        orders = [
            {"buyer": "A**", "sku": "SKU1", "time_str": "1天前"},
        ]
        engine.process_orders(orders)
        assert engine.stats["stopped_this_batch"] is True

        engine.reset()
        assert engine.stats["stopped_this_batch"] is False


# ────────────────────────────────────────────────────────────
# T7.6: 跨天状态机 + batch 隔离
# ────────────────────────────────────────────────────────────

class TestTemporalStateMachine:
    """T7.6 — 跨天批次隔离"""

    def test_initialize_generates_batch_id(self):
        """初始化生成唯一 batch_id。"""
        tsm = TemporalStateMachine()
        bid1 = tsm.initialize()
        assert len(bid1) == 12
        assert tsm.is_active

        tsm2 = TemporalStateMachine()
        bid2 = tsm2.initialize()
        assert bid1 != bid2

    def test_custom_batch_id(self):
        """自定义 batch_id 生效。"""
        tsm = TemporalStateMachine()
        bid = tsm.initialize(batch_id="my_custom_batch")
        assert bid == "my_custom_batch"
        assert tsm.batch_id == "my_custom_batch"

    def test_normalize_uses_anchor(self):
        """normalize 基于锚点日期解析。"""
        tsm = TemporalStateMachine()
        # 固定锚点
        anchor = datetime(2026, 5, 28, 14, 0, 0)
        tsm._task_start_epoch_ms = anchor.timestamp() * 1000
        tsm._batch_id = "test"
        tsm._normalizer = TimeNormalizer(task_start_epoch_ms=anchor.timestamp() * 1000)
        tsm._active = True

        date_str, should_stop = tsm.normalize("2小时前")
        assert date_str == "2026-05-28"

    def test_cross_midnight_detection(self):
        """跨天检测：锚点日期 ≠ 当前日期。"""
        tsm = TemporalStateMachine()
        # 锚点设为昨天
        yesterday = datetime.now() - timedelta(days=1)
        tsm._task_start_epoch_ms = yesterday.timestamp() * 1000
        tsm._batch_id = "test"
        tsm._normalizer = TimeNormalizer(task_start_epoch_ms=yesterday.timestamp() * 1000)
        tsm._active = True

        assert tsm.is_cross_midnight() is True
        assert tsm.get_cross_midnight_offset_days() >= 1

    def test_same_day_not_cross_midnight(self):
        """同天锚点不触发跨天。"""
        tsm = TemporalStateMachine()
        tsm.initialize()

        # 锚点就是现在
        assert tsm.is_cross_midnight() is False
        assert tsm.get_cross_midnight_offset_days() == 0

    def test_deactivate_stops_state(self):
        """停用后不再活跃。"""
        tsm = TemporalStateMachine()
        tsm.initialize()
        assert tsm.is_active

        tsm.deactivate()
        assert not tsm.is_active


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
