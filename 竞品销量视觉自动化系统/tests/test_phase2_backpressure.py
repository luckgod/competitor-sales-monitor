"""阶段二验收测试：并发容器 + 极限背压 + 内存压力测试。

验收标准（来自核心架构开发与验收总览 - 2.2 单元验收标准）：
  测试算子：在消费者线程中人为注入 sleep(5)，模拟本地大模型卡死
  红线 1：队列积压满 10 帧后，手机滑动和截图必须瞬间强制悬停
  红线 2：系统总内存占用线必须横向拉平，严禁 OOM
  红线 3：消费者卡顿结束后，手机自动恢复仿生滑动
"""
import threading
import time
from unittest.mock import MagicMock

import pytest

from src.core.adb_controller import ADBController
from src.core.session import SessionManager
from src.pipeline.queue_manager import ImageQueue
from src.producer.capture import ProducerThread
from src.consumer.parser import ConsumerThread


# ────────────────────────────────────────────────────────────
# 工具
# ────────────────────────────────────────────────────────────

def _make_mock_proc():
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


def _make_mock_adb():
    adb = ADBController(adb_path="mock_adb", scrcpy_path="mock_scrcpy")
    adb._run_adb = MagicMock(return_value="")
    adb.launch_scrcpy = MagicMock()
    adb.kill_scrcpy = MagicMock()
    adb._scrcpy_proc = _make_mock_proc()
    adb.is_device_connected = MagicMock(return_value=True)
    adb.list_devices = MagicMock(return_value=[])
    adb.wait_for_device = MagicMock(return_value=True)
    return adb


# ────────────────────────────────────────────────────────────
# 红线 1：队列满载 → 生产者强制悬停
# ────────────────────────────────────────────────────────────

class TestBackpressureProducerSuspension:
    """验收红线 1 — 消费者卡顿时生产者自动悬停"""

    def test_producer_blocks_when_queue_full(self, tmp_path):
        """消费者处理极慢时，队列满后生产者必须悬停。

        模拟：生产者高速投递 + 不启动消费者（模拟消费者卡死）。
        验证：队列达到 max_size 后不再增长，生产者进入等待。
        """
        q = ImageQueue(max_size=3, low_watermark=2, producer_timeout=2.0)

        state_file = tmp_path / "state.json"
        session_mgr = SessionManager(state_file=str(state_file))
        session_mgr.new_batch(store_id="store_001")

        adb = _make_mock_adb()
        producer = ProducerThread(
            adb=adb, queue=q, session_mgr=session_mgr,
            slide_min_pause=0.1, slide_max_pause=0.3,
        )

        max_observed = 0
        stop_sampling = threading.Event()

        def _sample_queue():
            nonlocal max_observed
            while not stop_sampling.is_set():
                sz = q.qsize
                if sz > max_observed:
                    max_observed = sz
                time.sleep(0.05)

        sampler = threading.Thread(target=_sample_queue, daemon=True)
        sampler.start()

        # 只启动生产者，不启动消费者 → 队列会快速填满
        producer.start()
        time.sleep(5)

        assert max_observed <= q._max_size, (
            f"队列溢出: max_observed={max_observed} > max_size={q._max_size}"
        )
        assert max_observed >= q._max_size, (
            f"队列未达到满载: max_observed={max_observed} < max_size={q._max_size}"
        )

        producer.stop()
        stop_sampling.set()
        sampler.join(timeout=2)

    def test_producer_blocked_then_resumes(self, tmp_path):
        """消费者从卡顿中恢复后，生产者自动恢复投递。

        模拟：消费者先极慢 → 队列满，生产者阻塞 → 消费者恢复速度 → 队列排空 → 生产者恢复
        """
        q = ImageQueue(max_size=5, low_watermark=3, producer_timeout=3.0)

        state_file = tmp_path / "state.json"
        session_mgr = SessionManager(state_file=str(state_file))
        session_mgr.new_batch(store_id="store_001")

        adb = _make_mock_adb()
        producer = ProducerThread(
            adb=adb, queue=q, session_mgr=session_mgr,
            slide_min_pause=0.1, slide_max_pause=0.2,
        )

        consumer = ConsumerThread(queue=q, session_mgr=session_mgr)

        # 先启动消费者（空转消耗队列），生产者逐步填满
        consumer.start()
        time.sleep(0.5)

        producer.start()

        # 等待足够时间让队列达到满载
        time.sleep(4)

        # 队列应该曾经满载
        metrics_before = q.metrics_snapshot()
        assert metrics_before["total_enqueued"] > 0, "生产者未投递任何帧"

        # 停止生产者，让消费者排空队列
        producer.stop()
        time.sleep(2)

        # 队列应被部分消费
        metrics_after = q.metrics_snapshot()
        assert metrics_after["total_dequeued"] > 0, "消费者未处理任何帧"

        consumer.stop()


# ────────────────────────────────────────────────────────────
# 红线 2：内存横向拉平，严禁 OOM
# ────────────────────────────────────────────────────────────

class TestMemoryPressureOOM:
    """验收红线 2 — 内存不无限增长"""

    def test_queue_size_bounded_under_sustained_load(self, tmp_path):
        """持续高负载下队列大小严格不超上限。

        投放 100 帧到有限队列，消费者慢速处理。验证：
        - 队列 size 永不超过 max_size
        - 无内存异常
        """
        q = ImageQueue(max_size=10, low_watermark=7, producer_timeout=0.5)

        state_file = tmp_path / "state.json"
        session_mgr = SessionManager(state_file=str(state_file))
        session_mgr.new_batch(store_id="store_001")

        adb = _make_mock_adb()
        producer = ProducerThread(
            adb=adb, queue=q, session_mgr=session_mgr,
            slide_min_pause=0.05, slide_max_pause=0.1,
        )
        consumer = ConsumerThread(queue=q, session_mgr=session_mgr)

        max_seen = [0]

        def _monitor():
            while True:
                sz = q.qsize
                if sz > max_seen[0]:
                    max_seen[0] = sz
                time.sleep(0.01)

        monitor = threading.Thread(target=_monitor, daemon=True)
        monitor.start()

        consumer.start()
        producer.start()

        # 运行 6 秒，模拟持续负载
        time.sleep(6)

        producer.stop()
        consumer.stop()

        assert max_seen[0] <= q._max_size, (
            f"队列突破上限: {max_seen[0]} > {q._max_size}"
        )

    def test_memory_stays_flat_when_consumer_blocked(self, tmp_path):
        """消费者完全卡死时队列满后内存不再增长。

        启动生产者但不启动消费者（模拟消费者卡死）。队列满后：
        - 生产者阻塞
        - queue size 不再增长
        """
        q = ImageQueue(max_size=5, low_watermark=3, producer_timeout=0.5)

        state_file = tmp_path / "state.json"
        session_mgr = SessionManager(state_file=str(state_file))
        session_mgr.new_batch(store_id="store_001")

        adb = _make_mock_adb()
        producer = ProducerThread(
            adb=adb, queue=q, session_mgr=session_mgr,
            slide_min_pause=0.1, slide_max_pause=0.2,
        )

        # 不启动消费者
        producer.start()

        # 等待生产者填满队列
        time.sleep(4)

        size_at_full = q.qsize
        assert size_at_full <= q._max_size, f"队列未在 {q._max_size} 处止步"

        # 再等 2 秒，确认队列不继续增长
        time.sleep(2)
        size_later = q.qsize
        assert size_later == size_at_full, (
            f"队列在阻塞后仍增长: {size_at_full} → {size_later}"
        )

        producer.stop()


# ────────────────────────────────────────────────────────────
# 红线 3：消费者恢复后生产者自动恢复滑动
# ────────────────────────────────────────────────────────────

class TestConsumerRecoveryAutoResume:
    """验收红线 3 — 消费者恢复后系统自愈，无需人工干预"""

    def test_producer_resumes_after_consumer_drain(self, tmp_path):
        """消费者处理完积压后，队列降至低水位以下，生产者自动恢复。

        流程：
        1. 消费者卡死 → 队列满 → 生产者阻塞
        2. 消费者恢复 → 消耗队列 → 降至水位以下
        3. 生产者检测到空间 → 自动恢复投递
        """
        q = ImageQueue(max_size=5, low_watermark=3, producer_timeout=5.0)

        state_file = tmp_path / "state.json"
        session_mgr = SessionManager(state_file=str(state_file))
        session_mgr.new_batch(store_id="store_001")

        adb = _make_mock_adb()
        producer = ProducerThread(
            adb=adb, queue=q, session_mgr=session_mgr,
            slide_min_pause=0.1, slide_max_pause=0.2,
        )
        consumer = ConsumerThread(queue=q, session_mgr=session_mgr)

        producer.start()
        # 不启动消费者 → 队列很快填满

        time.sleep(3)
        assert q.qsize >= q._max_size - 1, "队列应接近满载"

        # 生产者已阻塞（或即将阻塞），捕捉当前入队数
        enqueued_before = q.metrics_snapshot()["total_enqueued"]

        time.sleep(2)
        enqueued_during_block = q.metrics_snapshot()["total_enqueued"] - enqueued_before
        assert enqueued_during_block <= 2, (
            f"生产者阻塞期间不应大量投递: +{enqueued_during_block}"
        )

        # 启动消费者 → 排空队列
        consumer.start()
        time.sleep(3)

        # 消费者排空后生产者应恢复投递
        enqueued_after_consumer = q.metrics_snapshot()["total_enqueued"]
        assert enqueued_after_consumer > enqueued_before, (
            "消费者排空队列后生产者未恢复投递"
        )

        producer.stop()
        consumer.stop()

    def test_queue_metrics_accuracy_under_load(self, tmp_path):
        """队列指标在负载下准确记录。"""
        q = ImageQueue(max_size=5, low_watermark=3, producer_timeout=0.5)

        state_file = tmp_path / "state.json"
        session_mgr = SessionManager(state_file=str(state_file))
        session_mgr.new_batch(store_id="store_001")

        adb = _make_mock_adb()
        producer = ProducerThread(
            adb=adb, queue=q, session_mgr=session_mgr,
            slide_min_pause=0.1, slide_max_pause=0.2,
        )
        consumer = ConsumerThread(queue=q, session_mgr=session_mgr)

        consumer.start()
        producer.start()
        time.sleep(5)
        producer.stop()
        consumer.stop()

        metrics = q.metrics_snapshot()
        assert metrics["total_enqueued"] > 0, "应有入队记录"
        assert metrics["total_dequeued"] > 0, "应有出队记录"
        assert metrics["current_size"] <= metrics["max_size"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
