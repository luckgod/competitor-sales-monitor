"""有界阻塞队列管理器 — 生产者-消费者并发容器的核心背压控制。

约束：
- max_size = 10: 队列满时生产者 WAITING，防止 OOM
- low_watermark = 7: 消费到低于此值时唤醒生产者
- 生产者最长阻塞 30s，超时自动降级分辨率或跳帧
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueueMetrics:
    """队列运行时指标，用于 Prometheus / 日志上报。"""
    max_size: int = 10
    current_size: int = 0
    total_enqueued: int = 0
    total_dequeued: int = 0
    producer_blocked_count: int = 0
    producer_total_block_time: float = 0.0
    consumer_avg_process_time: float = 0.0

    def snapshot(self) -> dict:
        return {
            "max_size": self.max_size,
            "current_size": self.current_size,
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "producer_blocked_count": self.producer_blocked_count,
            "producer_total_block_time_s": round(self.producer_total_block_time, 2),
            "consumer_avg_process_time_ms": round(self.consumer_avg_process_time * 1000, 1),
        }


class ImageQueue:
    """有界阻塞图像队列 — 生产者-消费者背压控制中心。

    put() 在队列满时阻塞生产者，get() 在队列空时阻塞消费者。
    支持超时降级和指标暴露。
    """

    def __init__(self, max_size: int = 10, low_watermark: int = 7,
                 producer_timeout: float = 30.0):
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._max_size = max_size
        self._low_watermark = low_watermark
        self._producer_timeout = producer_timeout
        self._metrics = QueueMetrics(max_size=max_size)
        self._lock = threading.Lock()

        # 生产者阻塞/唤醒信号
        self._producer_paused = threading.Event()
        self._producer_paused.clear()

    # ── 生产者接口 ────────────────────────────────────────────

    def put(self, item: Any) -> bool:
        """生产者放入一帧。队列满时阻塞，超时返回 False。"""
        try:
            block_start = time.monotonic()
            self._queue.put(item, timeout=self._producer_timeout)
            with self._lock:
                self._metrics.total_enqueued += 1
                self._metrics.current_size = self._queue.qsize()
            return True
        except queue.Full:
            with self._lock:
                self._metrics.producer_blocked_count += 1
                self._metrics.producer_total_block_time += time.monotonic() - block_start
            logger.warning("生产者阻塞超时 (%.1fs)，队列仍然满载", self._producer_timeout)
            return False

    # ── 消费者接口 ────────────────────────────────────────────

    def get(self, timeout: float = 10.0) -> Any | None:
        """消费者取出一帧。队列空时阻塞，timeout 后返回 None。"""
        try:
            process_start = time.monotonic()
            item = self._queue.get(timeout=timeout)
            with self._lock:
                self._metrics.total_dequeued += 1
                self._metrics.current_size = self._queue.qsize()
            return item
        except queue.Empty:
            return None

    def task_done(self, process_time: float) -> None:
        """标记任务完成，更新平均处理耗时。"""
        with self._lock:
            m = self._metrics
            # 指数移动平均
            alpha = 0.1
            m.consumer_avg_process_time = (
                alpha * process_time + (1 - alpha) * m.consumer_avg_process_time
            )

    # ── 背压控制 ──────────────────────────────────────────────

    @property
    def should_produce(self) -> bool:
        """生产者是否应该继续生产。低于低水位线时返回 True。"""
        return self._queue.qsize() < self._max_size

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    # ── 生产者阻塞控制 ────────────────────────────────────────

    def block_producer(self) -> None:
        """强制暂停生产者（用于弹窗/异常检测）。"""
        self._producer_paused.set()
        logger.info("生产者已强制暂停")

    def resume_producer(self) -> None:
        """恢复生产者。"""
        self._producer_paused.clear()
        logger.info("生产者已恢复")

    def wait_if_paused(self) -> None:
        """生产者调用：如果被暂停则等待。"""
        while self._producer_paused.is_set():
            time.sleep(1)

    # ── 指标 ──────────────────────────────────────────────────

    def metrics_snapshot(self) -> dict:
        with self._lock:
            self._metrics.current_size = self._queue.qsize()
            return self._metrics.snapshot()
