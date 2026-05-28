"""微批次双缓冲区 — 三维级联刷盘闸门。

设计文档 5.4.1 / 5.4.2：
- 双缓冲区交替读写，禁止消费者直连数据库
- 三级联条件触发刷盘：容量 30 条 / 5秒定时 / 析构逃生
"""
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class MicroBatchBuffer:
    """微批次双缓冲区 — 防订单数据蒸发。

    Buffer_A 与 Buffer_B 交替读写，满足任一条件触发批量刷盘。
    提供 emergency_flush 用于信号/异常时的最后抢救。
    """

    def __init__(self, batch_size: int = 30, flush_interval: float = 5.0,
                 flush_callback: Optional[Callable[[list[dict]], int]] = None):
        """
        Args:
            batch_size: 容量阈值，达到即刷盘
            flush_interval: 定时器间隔（秒），超时即刷盘
            flush_callback: 刷盘回调，签名为 (orders: list[dict]) -> int (写入条数)
        """
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._flush_cb = flush_callback

        self._buffer_a: list[dict] = []
        self._buffer_b: list[dict] = []
        self._active: list[dict] = self._buffer_a
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

        self._total_inserted = 0
        self._total_flushed = 0
        self._total_emergency = 0

        self._timer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── 生命周期 ──────────────────────────────────────────────

    def start_timer(self) -> None:
        """启动定时刷盘线程。"""
        if self._timer_thread is not None and self._timer_thread.is_alive():
            return
        self._stop_event.clear()
        self._timer_thread = threading.Thread(
            target=self._timer_loop, daemon=True, name="micro-batch-timer",
        )
        self._timer_thread.start()

    def stop_timer(self) -> None:
        self._stop_event.set()

    def _timer_loop(self) -> None:
        while not self._stop_event.wait(self._flush_interval):
            self._check_timer_flush()

    def _check_timer_flush(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_flush
            if elapsed >= self._flush_interval and self._active:
                self._flush()

    # ── 写入 ──────────────────────────────────────────────────

    def insert(self, order: dict) -> None:
        """插入一条订单到活跃缓冲区。

        达到容量阈值时自动触发刷盘。
        """
        with self._lock:
            self._active.append(order)
            if len(self._active) >= self._batch_size:
                self._flush()

    def insert_batch(self, orders: list[dict]) -> None:
        """批量插入。"""
        with self._lock:
            for order in orders:
                self._active.append(order)
                if len(self._active) >= self._batch_size:
                    self._flush()

    def _flush(self) -> None:
        """切换缓冲区并执行批量刷盘。"""
        to_flush = self._active
        # 切换活跃缓冲区
        self._active = self._buffer_b if self._active is self._buffer_a else self._buffer_a
        self._active.clear()
        self._last_flush = time.monotonic()

        if to_flush and self._flush_cb:
            try:
                count = self._flush_cb(to_flush)
                self._total_inserted += count
                self._total_flushed += 1
                logger.debug("微批次刷盘: %d 条 (第 %d 次)",
                             len(to_flush), self._total_flushed)
            except Exception:
                logger.exception("微批次刷盘回调异常")

    # ── 析构逃生 ──────────────────────────────────────────────

    def emergency_flush(self) -> int:
        """紧急刷盘：刷出活跃缓冲区所有残留数据。

        信号处理器 / atexit / 异常捕获时调用。
        返回抢救出的记录数。
        """
        with self._lock:
            rescued = list(self._active)
            self._active.clear()
            self._total_emergency += 1

        if rescued and self._flush_cb:
            try:
                count = self._flush_cb(rescued)
                self._total_inserted += count
                logger.warning("紧急刷盘: 抢救 %d 条 → 实际写入 %d 条",
                               len(rescued), count)
                return count
            except Exception:
                logger.exception("紧急刷盘失败")
                return 0

        return 0

    # ── 指标 ──────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "pending": len(self._active),
                "total_inserted": self._total_inserted,
                "total_flushes": self._total_flushed,
                "total_emergency_flushes": self._total_emergency,
                "last_flush_ago": round(time.monotonic() - self._last_flush, 1),
            }
