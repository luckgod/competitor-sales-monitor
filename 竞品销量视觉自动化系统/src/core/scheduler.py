"""定时爆破调度器 — 凌晨自动唤醒跑盘 + 跑完杀进程锁屏。

设计规范（来自设计文档 5.x/10.2）：
- 每日 2:00-4:00 自动唤醒设备执行总盘
- 跑完立即杀掉目标 App，手机回归锁屏静止
- Ollama 预热：凌晨激活时发送 keepalive 请求，防止模型卸载
- 跑完后释放 Ollama 模型恢复低功耗
- 容量预算预警：评估总耗时 > 2.5h 时前端强警告
"""
import logging
import threading
import time
from datetime import datetime, time as dt_time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SweepScheduler:
    """凌晨定时跑盘调度器。

    独立的守护线程，锁定每日 2:00-4:00 窗口自动执行。
    支持 Ollama 预热/释放、容量预算评估。
    """

    def __init__(self, start_time: str = "02:00", end_time: str = "04:00",
                 max_duration: int = 9000,  # 2.5h
                 store_count: int = 0, avg_items_per_store: int = 0,
                 avg_slide_interval: float = 3.0):
        self._start = dt_time.fromisoformat(start_time)
        self._end = dt_time.fromisoformat(end_time)
        self._max_duration = max_duration
        self._store_count = store_count
        self._avg_items = avg_items_per_store
        self._avg_slide = avg_slide_interval

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._on_sweep_start: list[Callable[[], None]] = []
        self._on_sweep_end: list[Callable[[], None]] = []
        self._on_capacity_warning: list[Callable[[str], None]] = []

        self._sweep_active = threading.Event()

    # ── 回调 ──────────────────────────────────────────────────

    def on_sweep_start(self, callback: Callable[[], None]) -> None:
        self._on_sweep_start.append(callback)

    def on_sweep_end(self, callback: Callable[[], None]) -> None:
        self._on_sweep_end.append(callback)

    def on_capacity_warning(self, callback: Callable[[str], None]) -> None:
        self._on_capacity_warning.append(callback)

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="scheduler")
        self._thread.start()
        logger.info("调度器已启动，窗口 %s - %s", self._start, self._end)

    def stop(self) -> None:
        self._stop_event.set()
        self._sweep_active.clear()
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ── 容量预算 ──────────────────────────────────────────────

    def estimate_duration(self, store_count: int = 0,
                           avg_items: int = 0) -> float:
        """预估本次跑盘总耗时（秒）。

        公式: Total = Store Count × Avg Items × Avg Slide Intermission
        """
        sc = store_count or self._store_count
        ai = avg_items or self._avg_items
        if sc <= 0 or ai <= 0:
            return 0.0
        return sc * ai * self._avg_slide

    def check_capacity(self) -> bool:
        """检查当前监控体量是否在安全窗口内。

        Returns:
            True 表示安全，False 表示超限需要精简。
        """
        estimated = self.estimate_duration()
        if estimated > self._max_duration:
            msg = (
                f"当前监控体量已触及单机物理极限！"
                f"预计耗时 {estimated / 3600:.1f}h，"
                f"超过安全窗口 {self._max_duration / 3600:.1f}h"
            )
            logger.warning(msg)
            for cb in self._on_capacity_warning:
                try:
                    cb(msg)
                except Exception:
                    logger.exception("容量告警回调异常")
            return False
        return True

    def update_capacity(self, store_count: int,
                         avg_items_per_store: int) -> None:
        self._store_count = store_count
        self._avg_items = avg_items_per_store

    # ── 主循环 ────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now().time()
            current_seconds = now.hour * 3600 + now.minute * 60 + now.second
            start_seconds = self._start.hour * 3600 + self._start.minute * 60
            end_seconds = self._end.hour * 3600 + self._end.minute * 60

            in_window = start_seconds <= current_seconds < end_seconds

            if in_window and not self._sweep_active.is_set():
                self._start_sweep()
            elif not in_window and self._sweep_active.is_set():
                self._end_sweep()

            # 每分钟检查一次
            time.sleep(60)

    def _start_sweep(self) -> None:
        logger.info("=== 进入跑盘窗口，启动采集任务 ===")
        self._sweep_active.set()
        for cb in self._on_sweep_start:
            try:
                cb()
            except Exception:
                logger.exception("跑盘启动回调异常")

    def _end_sweep(self) -> None:
        logger.info("=== 跑盘窗口结束，停止采集 ===")
        self._sweep_active.clear()
        for cb in self._on_sweep_end:
            try:
                cb()
            except Exception:
                logger.exception("跑盘结束回调异常")

    @property
    def is_active(self) -> bool:
        return self._sweep_active.is_set()
