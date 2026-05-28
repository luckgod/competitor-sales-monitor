"""进程动态生命周期管理 — Windows 堆内存碎片物理回收。

设计文档 6.8（V5.0 优化）：
- Windows HeapAlloc 碎片回收保守 → 假性内存泄漏
- 每完成一个店铺，销毁子进程并 Fork 新进程
- OS 级进程退出强制物理回收所有顽固内存
"""
import logging
import multiprocessing as mp
import signal
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class StoreProcessManager:
    """店铺级进程生命周期管理。

    每完成一个店铺的采集，主动终止旧子进程并 Fork 全新子进程，
    利用 OS 进程退出强制回收 Windows 堆管理器残留的内存碎片。

    用法:
        mgr = StoreProcessManager()
        for store in stores:
            mgr.run_store(store, worker_func=collect_store)
        mgr.shutdown()
    """

    def __init__(self, max_memory_mb: int = 3600):
        self._max_memory_mb = max_memory_mb
        self._current_process: Optional[mp.Process] = None
        self._store_count = 0
        self._shutdown_flag = mp.Event()

    def run_store(self, store_config: dict,
                  worker_func: Callable[[dict, mp.Event], None]) -> None:
        """运行单个店铺的采集任务。

        先销毁上一个店铺的子进程（强制 OS 回收内存），
        再 Fork 全新干净子进程。

        Args:
            store_config: 店铺配置（targets.yaml 中的条目）
            worker_func: 子进程入口函数，签名 (config, stop_event) -> None
        """
        # 先销毁旧进程，强制 OS 回收所有内存碎片
        self._terminate_current()

        # Fork 全新干净子进程
        ctx = mp.get_context("spawn")
        self._current_process = ctx.Process(
            target=self._worker_wrapper,
            args=(worker_func, store_config, self._shutdown_flag),
            name=f"store-worker-{self._store_count}",
        )
        self._current_process.start()
        self._store_count += 1

        logger.info("Fork 新子进程 (pid=%d) 处理店铺: %s",
                     self._current_process.pid,
                     store_config.get("store_name", "unknown"))

    def wait_current(self, timeout: float | None = None) -> bool:
        """等待当前子进程完成。

        Returns:
            True 表示正常完成，False 表示超时或异常。
        """
        if self._current_process is None:
            return True

        self._current_process.join(timeout=timeout)
        if self._current_process.is_alive():
            logger.warning("子进程超时，强制终止")
            self._terminate_current()
            return False

        exitcode = self._current_process.exitcode
        if exitcode != 0:
            logger.warning("子进程退出码: %d", exitcode)
            return False
        return True

    def shutdown(self) -> None:
        """关闭所有子进程。"""
        self._shutdown_flag.set()
        self._terminate_current()
        logger.info("进程管理器已关闭 (共处理 %d 个店铺)", self._store_count)

    def _terminate_current(self) -> None:
        if self._current_process is None:
            return
        if self._current_process.is_alive():
            self._current_process.terminate()
            self._current_process.join(timeout=10)
            if self._current_process.is_alive():
                self._current_process.kill()
                self._current_process.join(timeout=5)
        self._current_process = None
        # 显式触发 GC，辅助回收
        import gc
        gc.collect()

    @staticmethod
    def _worker_wrapper(func: Callable, config: dict, stop_event: mp.Event) -> None:
        """子进程包装器 — 信号隔离 + 异常捕获。"""
        # 子进程内忽略 SIGINT，由主进程管理
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            func(config, stop_event)
        except KeyboardInterrupt:
            pass
        except Exception:
            logger.exception("子进程异常退出: %s",
                             config.get("store_name", "unknown"))
        finally:
            # 子进程退出前强制 GC
            import gc
            gc.collect()

    @property
    def store_count(self) -> int:
        return self._store_count

    @property
    def is_running(self) -> bool:
        return self._current_process is not None and self._current_process.is_alive()
