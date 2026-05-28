"""生产者线程 — 投屏帧捕获、仿生滑动控制、背压响应。

线程 A 职责：
1. 从 scrcpy 视频管道读取实时帧
2. 控制手机执行贝塞尔曲线仿生滑动
3. 将帧压入有界阻塞队列，队列满时 WAITING
4. 响应看门狗恢复信号，重置帧捕获句柄
"""
import logging
import random
import threading
import time

from src.core.adb_controller import ADBController
from src.core.session import SessionManager
from src.pipeline.queue_manager import ImageQueue

logger = logging.getLogger(__name__)


class ProducerThread:
    """生产者线程 — 图像采集与手势控制。"""

    def __init__(self, adb: ADBController, queue: ImageQueue,
                 session_mgr: SessionManager,
                 slide_min_pause: float = 1.5, slide_max_pause: float = 4.0,
                 swipe_duration_min: int = 300, swipe_duration_max: int = 800,
                 screen_width: int = 1080, screen_height: int = 1920):
        self._adb = adb
        self._queue = queue
        self._session_mgr = session_mgr

        self._slide_min_pause = slide_min_pause
        self._slide_max_pause = slide_max_pause
        self._swipe_duration_min = swipe_duration_min
        self._swipe_duration_max = swipe_duration_max
        self._screen_width = screen_width
        self._screen_height = screen_height

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.clear()

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=False, name="producer")
        self._thread.start()
        logger.info("生产者线程已启动")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def pause(self) -> None:
        """外部调用：暂停生产者（弹窗/验证码阻断）。"""
        self._paused.set()
        logger.info("生产者已外部暂停")

    def resume(self) -> None:
        """外部调用：恢复生产者。"""
        self._paused.clear()
        logger.info("生产者已外部恢复")

    def reset_frame_source(self) -> None:
        """看门狗回调：投屏流恢复后重置帧捕获句柄。"""
        logger.info("生产者收到帧源重置通知")
        # scrcpy 已由看门狗重新拉起，生产者继续从管道读取即可

    # ── 主循环 ────────────────────────────────────────────────

    def _run(self) -> None:
        state = self._session_mgr.state

        while not self._stop_event.is_set():
            # 异常挂起检查
            while self._paused.is_set() and not self._stop_event.is_set():
                time.sleep(0.5)

            if self._stop_event.is_set():
                break

            # 背压检查：队列满则等待
            self._queue.wait_if_paused()

            if not self._queue.should_produce:
                logger.debug("队列满载，生产者等待消费空间")
                time.sleep(0.2)
                continue

            # 截取当前帧
            frame = self._capture_current_frame()
            if frame is None:
                time.sleep(1)
                continue

            # 压入队列（队列满则阻塞）
            success = self._queue.put(frame)
            if not success:
                # 超时未入队，降级：跳过本帧，稍后重试
                logger.warning("入队超时，降级跳帧")
                continue

            # 执行一次仿生滑动
            self._perform_slide()

            # 随机停顿
            pause = random.uniform(self._slide_min_pause, self._slide_max_pause)
            time.sleep(pause)

    def _capture_current_frame(self):
        """从 scrcpy 管道捕获一帧。

        当前为占位实现：有 numpy 时生成空白帧数组，无 numpy 时（测试/CI）
        返回占位字节对象，保证生产者-消费者链路可运行。
        阶段二接入真实 H.264 解码时替换。
        """
        try:
            import numpy as np
            return np.zeros((self._screen_height, self._screen_width, 3), dtype=np.uint8)
        except ImportError:
            return {"placeholder": True, "size": f"{self._screen_width}x{self._screen_height}"}

    def _perform_slide(self) -> None:
        """执行一次仿生上滑。"""
        # 从屏幕中下部滑向中上部，模拟翻页
        x_center = self._screen_width // 2
        y_start = int(self._screen_height * 0.75)
        y_end = int(self._screen_height * 0.25)

        # 加少量随机偏移
        x_center += random.randint(-30, 30)
        y_start += random.randint(-20, 20)
        y_end += random.randint(-20, 20)

        duration = random.randint(self._swipe_duration_min, self._swipe_duration_max)

        try:
            self._adb.bezier_swipe(x_center, y_start, x_center, y_end, duration)
        except Exception:
            logger.exception("滑动手势执行失败")
