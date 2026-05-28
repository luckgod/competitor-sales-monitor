"""看门狗守护线程 — 监控 scrcpy/ADB 健康状态，检测断连后自动执行热插拔自愈。

职责：
1. 每 5 秒轮询 scrcpy 进程存活与 ADB 设备连接状态
2. 检测到异常后自动 kill-server → start-server → wait-for-device → relaunch scrcpy
3. 通知生产者线程重置帧捕获句柄
4. 连续 3 次失败触发远程告警
"""
import logging
import threading
import time
from typing import Callable

from .adb_controller import ADBController

logger = logging.getLogger(__name__)


class Watchdog:
    """独立守护线程 — 进程级 ADB/scrcpy 投屏流热插拔与状态自愈。"""

    def __init__(self, adb: ADBController, check_interval: float = 5.0,
                 reconnect_retries: int = 3, wireless_fallback: bool = True,
                 device_ip: str | None = None):
        self._adb = adb
        self._check_interval = check_interval
        self._reconnect_retries = reconnect_retries
        self._wireless_fallback = wireless_fallback
        self._device_ip = device_ip

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._on_recovered_callbacks: list[Callable[[], None]] = []
        self._on_alert_callbacks: list[Callable[[str], None]] = []

        self._consecutive_failures = 0
        self._lock = threading.Lock()

    # ── 回调注册 ──────────────────────────────────────────────

    def on_recovered(self, callback: Callable[[], None]) -> None:
        """注册恢复回调 — 投屏流恢复后通知生产者重置帧句柄。"""
        self._on_recovered_callbacks.append(callback)

    def on_alert(self, callback: Callable[[str], None]) -> None:
        """注册告警回调 — 连续失败时触发远程通知。"""
        self._on_alert_callbacks.append(callback)

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()
        logger.info("看门狗线程已启动，检查间隔 %.1fs", self._check_interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ── 主循环 ────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.wait(self._check_interval):
            try:
                self._check_and_heal()
            except Exception:
                logger.exception("看门狗检查周期异常")

    def _check_and_heal(self) -> None:
        scrcpy_ok = self._adb.scrcpy_alive
        device_ok = self._adb.is_device_connected()

        if scrcpy_ok and device_ok:
            with self._lock:
                self._consecutive_failures = 0
            return

        logger.warning(
            "检测到异常 — scrcpy 存活:%s, 设备在线:%s, 开始自愈流程",
            scrcpy_ok, device_ok,
        )

        if not device_ok:
            self._recover_device()
        if not self._adb.scrcpy_alive:
            self._recover_scrcpy()

        if self._adb.scrcpy_alive and self._adb.is_device_connected():
            with self._lock:
                self._consecutive_failures = 0
            logger.info("自愈成功，投屏流已恢复")
            self._notify_recovered()
        else:
            with self._lock:
                self._consecutive_failures += 1
            if self._consecutive_failures >= self._reconnect_retries:
                self._send_alert("ADB/scrcpy 连续自愈失败，已达告警阈值")

    def _recover_device(self) -> None:
        """恢复 ADB 设备连接。"""
        logger.info("执行 ADB 重连: kill-server → start-server")
        self._adb._run_adb(["kill-server"])
        time.sleep(1)
        self._adb._run_adb(["start-server"])

        if self._wireless_fallback and self._device_ip:
            logger.info("尝试无线 ADB 兜底: connect %s", self._device_ip)
            self._adb._run_adb(["connect", self._device_ip], timeout=10)

        if self._adb.wait_for_device():
            logger.info("设备已重新就绪")
        else:
            logger.error("等待设备超时")

    def _recover_scrcpy(self) -> None:
        """重新拉起 scrcpy。"""
        self._adb.kill_scrcpy()
        time.sleep(1)
        self._adb.launch_scrcpy()

    # ── 通知 ──────────────────────────────────────────────────

    def _notify_recovered(self) -> None:
        for cb in self._on_recovered_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("恢复回调执行失败")

    def _send_alert(self, message: str) -> None:
        logger.error("发送告警: %s", message)
        for cb in self._on_alert_callbacks:
            try:
                cb(message)
            except Exception:
                logger.exception("告警回调执行失败")
