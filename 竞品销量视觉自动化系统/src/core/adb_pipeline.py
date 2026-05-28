"""ADB 持久化 Shell 管道 — 消除高频 fork 开销。

设计文档 6.1.2：
- 严禁高频开关 adb.exe 进程
- 初始化时建立持久化 subprocess.Popen 管道
- 通过 stdin.write 持续注入指令
- stdin 写入受 threading.Lock 保护，防止多线程交织
"""
import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


class ADBPipeline:
    """ADB 持久化交互式 Shell 管道。

    初始化时建立一条长连接，后续所有 ADB shell 指令
    通过 stdin 写入，彻底消除进程 fork 开销。

    用法:
        pipe = ADBPipeline()
        pipe.start()
        pipe.swipe(540, 1440, 540, 480, 500)
        pipe.tap(540, 960)
        pipe.stop()
    """

    def __init__(self, adb_path: str = "adb",
                 device_serial: str | None = None):
        self._adb = adb_path
        self._serial = device_serial
        self._proc: subprocess.Popen | None = None
        self._started = False
        self._stdin_lock = threading.Lock()  # 防止多线程交织写入

    def start(self) -> bool:
        """启动持久化 ADB Shell 管道。"""
        if self._started:
            return True

        cmd = [self._adb]
        if self._serial:
            cmd.extend(["-s", self._serial])
        cmd.append("shell")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._started = True
            logger.info("ADB 持久化管道已建立")
            return True
        except Exception:
            logger.exception("ADB 管道启动失败")
            return False

    def stop(self) -> None:
        """关闭管道。"""
        if self._proc and self._started:
            try:
                self._exec("exit\n")
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.terminate()
            self._started = False
            logger.info("ADB 管道已关闭")

    # ── 手势指令 ──────────────────────────────────────────────

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 500) -> None:
        """执行滑动（无进程创建开销）。"""
        cmd = f"input swipe {x1} {y1} {x2} {y2} {duration_ms}\n"
        self._exec(cmd)

    def tap(self, x: int, y: int) -> None:
        """点击。"""
        cmd = f"input tap {x} {y}\n"
        self._exec(cmd)

    def keyevent(self, keycode: str) -> None:
        """发送按键事件。"""
        cmd = f"input keyevent {keycode}\n"
        self._exec(cmd)

    def text(self, text: str) -> None:
        """输入文本。"""
        escaped = text.replace(" ", "%s").replace("'", "\\'")
        cmd = f"input text '{escaped}'\n"
        self._exec(cmd)

    # ── 系统指令 ──────────────────────────────────────────────

    def airplane_mode(self, on: bool = True) -> None:
        """飞行模式开关。"""
        if on:
            self._exec("svc data disable\n")
            self._exec("settings put global airplane_mode_on 1\n")
        else:
            self._exec("settings put global airplane_mode_on 0\n")
            self._exec("svc data enable\n")

    def force_stop_app(self, package: str = "com.taobao.taobao") -> None:
        self._exec(f"am force-stop {package}\n")

    def start_app(self, package: str = "com.taobao.taobao",
                  activity: str = "com.taobao.tao.welcome.Welcome") -> None:
        self._exec(f"am start -n {package}/{activity}\n")

    def get_temperature(self) -> float | None:
        """读取热 Zone 温度（需要 root / KernelSU）。"""
        try:
            # 通过独立 adb 调用获取温度（管道读取 stdout 复杂）
            import subprocess as sp
            cmd = [self._adb]
            if self._serial:
                cmd.extend(["-s", self._serial])
            cmd.extend(["shell", "cat", "/sys/class/thermal/thermal_zone0/temp"])
            result = sp.run(cmd, capture_output=True, text=True, timeout=5)
            temp_str = result.stdout.strip()
            return float(temp_str) / 1000.0  # 毫度 → 度
        except Exception:
            return None

    # ── 内部 ──────────────────────────────────────────────────

    def _exec(self, cmd: str) -> None:
        if not self._started or self._proc is None:
            logger.warning("ADB 管道未启动")
            return
        try:
            with self._stdin_lock:
                self._proc.stdin.write(cmd)
                self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.warning("ADB 管道断开，尝试重连")
            self._started = False
            if self.start():
                with self._stdin_lock:
                    self._proc.stdin.write(cmd)
                    self._proc.stdin.flush()
