"""ADB 协议与 scrcpy 投屏引擎控制 — 物理层通信总控。"""
import logging
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    serial: str
    model: str = ""
    android_version: str = ""


class ADBController:
    """封装 ADB 指令与 scrcpy 进程生命周期管理。"""

    def __init__(self, adb_path: str = "adb", scrcpy_path: str = "scrcpy",
                 device_serial: str | None = None):
        self._adb = adb_path
        self._scrcpy = scrcpy_path
        self._serial = device_serial
        self._scrcpy_proc: subprocess.Popen | None = None

    # ── 设备检测 ──────────────────────────────────────────────

    def list_devices(self) -> list[DeviceInfo]:
        """返回当前 ADB 连接的设备列表。"""
        result = self._run_adb(["devices", "-l"], capture=True)
        devices = []
        for line in result.splitlines()[1:]:
            if not line.strip() or "offline" in line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                serial = parts[0]
                model = ""
                for p in parts:
                    if p.startswith("model:"):
                        model = p.split(":", 1)[1]
                devices.append(DeviceInfo(serial=serial, model=model))
        return devices

    def is_device_connected(self) -> bool:
        return len(self.list_devices()) > 0

    def wait_for_device(self, timeout: int = 60) -> bool:
        """阻塞等待设备就绪，返回 True 表示成功。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_device_connected():
                return True
            time.sleep(1)
        return False

    def reconnect(self, device_ip: str | None = None) -> bool:
        """无线 ADB 兜底重连。"""
        self._run_adb(["kill-server"])
        time.sleep(1)
        self._run_adb(["start-server"])
        if device_ip:
            self._run_adb(["connect", device_ip], timeout=10)
        return self.wait_for_device()

    # ── scrcpy 投屏 ───────────────────────────────────────────

    def launch_scrcpy(self, max_size: int = 1080, max_fps: int = 15,
                      bit_rate: str = "8M", stay_awake: bool = True) -> subprocess.Popen:
        """启动 scrcpy 视频流进程，返回 Popen 对象。"""
        self.kill_scrcpy()
        args = [
            self._scrcpy,
            f"--max-size={max_size}",
            f"--max-fps={max_fps}",
            f"--video-bit-rate={bit_rate}",
            "--no-audio",
            "--no-window",
            "--no-playback",
            "--no-control",
            "--raw-video-stream",
        ]
        if stay_awake:
            args.append("--stay-awake")
        if self._serial:
            args.extend(["--serial", self._serial])

        logger.info("启动 scrcpy: %s", " ".join(args))
        self._scrcpy_proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return self._scrcpy_proc

    def kill_scrcpy(self) -> None:
        if self._scrcpy_proc is not None:
            try:
                self._scrcpy_proc.terminate()
                self._scrcpy_proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                self._scrcpy_proc.kill()
            self._scrcpy_proc = None

    @property
    def scrcpy_alive(self) -> bool:
        if self._scrcpy_proc is None:
            return False
        return self._scrcpy_proc.poll() is None

    # ── 屏幕控制 ──────────────────────────────────────────────

    def screen_on(self) -> None:
        self._run_adb(["shell", "input", "keyevent", "KEYCODE_WAKEUP"])

    def screen_off(self) -> None:
        self._run_adb(["shell", "input", "keyevent", "KEYCODE_SLEEP"])

    def is_screen_on(self) -> bool:
        result = self._run_adb(
            ["shell", "dumpsys", "power"], capture=True
        )
        return "mWakefulness=Awake" in result or "Display Power: state=ON" in result

    def keep_screen_on(self) -> None:
        """设置充电时屏幕永不休眠。"""
        self._run_adb(["shell", "svc", "power", "stayon", "true"])

    # ── 应用管理 ──────────────────────────────────────────────

    def force_stop_app(self, package: str = "com.taobao.taobao") -> None:
        self._run_adb(["shell", "am", "force-stop", package])

    def start_app(self, package: str = "com.taobao.taobao",
                  activity: str = "com.taobao.tao.welcome.Welcome") -> None:
        self._run_adb(["shell", "am", "start", "-n", f"{package}/{activity}"])

    def clear_app_data(self, package: str = "com.taobao.taobao") -> None:
        self._run_adb(["shell", "pm", "clear", package])

    # ── 仿生手势 ──────────────────────────────────────────────

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 500) -> None:
        """执行直线滑动。"""
        self._run_adb([
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2),
            str(duration_ms),
        ])

    def bezier_swipe(self, x1: int, y1: int, x2: int, y2: int,
                     duration_ms: int | None = None,
                     jitter: int = 15) -> None:
        """贝塞尔曲线仿生滑动 — 模拟人类手指的加速/减速/抖动轨迹。

        通过分段 ADB swipe 命令组合成一条带随机偏移的曲线路径。
        每段耗时按缓入缓出分布。
        """
        if duration_ms is None:
            duration_ms = random.randint(300, 800)

        segments = random.randint(3, 6)
        jitter_x = random.randint(-jitter, jitter)
        jitter_y = random.randint(-jitter, jitter)
        cx = (x1 + x2) // 2 + jitter_x
        cy = (y1 + y2) // 2 + jitter_y

        total_weight = 0.0
        for i in range(segments):
            t = (i + 1) / segments
            # 二次贝塞尔: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
            px = int((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2)
            py = int((1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2)
            px += random.randint(-3, 3)
            py += random.randint(-3, 3)
            # 缓入缓出权重分配
            w = 1.0 - abs(2 * t - 1) + 0.3
            total_weight += w

        prev_x, prev_y = x1, y1
        for i in range(segments):
            t = (i + 1) / segments
            px = int((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2)
            py = int((1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2)
            px += random.randint(-3, 3)
            py += random.randint(-3, 3)
            w = 1.0 - abs(2 * t - 1) + 0.3
            seg_ms = int(duration_ms * w / total_weight)
            self.swipe(prev_x, prev_y, px, py, max(seg_ms, 30))
            prev_x, prev_y = px, py

    def capture_screenshot(self, output_path: str) -> bool:
        """截取当前屏幕并拉取到本地。"""
        remote_path = "/sdcard/screenshot_tmp.png"
        self._run_adb(["shell", "screencap", "-p", remote_path])
        result = self._run_adb(["pull", remote_path, output_path], capture=True)
        self._run_adb(["shell", "rm", remote_path])
        return "pulled" in result.lower() or Path(output_path).exists()

    # ── 内部工具 ──────────────────────────────────────────────

    def _adb_prefix(self) -> list[str]:
        if self._serial:
            return [self._adb, "-s", self._serial]
        return [self._adb]

    def _run_adb(self, args: list[str], capture: bool = True,
                 timeout: int = 15) -> str:
        cmd = self._adb_prefix() + args
        try:
            result = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            logger.warning("ADB 命令超时: %s", " ".join(cmd))
            return ""
        except FileNotFoundError:
            logger.error("ADB 未找到，请确认 adb 已加入 PATH")
            return ""
