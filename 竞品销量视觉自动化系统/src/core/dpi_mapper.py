"""多机型 DPI 动态映射 — 跨层坐标转换矩阵。

设计文档 6.1.1：
- Linux 触控板绝对坐标系与 Android Viewport 像素坐标系非 1:1
- 通过 getevent -p 查询触控边界，构建仿射转换矩阵
"""
import logging
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DPIConfig:
    """触控设备 DPI 配置。"""
    device_path: str = "/dev/input/event1"
    abs_max_x: int = 32767
    abs_max_y: int = 32767
    screen_width: int = 1080
    screen_height: int = 1920


class DPIMapper:
    """跨层坐标转换矩阵。

    将 Viewport 像素坐标 (X_pixel, Y_pixel) 转换为
    底层触控绝对坐标 (X_abs, Y_abs)。

    用法:
        mapper = DPIMapper()
        mapper.detect()           # 通过 ADB 自动检测
        x_abs, y_abs = mapper.to_abs(540, 960)
    """

    def __init__(self, config: DPIConfig | None = None):
        self._config = config or DPIConfig()

    def detect(self, adb_path: str = "adb",
               device_serial: str | None = None) -> DPIConfig:
        """通过 ADB getevent -p 动态检测触控设备参数。

        Returns:
            检测到的 DPIConfig
        """
        cmd = [adb_path]
        if device_serial:
            cmd.extend(["-s", device_serial])
        cmd.extend(["shell", "getevent", "-p", self._config.device_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout

            # 解析 ABS_MT_POSITION_X 的 max 值
            max_x = self._parse_abs_max(output, "ABS_MT_POSITION_X")
            max_y = self._parse_abs_max(output, "ABS_MT_POSITION_Y")

            if max_x and max_y:
                self._config.abs_max_x = max_x
                self._config.abs_max_y = max_y
                logger.info("DPI 检测: max_x=%d, max_y=%d", max_x, max_y)

            # 同时检测屏幕分辨率
            sz_cmd = [adb_path]
            if device_serial:
                sz_cmd.extend(["-s", device_serial])
            sz_cmd.extend(["shell", "wm", "size"])
            sz_result = subprocess.run(sz_cmd, capture_output=True, text=True, timeout=5)
            sz_match = re.search(r'(\d+)x(\d+)', sz_result.stdout)
            if sz_match:
                self._config.screen_width = int(sz_match.group(1))
                self._config.screen_height = int(sz_match.group(2))

        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("ADB 不可用，使用默认 DPI 配置")

        return self._config

    def to_abs(self, x_pixel: int, y_pixel: int) -> tuple[int, int]:
        """Viewport 像素 → 底层绝对坐标。"""
        x_abs = int(x_pixel / self._config.screen_width * self._config.abs_max_x)
        y_abs = int(y_pixel / self._config.screen_height * self._config.abs_max_y)

        # 边界裁剪
        x_abs = max(0, min(self._config.abs_max_x, x_abs))
        y_abs = max(0, min(self._config.abs_max_y, y_abs))

        return x_abs, y_abs

    def to_pixel(self, x_abs: int, y_abs: int) -> tuple[int, int]:
        """底层绝对坐标 → Viewport 像素。"""
        x_px = int(x_abs / self._config.abs_max_x * self._config.screen_width)
        y_px = int(y_abs / self._config.abs_max_y * self._config.screen_height)
        return x_px, y_px

    @property
    def config(self) -> DPIConfig:
        return self._config

    @staticmethod
    def _parse_abs_max(output: str, axis_name: str) -> int | None:
        """从 getevent -p 输出中解析指定轴的最大值。"""
        pattern = rf'{axis_name}.*?max\s+(\d+)'
        match = re.search(pattern, output, re.DOTALL)
        if match:
            return int(match.group(1))
        return None
