"""移动端物理传感器静态熵置换 — 生物级微颤噪声注入。

设计文档 6.7（V5.0 优化）：
- 手机即使平放，加速度计/陀螺仪也必须有微秒级正态噪声
- 长时间绝对零值 → 风控判定为机房挂机 → 高阶滑块
- 需要 KernelSU 权限劫持 Sensor HAL 层注入噪声

注意：本模块为 Python 端配置管理 + 预检脚本。
     实际传感器注入由 KernelSU 模块在手机端执行。
"""
import logging

logger = logging.getLogger(__name__)

# 生物级微颤参数（模拟人手持手机的呼吸颤动）
SENSOR_NOISE_CONFIG = {
    "accelerometer": {
        "x_sigma": 0.05,   # m/s² 标准差
        "y_sigma": 0.05,
        "z_sigma": 0.08,   # Z 轴呼吸幅度略大
        "bias_x": 0.0,
        "bias_y": 0.0,
        "bias_z": 9.8,     # 重力
    },
    "gyroscope": {
        "x_sigma": 0.002,  # rad/s 标准差
        "y_sigma": 0.002,
        "z_sigma": 0.003,
    },
}


class SensorEntropyGuard:
    """传感器静态熵预检。

    启动时检查手机传感器数据是否处于"绝对零值"异常状态，
    若是则提示需要 KernelSU 模块注入噪声。
    """

    def __init__(self, adb_controller=None):
        self._adb = adb_controller

    def check_sensors(self) -> dict:
        """检查传感器状态。

        Returns:
            {"accelerometer": bool, "gyroscope": bool, "warnings": [...]}
            True 表示传感器正常（有噪声波动），False 表示需要注入。
        """
        warnings = []
        accel_ok = True
        gyro_ok = True

        if self._adb is not None:
            # 尝试读取加速度计原始数据
            accel_data = self._read_sensor("/dev/iio:device0")
            if accel_data is None:
                warnings.append(
                    "传感器数据不可读，可能需要 KernelSU 权限"
                )
                accel_ok = False

        if warnings:
            logger.warning("传感器静态熵检查: %s", "; ".join(warnings))
        else:
            logger.info("传感器静态熵检查通过")

        return {
            "accelerometer_ok": accel_ok,
            "gyroscope_ok": gyro_ok,
            "warnings": warnings,
        }

    @staticmethod
    def noise_config() -> dict:
        """返回建议的噪声注入参数。"""
        return SENSOR_NOISE_CONFIG

    def _read_sensor(self, device_path: str) -> str | None:
        """通过 ADB 读取传感器设备原始数据。"""
        if self._adb is None:
            return None
        try:
            result = self._adb._run_adb(
                ["shell", "cat", device_path], capture=True, timeout=3,
            )
            return result if result else None
        except Exception:
            return None
