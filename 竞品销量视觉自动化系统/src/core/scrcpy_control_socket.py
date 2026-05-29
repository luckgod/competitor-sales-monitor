"""Scrcpy Control Socket 直连 — 替代 ADB Shell 高频输入。

设计文档 6.1.3：
- 接管 Scrcpy 本地控制端口（Control Socket）
- 通过 TCP 长连接直接发送二进制控制消息
- 输入延迟 300ms → < 1ms，零 fork 开销

Scrcpy Control Protocol 基础消息格式:
- inject_text:   type=2, text bytes
- inject_touch:  type=0, action, pointer_id, position, pressure, ...
"""
import logging
import socket
import struct
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Scrcpy 控制消息类型
CONTROL_TYPE_INJECT_KEYCODE = 0
CONTROL_TYPE_INJECT_TEXT = 2
CONTROL_TYPE_INJECT_TOUCH_EVENT = 3
CONTROL_TYPE_INJECT_SCROLL_EVENT = 4
CONTROL_TYPE_BACK_OR_SCREEN_ON = 5
CONTROL_TYPE_EXPAND_NOTIFICATION_PANEL = 7
CONTROL_TYPE_COLLAPSE_NOTIFICATION_PANEL = 9
CONTROL_TYPE_SET_CLIPBOARD = 10
CONTROL_TYPE_SET_SCREEN_POWER_MODE = 11

# Touch 动作
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2


class ScrcpyControlSocket:
    """Scrcpy 控制协议直连 — 持久 TCP 长连接。

    用法:
        ctrl = ScrcpyControlSocket(port=27183)
        ctrl.connect()
        ctrl.inject_text("江西三支一扶")
        ctrl.touch(540, 960, ACTION_DOWN)
        ctrl.close()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 27183):
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._connected = False

    def connect(self) -> bool:
        """建立持久 TCP 连接。"""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5)
            self._sock.connect((self._host, self._port))
            self._connected = True
            logger.info("Scrcpy Control Socket 已连接: %s:%d",
                        self._host, self._port)
            return True
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            logger.warning("Control Socket 连接失败: %s", e)
            return False

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._connected = False

    # ── 文本注入 ──────────────────────────────────────────────

    def inject_text(self, text: str) -> bool:
        """直接注入文本（延迟 < 1ms，零 fork 开销）。

        替代 adb shell input text，支持中文 UTF-8。
        """
        return self._send(ControlMessage.inject_text(text))

    # ── 触控事件 ──────────────────────────────────────────────

    def touch(self, x: int, y: int, action: int = ACTION_DOWN,
              pressure: float = 0.6, pointer_id: int = 0) -> bool:
        """注入触控事件。

        Args:
            x, y: 屏幕像素坐标
            action: ACTION_DOWN / ACTION_MOVE / ACTION_UP
            pressure: 触控压力 (0.0 ~ 1.0)，模拟真实手指
        """
        return self._send(ControlMessage.inject_touch(
            action, pointer_id, x, y,
            int(0xFFFF * min(1.0, max(0.0, pressure))),
        ))

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 500, steps: int = 10) -> None:
        """高频插值滑动（Control Socket 直连，无 ADB fork 开销）。"""
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            action = ACTION_DOWN if i == 0 else (ACTION_UP if i == steps else ACTION_MOVE)
            pressure = 0.3 if i == 0 else (0.2 if i == steps else 0.5 + 0.3 * (1 - abs(2 * t - 1)))
            self.touch(x, y, action, pressure)
            if i < steps:
                time.sleep(duration_ms / 1000 / steps)

    # ── 按键 ──────────────────────────────────────────────────

    def back(self) -> bool:
        """物理返回键。"""
        return self._send(ControlMessage.back_or_screen_on(
            ControlMessage.ACTION_BACK,
        ))

    def home(self) -> bool:
        """Home 键。"""
        return self.inject_keycode(3)  # KEYCODE_HOME

    def inject_keycode(self, keycode: int, repeat: int = 0) -> bool:
        """注入按键码。"""
        return self._send(ControlMessage.inject_keycode(keycode, repeat))

    # ── 内部 ──────────────────────────────────────────────────

    def _send(self, data: bytes) -> bool:
        if not self._connected or self._sock is None:
            return False
        try:
            self._sock.sendall(data)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning("Control Socket 发送失败: %s", e)
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected


class ControlMessage:
    """Scrcpy 控制协议二进制消息构造器。"""

    ACTION_BACK = 1

    @staticmethod
    def inject_text(text: str) -> bytes:
        buf = text.encode("utf-8")
        return (
            struct.pack(">BB", CONTROL_TYPE_INJECT_TEXT, 0) +
            struct.pack(">I", len(buf)) + buf
        )

    @staticmethod
    def inject_touch(action: int, pointer_id: int,
                     x: int, y: int, pressure: int) -> bytes:
        return (
            struct.pack(">BB", CONTROL_TYPE_INJECT_TOUCH_EVENT, action) +
            struct.pack(">Q", pointer_id) +
            struct.pack(">III", x, y, 0xFFFF) +  # position + screen size
            struct.pack(">H", pressure) +
            struct.pack(">I", ACTION_DOWN if action == ACTION_DOWN else
                        (ACTION_UP if action == ACTION_UP else ACTION_MOVE))
        )

    @staticmethod
    def inject_keycode(keycode: int, repeat: int = 0) -> bytes:
        return (
            struct.pack(">BB", CONTROL_TYPE_INJECT_KEYCODE, 0) +
            struct.pack(">I", keycode) +
            struct.pack(">I", repeat)
        )

    @staticmethod
    def back_or_screen_on(action: int) -> bytes:
        return struct.pack(">BB", CONTROL_TYPE_BACK_OR_SCREEN_ON, action)
