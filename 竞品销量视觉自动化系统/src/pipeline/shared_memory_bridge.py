"""进程沙箱零拷贝优化 — multiprocessing.shared_memory 图像传递。

设计文档 6.9.1：
- 废弃 multiprocessing.Queue 传图（pickle 序列化吃满 CPU）
- 主进程在共享内存中写入帧，子进程零拷贝读取
- 仅传递内存名称 + 偏移量，CPU 序列化开销归零
"""
import logging
import time
from multiprocessing import shared_memory
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 1080P BGR 单帧体积: 1920 * 1080 * 3 = 6,220,800 bytes ≈ 6MB
FRAME_SIZE_BYTES = 1920 * 1080 * 3


class SharedMemoryBridge:
    """共享内存零拷贝桥梁。

    主进程创建共享内存并写入帧，子进程通过名称附加到同一块物理内存。
    只传递 (name, frame_id) 元组，无序列化开销。

    用法（主进程）:
        bridge = SharedMemoryBridge()
        bridge.create()
        bridge.write_frame(frame, frame_id=42)
        # 通过轻量 Queue 传递 (name, frame_id) 给子进程

    用法（子进程）:
        bridge = SharedMemoryBridge()
        bridge.attach(shm_name)
        frame = bridge.read_frame()
    """

    def __init__(self, name: str = "vlm_frame_buffer", size: int = FRAME_SIZE_BYTES):
        self._name = name
        self._size = size
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._buffer: Optional[np.ndarray] = None

    def create(self) -> str:
        """主进程：创建共享内存块。

        Returns:
            共享内存名称（传递给子进程）。
        """
        # 销毁旧的同名共享内存（防止残留）
        try:
            old = shared_memory.SharedMemory(name=self._name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass

        self._shm = shared_memory.SharedMemory(
            name=self._name, create=True, size=self._size,
        )
        self._buffer = np.ndarray(
            (1080, 1920, 3), dtype=np.uint8, buffer=self._shm.buf,
        )
        logger.info("共享内存已创建: %s (%d bytes)", self._name, self._size)
        return self._name

    def attach(self, name: str) -> None:
        """子进程：附加到已存在的共享内存（零拷贝）。"""
        self._name = name
        self._shm = shared_memory.SharedMemory(name=name)
        self._buffer = np.ndarray(
            (1080, 1920, 3), dtype=np.uint8, buffer=self._shm.buf,
        )
        logger.debug("共享内存已附加: %s", name)

    def write_frame(self, frame: np.ndarray, frame_id: int = 0) -> None:
        """主进程：将帧写入共享内存（直接内存拷贝，无序列化）。

        Args:
            frame: (H, W, 3) BGR uint8 图像
            frame_id: 帧序号（用于子进程判断是否为新帧）
        """
        if self._buffer is None:
            raise RuntimeError("共享内存未创建，请先调用 create()")
        h, w = frame.shape[:2]
        self._buffer[:h, :w] = frame
        self._buffer.flags.writeable = True  # 确保可写

    def read_frame(self) -> np.ndarray:
        """子进程：零拷贝读取当前帧视图。"""
        if self._buffer is None:
            raise RuntimeError("共享内存未附加，请先调用 attach()")
        return self._buffer.copy()  # 返回拷贝防止主进程覆盖

    def close(self) -> None:
        """关闭共享内存引用（不销毁物理内存）。"""
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._buffer = None

    def unlink(self) -> None:
        """主进程：强制物理销毁共享内存，防止 Windows 僵尸悬空泄漏。

        必须在 try...finally 或全局异常钩子中调用，
        确保子进程崩溃后内存不被永久占用。
        """
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("共享内存销毁异常: %s", e)
            self._shm = None
            self._buffer = None
            logger.info("共享内存已物理销毁: %s", self._name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def size_mb(self) -> float:
        return self._size / (1024 * 1024)
