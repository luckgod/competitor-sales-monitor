"""零点跨天时间清洗状态机 — 任务锚点 + 级联解析 + 批次隔离。

设计文档 4.5.1：
- 会话初始化时锁定任务启动的绝对时间锚点
- 级联时间解析矩阵处理跨天边缘情况
- capture_batch_id 实行断点批次隔离
"""
import logging
import uuid
from datetime import datetime
from typing import Optional

from .time_normalizer import TimeNormalizer

logger = logging.getLogger(__name__)


class TemporalStateMachine:
    """全时段时序状态机。

    管理任务级时间锚点、批次标识和跨天安全解析。
    """

    def __init__(self):
        self._batch_id: str = ""
        self._task_start_epoch_ms: float = 0.0
        self._normalizer: Optional[TimeNormalizer] = None
        self._active = False

    def initialize(self, batch_id: str | None = None) -> str:
        """初始化时序状态机。

        在会话初始化时调用，锁定当前绝对时间锚点。

        Args:
            batch_id: 可选的自定义批次 ID

        Returns:
            capture_batch_id
        """
        now_ms = datetime.now().timestamp() * 1000
        self._task_start_epoch_ms = now_ms
        self._batch_id = batch_id or uuid.uuid4().hex[:12]
        self._normalizer = TimeNormalizer(task_start_epoch_ms=now_ms)
        self._active = True

        logger.info(
            "时序状态机初始化: batch=%s, anchor=%s",
            self._batch_id,
            datetime.fromtimestamp(now_ms / 1000).isoformat(),
        )
        return self._batch_id

    def normalize(self, time_str: str) -> tuple[str, bool]:
        """基于任务锚点归一化时间。

        Returns:
            (YYYY-MM-DD, should_stop)
        """
        if not self._active or self._normalizer is None:
            # 未初始化，降级到当前时间
            fallback = TimeNormalizer()
            return fallback.normalize(time_str)

        return self._normalizer.normalize(time_str)

    def is_cross_midnight(self) -> bool:
        """检测当前是否跨越了午夜（锚点日期 != 当前日期）。"""
        if not self._active:
            return False

        anchor_date = datetime.fromtimestamp(
            self._task_start_epoch_ms / 1000
        ).date()
        today = datetime.now().date()

        if anchor_date != today:
            logger.warning(
                "检测到跨天运行: 锚点=%s, 当前=%s",
                anchor_date.isoformat(), today.isoformat(),
            )
            return True
        return False

    def get_cross_midnight_offset_days(self) -> int:
        """获取跨天偏移天数（锚点距今多少天）。"""
        if not self._active:
            return 0
        anchor_date = datetime.fromtimestamp(
            self._task_start_epoch_ms / 1000
        ).date()
        today = datetime.now().date()
        return (today - anchor_date).days

    @property
    def batch_id(self) -> str:
        return self._batch_id

    @property
    def anchor_iso(self) -> str:
        if not self._active:
            return ""
        return datetime.fromtimestamp(
            self._task_start_epoch_ms / 1000
        ).isoformat()

    @property
    def is_active(self) -> bool:
        return self._active

    def deactivate(self) -> None:
        """停用时序状态机（任务结束后调用）。"""
        self._active = False
        logger.info("时序状态机已停用 (batch=%s)", self._batch_id)
