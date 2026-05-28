"""时间归一化引擎 — 相对时间文本 → 绝对日期 YYYY-MM-DD。

设计文档 4.5：
- 将 "15小时前"、"3分钟前"、"1天前" 等相对时间转换为绝对日期
- 基于任务启动锚点（Epoch ms）执行级联解析
- 支持跨天场景的正确归属
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# 时间表达式正则
RE_HOURS_AGO = re.compile(r'(\d+)\s*小时前')
RE_MINUTES_AGO = re.compile(r'(\d+)\s*分钟前')
RE_DAYS_AGO = re.compile(r'(\d+)\s*天前')
RE_JUST_NOW = re.compile(r'刚刚|几秒前|片刻前')
RE_TODAY_TIME = re.compile(r'今天\s*(\d{1,2}:\d{2})')


class TimeNormalizer:
    """相对时间 → 绝对日期转换器。

    基于任务启动锚点时间戳，将电商平台显示的
    "X小时前" / "X分钟前" / "X天前" 等文本归一化为 YYYY-MM-DD。
    """

    def __init__(self, task_start_epoch_ms: Optional[float] = None):
        """
        Args:
            task_start_epoch_ms: 任务启动的绝对时间锚点（毫秒）。
                                 为 None 时使用当前时间。
        """
        self._anchor_ms = task_start_epoch_ms or (datetime.now().timestamp() * 1000)
        self._anchor_dt = datetime.fromtimestamp(self._anchor_ms / 1000)

    @property
    def anchor(self) -> datetime:
        return self._anchor_dt

    @property
    def anchor_ms(self) -> float:
        return self._anchor_ms

    def normalize(self, time_str: str) -> tuple[str, bool]:
        """归一化时间文本。

        Args:
            time_str: 原始时间文本，如 "15小时前"、"1天前"

        Returns:
            (YYYY-MM-DD 日期字符串, should_stop 是否触发 24h 熔断)
        """
        if not time_str:
            return self._anchor_dt.strftime('%Y-%m-%d'), False

        text = time_str.strip()

        # 1. "X天前" → 触发熔断
        if match := RE_DAYS_AGO.search(text):
            days = int(match.group(1))
            dt = self._anchor_dt - timedelta(days=days)
            logger.info("归一化: '%s' → %s (触发熔断)", text, dt.strftime('%Y-%m-%d'))
            return dt.strftime('%Y-%m-%d'), True

        # 2. "今天 HH:MM"
        if match := RE_TODAY_TIME.search(text):
            return self._anchor_dt.strftime('%Y-%m-%d'), False

        # 3. "X小时前"
        if match := RE_HOURS_AGO.search(text):
            hours = int(match.group(1))
            dt = self._anchor_dt - timedelta(hours=hours)
            return dt.strftime('%Y-%m-%d'), False

        # 4. "X分钟前"
        if match := RE_MINUTES_AGO.search(text):
            minutes = int(match.group(1))
            dt = self._anchor_dt - timedelta(minutes=minutes)
            return dt.strftime('%Y-%m-%d'), False

        # 5. "刚刚" / "几秒前"
        if RE_JUST_NOW.search(text):
            return self._anchor_dt.strftime('%Y-%m-%d'), False

        # 6. 无法解析，返回锚点日期
        logger.debug("无法解析时间文本: '%s'，使用锚点日期", text)
        return self._anchor_dt.strftime('%Y-%m-%d'), False

    def normalize_batch(self, time_strs: list[str]) -> list[tuple[str, bool]]:
        """批量归一化。

        Returns:
            [(日期, should_stop), ...]
        """
        return [self.normalize(ts) for ts in time_strs]

    @property
    def today_str(self) -> str:
        return self._anchor_dt.strftime('%Y-%m-%d')
