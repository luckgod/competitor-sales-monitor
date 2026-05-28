"""数据高可用与异常恢复 — 时序空洞拟合 + 二次补抓调度。

设计规范（来自设计文档 10.1 / 10.2）：
- 时序空洞：线性插值填补缺失快照 → Value_target = (Value_date-1 + Value_date+1) / 2
- 二次调度：每日 8:30 盘点数据完整率，< 90% 触发补抓
- 补抓标记：session_label = "Replenish_Snapshot"
"""
import logging
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class DataHealer:
    """数据修复器 — 时序空洞拟合 + 补抓判定。"""

    def __init__(self, repo, completeness_threshold: float = 0.90,
                 replenish_check_time: str = "08:30"):
        self._repo = repo
        self._threshold = completeness_threshold
        self._check_time = replenish_check_time

    def interpolate_timeline(self, virtual_id: str,
                              days: int = 30) -> list[dict]:
        """获取商品时间线，对缺失点执行线性插值。

        Returns:
            包含插值填充的完整时间序列，
            每条记录含 is_interpolated 标记。
        """
        raw = self._repo.get_product_timeline(virtual_id, days)
        if not raw:
            return []

        # 建立日期 → 值的映射
        by_date = {}
        for r in raw:
            d = r["record_date"]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            by_date[d] = r["snapshot_rolling_sales"]

        # 生成完整日期序列
        today = date.today()
        result = []
        for i in range(days):
            d = today - timedelta(days=days - 1 - i)
            if d in by_date:
                result.append({
                    "record_date": d.isoformat(),
                    "snapshot_rolling_sales": by_date[d],
                    "is_interpolated": False,
                })
            else:
                # 线性插值
                value = self._linear_interpolate(by_date, d)
                result.append({
                    "record_date": d.isoformat(),
                    "snapshot_rolling_sales": value,
                    "is_interpolated": True,
                })

        return result

    @staticmethod
    def _linear_interpolate(by_date: dict, target_date: date) -> Optional[int]:
        """线性插值：Value = (前一日 + 后一日) / 2。"""
        before = None
        after = None

        for offset in range(1, 60):
            d_before = target_date - timedelta(days=offset)
            if d_before in by_date:
                before = by_date[d_before]
                break

        for offset in range(1, 60):
            d_after = target_date + timedelta(days=offset)
            if d_after in by_date:
                after = by_date[d_after]
                break

        if before is not None and after is not None:
            return int((before + after) / 2)
        if before is not None:
            return before
        if after is not None:
            return after
        return 0

    def needs_replenishment(self, target_date: date) -> bool:
        """检查是否需要二次补抓。

        Returns:
            True 表示数据完整率低于 90%，需要补抓。
        """
        completeness = self._repo.get_data_completeness(target_date)
        rate = completeness["completeness"]
        if rate < self._threshold:
            logger.warning(
                "数据完整率 %.1f%% < %.0f%%，触发二次补抓",
                rate * 100, self._threshold * 100,
            )
            return True
        return False

    def should_replenish_now(self) -> bool:
        """判断当前时间是否为补抓检查点（8:30）。"""
        now = datetime.now()
        h, m = map(int, self._check_time.split(":"))
        return now.hour == h and now.minute >= m and now.minute < m + 10
