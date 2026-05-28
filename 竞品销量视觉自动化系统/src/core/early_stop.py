"""24小时智能熔断截断 — 首条"天前"订单命中后立即关闭弹窗。

设计文档 4.5：
- 遍历 Qwen-VL 返回的 orders 列表
- 首次遇到 time_str 含"天前" → 抛出 STOP_SCROLL_AND_CLOSE 信号
- 今日订单正常入库，昨日订单截断不混入
"""
import logging
from enum import Enum
from typing import Optional

from .time_normalizer import TimeNormalizer

logger = logging.getLogger(__name__)


class StopSignal(Enum):
    NONE = 0
    STOP_SCROLL_AND_CLOSE = 1  # 24h 边界命中，关闭弹窗


class EarlyStopEngine:
    """24小时边界熔断引擎。

    消费者线程在收到 VLM 返回的订单列表后，
    逐条归一化时间戳，首条跨天订单触发 STOP_SCROLL_AND_CLOSE。
    """

    def __init__(self, normalizer: TimeNormalizer | None = None):
        self._normalizer = normalizer or TimeNormalizer()
        self._today_orders: list[dict] = []
        self._stop_triggered = False
        self._total_processed = 0
        self._total_stopped = 0

    def process_orders(self, orders: list[dict]) -> tuple[list[dict], StopSignal]:
        """处理一批订单，返回今日订单 + 停止信号。

        Args:
            orders: VLM 提取的原始订单列表
                    [{"buyer": ..., "sku": ..., "time_str": ...}, ...]

        Returns:
            (today_only_orders, stop_signal)
            - today_only_orders: 仅包含今日的订单（已附加 order_date 字段）
            - stop_signal: 是否应停止
        """
        self._today_orders = []
        self._stop_triggered = False

        for i, order in enumerate(orders):
            time_str = order.get("time_str", "")
            date_str, should_stop = self._normalizer.normalize(time_str)

            if should_stop:
                self._stop_triggered = True
                self._total_stopped += 1
                logger.info(
                    "24h 熔断: 第 %d 条订单 '%s' → %s，截断 %d 条后续",
                    i + 1, time_str, date_str, len(orders) - i - 1,
                )
                break  # 截断，后续订单不入库

            self._today_orders.append({
                **order,
                "order_date": date_str,
            })

        self._total_processed += len(orders)

        signal = StopSignal.STOP_SCROLL_AND_CLOSE if self._stop_triggered else StopSignal.NONE
        return self._today_orders, signal

    def should_stop(self, time_str: str) -> bool:
        """单条检查：是否应停止。

        用于快速预检，不依赖完整订单列表。
        """
        _, should_stop = self._normalizer.normalize(time_str)
        return should_stop

    def reset(self) -> None:
        """重置状态（切换单品时调用）。"""
        self._today_orders = []
        self._stop_triggered = False

    @property
    def stats(self) -> dict:
        return {
            "total_processed": self._total_processed,
            "total_stopped": self._total_stopped,
            "stopped_this_batch": self._stop_triggered,
        }
