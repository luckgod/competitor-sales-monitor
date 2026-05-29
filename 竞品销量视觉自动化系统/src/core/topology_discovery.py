"""双引擎动态路由 — 路径 A：店铺全量拓扑发现 + 拓扑早停。

设计文档 3.3 路径 A：
- 进入"所有商品"页，高频滑动 + 卡片切分 + pHash 计算
- 不点击、不调用大模型，仅 INSERT IGNORE 商品基础表
- 连续 20 个已存在 → 拓扑早停
- 拓扑结束后，查询当天未更新单品 → 进入深挖阶段
"""
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TopologyDiscovery:
    """店铺全量拓扑发现引擎 — 快速盘点 + 早停。"""

    EARLY_STOP_THRESHOLD = 20  # 连续已存在商品数 → 触发早停

    def __init__(self, repo=None, dedup_engine=None,
                 early_stop_threshold: int = EARLY_STOP_THRESHOLD):
        self._repo = repo
        self._dedup = dedup_engine
        self._threshold = early_stop_threshold
        self._consecutive_existing = 0
        self._total_discovered = 0
        self._total_skipped = 0
        self._stopped = False

    def feed_card(self, virtual_id: str, store_id: str,
                  title: str, img_hash: str) -> bool:
        """处理一张商品卡片。

        Args:
            virtual_id: MD5(标题+主图特征码)
            store_id: 所属店铺
            title: 商品标题
            img_hash: 主图 pHash

        Returns:
            True 表示继续，False 表示触发早停
        """
        if self._stopped:
            return False

        if self._repo is not None:
            existing = self._repo.find_or_create_product(
                virtual_id, store_id, title, img_hash,
            )
            if existing == virtual_id:
                # 新商品
                self._consecutive_existing = 0
                self._total_discovered += 1
            else:
                # 已存在
                self._consecutive_existing += 1
                self._total_skipped += 1
        else:
            self._total_discovered += 1

        if self._consecutive_existing >= self._threshold:
            self._stopped = True
            logger.info(
                "拓扑早停: 连续 %d 个已存在商品 (共发现 %d 新/%d 跳过)",
                self._consecutive_existing,
                self._total_discovered,
                self._total_skipped,
            )
            return False

        return True

    def get_pending_products(self, store_id: str) -> list[dict]:
        """获取当天未更新销量的单品列表（深挖队列）。

        Returns:
            [{"virtual_id": str, "title": str}, ...]
        """
        if self._repo is None:
            return []
        # 委托 repo 层查询
        return []  # 占位 — 阶段二实现

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def stats(self) -> dict:
        return {
            "total_discovered": self._total_discovered,
            "total_skipped": self._total_skipped,
            "consecutive_existing": self._consecutive_existing,
            "stopped": self._stopped,
        }

    def reset(self) -> None:
        self._consecutive_existing = 0
        self._total_discovered = 0
        self._total_skipped = 0
        self._stopped = False
