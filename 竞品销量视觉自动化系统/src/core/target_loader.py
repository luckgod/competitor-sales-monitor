"""靶点池加载器 — 读取 targets.yaml 并随机打散执行顺序。

设计文档 8：
- 禁止在 Python 核心代码中硬编码店铺信息
- 系统启动后 random.shuffle 打散顺序
- Tier 1 优先，Tier 2 间歇穿插
"""
import logging
import random
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class StoreTarget:
    """单个店铺靶点。"""
    def __init__(self, store_name: str, keywords: list[str] = None,
                 tier: int = 1):
        self.store_name = store_name
        self.keywords = keywords or []
        self.tier = tier


class TargetLoader:
    """靶点池加载与调度。

    用法:
        loader = TargetLoader("config/targets.yaml")
        targets = loader.load()
        for target in loader.shuffled():
            ...
    """

    def __init__(self, config_path: str = "config/targets.yaml"):
        self._path = Path(config_path)
        self._targets: list[StoreTarget] = []

    def load(self) -> list[StoreTarget]:
        """从 YAML 文件加载靶点池。"""
        if not self._path.exists():
            logger.warning("靶点池配置不存在: %s", self._path)
            return []

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            logger.exception("靶点池 YAML 解析失败")
            return []

        self._targets = []

        store_pool = data.get("store_pool", {})
        for tier_name, tier_key in [("tier_1", 1), ("tier_2", 2)]:
            tier_list = store_pool.get(tier_name, [])
            if not tier_list:
                continue
            for item in tier_list:
                self._targets.append(StoreTarget(
                    store_name=item.get("store_name", ""),
                    keywords=item.get("keywords", []),
                    tier=tier_key,
                ))

        logger.info("靶点池加载: Tier1=%d, Tier2=%d",
                     self.tier1_count, self.tier2_count)
        return self._targets

    def shuffled(self, tier1_first: bool = True) -> list[StoreTarget]:
        """返回随机打散的靶点列表。

        Args:
            tier1_first: True 时 Tier 1 排在 Tier 2 前面（但各自内部随机打散）

        Returns:
            打散后的靶点列表
        """
        if not self._targets:
            return []

        tier1 = [t for t in self._targets if t.tier == 1]
        tier2 = [t for t in self._targets if t.tier == 2]

        random.shuffle(tier1)
        random.shuffle(tier2)

        if tier1_first:
            return tier1 + tier2

        # 交叉穿插：Tier 1 和 Tier 2 交替
        result = []
        max_len = max(len(tier1), len(tier2))
        for i in range(max_len):
            if i < len(tier1):
                result.append(tier1[i])
            if i < len(tier2):
                result.append(tier2[i])
        return result

    def get_tier1_keywords(self, store_name: str) -> list[str]:
        """获取指定 Tier 1 店铺的搜索关键词。"""
        for t in self._targets:
            if t.store_name == store_name and t.tier == 1:
                return t.keywords
        return []

    @property
    def tier1_count(self) -> int:
        return sum(1 for t in self._targets if t.tier == 1)

    @property
    def tier2_count(self) -> int:
        return sum(1 for t in self._targets if t.tier == 2)

    @property
    def total_count(self) -> int:
        return len(self._targets)

    @property
    def targets(self) -> list[StoreTarget]:
        return self._targets
