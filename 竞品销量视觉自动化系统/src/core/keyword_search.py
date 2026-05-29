"""双引擎动态路由 — 路径 B：首页关键词精准直搜。

设计文档 3.3 路径 B：
- 首页搜索框仿生键入"核心词+类目过滤"
- 搜索结果页用 OpenCV 匹配主图 pHash 找到目标竞品
- 进入详情页 → 动态入口寻址 → VLM 深挖
- 流量来源伪装为"大盘自然搜索引流"
"""
import logging
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)


class KeywordSearchEngine:
    """首页关键词精准直搜引擎。

    通过仿生键入关键词，在搜索结果页匹配目标商品主图，
    将进店流量伪装为自然搜索推荐。
    """

    def __init__(self, adb_pipeline=None, adb_controller=None,
                 dpi_mapper=None, dedup_engine=None):
        self._adb = adb_pipeline
        self._adb_ctrl = adb_controller
        self._dpi = dpi_mapper
        self._dedup = dedup_engine
        self._search_count = 0
        self._match_count = 0

    def search_and_locate(self, keywords: list[str],
                           target_phash: str | None = None,
                           target_title: str = "") -> Optional[tuple[int, int]]:
        """执行仿生搜索并定位目标商品。

        Args:
            keywords: 搜索关键词列表（优先用第一个 + 后两个做类目过滤）
            target_phash: 目标商品主图 pHash（用于结果页匹配）
            target_title: 目标商品标题

        Returns:
            (x, y) 目标商品在屏幕上的点击坐标，None 表示未找到
        """
        if not keywords:
            return None

        # Step 1: 首页搜索框仿生键入
        primary_kw = keywords[0]
        self._type_keyword(primary_kw)

        # Step 2: 随机挑选一个次要关键词做类目过滤
        if len(keywords) > 1:
            time.sleep(random.uniform(1.0, 2.0))
            secondary = random.choice(keywords[1:])
            self._type_keyword(secondary, append=True)

        # Step 3: 点击搜索按钮
        self._tap_search_button()

        # Step 4: 等待结果页加载
        time.sleep(random.uniform(2.0, 4.0))

        # Step 5: 在结果页匹配目标商品（pHash 或标题）
        coords = self._match_in_results(target_phash, target_title)
        if coords:
            self._match_count += 1
            logger.info("搜索结果匹配: (%d, %d)", coords[0], coords[1])

        self._search_count += 1
        return coords

    def _type_keyword(self, text: str, append: bool = False) -> None:
        """仿生键入关键词（带随机打字速度）。"""
        if self._adb is not None:
            if not append:
                # 清除搜索框焦点区域
                pass
            for ch in text:
                self._adb.text(ch)
                time.sleep(random.uniform(0.05, 0.15))  # 仿生打字间隔
        elif self._adb_ctrl is not None:
            escaped = text.replace(" ", "%s")
            self._adb_ctrl._run_adb(
                ["shell", "input", "text", escaped], timeout=5,
            )

    def _tap_search_button(self) -> None:
        """点击搜索按钮。"""
        # 搜索按钮通常位于键盘右下角，发送 ENTER 键
        if self._adb is not None:
            self._adb.keyevent("KEYCODE_ENTER")
        elif self._adb_ctrl is not None:
            self._adb_ctrl._run_adb(
                ["shell", "input", "keyevent", "KEYCODE_ENTER"], timeout=5,
            )

    def _match_in_results(self, target_phash: str | None,
                           target_title: str) -> Optional[tuple[int, int]]:
        """在搜索结果页中匹配目标商品。

        优先使用 pHash 匹配，降级为标题 OCR 匹配。
        """
        if target_phash and self._dedup is not None:
            # pHash 匹配路径（精确）
            # 截取搜索结果页 → 卡片切分 → 逐个比对 pHash
            pass
        # 降级：标题 OCR 匹配
        return None

    @property
    def stats(self) -> dict:
        return {
            "search_count": self._search_count,
            "match_count": self._match_count,
            "match_rate": self._match_count / max(self._search_count, 1),
        }
