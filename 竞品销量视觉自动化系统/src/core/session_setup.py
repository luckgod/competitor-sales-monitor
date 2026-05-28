"""店铺会话初始化 — 采集前绑定 store_id。

流程（来自设计文档 3.1）：
1. 控制自动化手势进入目标店铺首页
2. 截取屏幕顶部区域（店铺 LOGO / 名称）
3. 本地 OCR 识别完整店铺名称
4. 后端根据名称检索或创建 store_id
5. 标记会话状态，切换至"所有商品"列表页
"""
import logging
from typing import Optional

from .adb_controller import ADBController
from .session import SessionManager
from ..db.repository import CompetitorRepository

logger = logging.getLogger(__name__)


class StoreSessionInitializer:
    """店铺会话初始化器 — 截取店铺名 → OCR → 绑定 store_id。"""

    def __init__(self, adb: ADBController, repo: CompetitorRepository,
                 session_mgr: SessionManager,
                 ocr_func: Optional[callable] = None):
        self._adb = adb
        self._repo = repo
        self._session_mgr = session_mgr
        self._ocr_func = ocr_func or self._default_ocr

    def initialize(self, store_url_hint: str = "") -> str:
        """执行店铺会话初始化，返回绑定的 store_id。

        Args:
            store_url_hint: 目标店铺的关键词或 URL 片段

        Returns:
            绑定的 store_id
        """
        # Step 1: 截取屏幕顶部
        logger.info("截取店铺首页顶部区域...")
        screenshot_path = "state/temp_store_header.png"
        self._adb.capture_screenshot(screenshot_path)

        # Step 2: OCR 识别店铺名称
        store_name = self._recognize_store_name(screenshot_path)
        if not store_name:
            store_name = f"unknown_store_{store_url_hint}"

        # Step 3: 查找或创建 store_id
        store_id = self._repo.find_or_create_store(store_name)
        logger.info("店铺绑定: %s → %s", store_name, store_id)

        # Step 4: 更新会话状态
        self._session_mgr.update_progress(store_id=store_id)

        # Step 5: 切换至全部商品列表（具体操作依赖目标 App）
        self._navigate_to_product_list()

        return store_id

    def _recognize_store_name(self, screenshot_path: str) -> str:
        """从截图中 OCR 识别店铺名称。"""
        try:
            import cv2
            import numpy as np
            from paddleocr import PaddleOCR

            img = cv2.imread(screenshot_path)
            if img is None:
                return ""

            # 取屏幕顶部 15% 区域（店铺名称通常在顶部）
            h, w = img.shape[:2]
            header = img[0: int(h * 0.15), :]

            ocr = PaddleOCR(lang="ch", show_log=False)
            results = ocr.ocr(header)
            if results and results[0]:
                texts = [line[1][0] for line in results[0]]
                return " ".join(texts)
            return ""
        except ImportError:
            return self._default_ocr(screenshot_path)
        except Exception:
            logger.exception("OCR 识别店铺名称失败")
            return ""

    @staticmethod
    def _default_ocr(screenshot_path: str) -> str:
        """无 PaddleOCR 时的占位。"""
        return ""

    def _navigate_to_product_list(self) -> None:
        """导航至全部商品列表页。当前为占位实现。"""
        logger.debug("导航至全部商品列表（占位）")
