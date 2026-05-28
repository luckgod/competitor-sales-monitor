"""懒加载无效帧过滤器 — 在 pHash 计算之前拦截未渲染的白块/占位卡片。

判定指标（来自设计文档 3.3.1）：
- 图像信息熵（Entropy）< 1.0 → 无效（正常 > 4.0）
- 灰度方差（Variance）< 10 → 无效（正常 > 500）
- OCR 字符长度 < 5 → 无效

命中任意一条即判定为无效帧，直接销毁，不进入去重队列和 LLM 调用。
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LazyLoadFilter:
    """无效帧过滤器 — 拦截白块/未加载占位卡片。"""

    # 判定阈值
    ENTROPY_THRESHOLD = 1.0      # 信息熵低于此值判定无效
    VARIANCE_THRESHOLD = 10.0    # 灰度方差低于此值判定无效
    OCR_MIN_LENGTH = 5           # OCR 文本最短长度

    def is_invalid(self, card: dict) -> bool:
        """判断商品卡片是否为未加载完成的空白帧。

        检查三个指标，任意一个命中即判定无效。

        Args:
            card: CardSplitter 输出的卡片 dict，含 image/sales 子图

        Returns:
            True 表示无效帧（应丢弃），False 表示正常帧（可继续处理）
        """
        image_roi = card.get("image")
        sales_roi = card.get("sales")

        # 指标 1：图像信息熵
        if image_roi is not None:
            entropy = self._calc_entropy(image_roi)
            if entropy < self.ENTROPY_THRESHOLD:
                logger.debug("无效帧: 主图熵 %.2f < %.1f", entropy, self.ENTROPY_THRESHOLD)
                return True

        # 指标 2：灰度方差
        if sales_roi is not None:
            variance = self._calc_variance(sales_roi)
            if variance < self.VARIANCE_THRESHOLD:
                logger.debug("无效帧: 销量区方差 %.1f < %.0f", variance, self.VARIANCE_THRESHOLD)
                return True

        return False

    def is_invalid_by_text(self, ocr_text: str) -> bool:
        """检查 OCR 文本是否为无效占位。

        Args:
            ocr_text: OCR 提取到的文本

        Returns:
            True 表示文本不足以判定为有效商品
        """
        if ocr_text is None or len(ocr_text.strip()) < self.OCR_MIN_LENGTH:
            logger.debug("无效帧: OCR 文本过短 '%s'", ocr_text)
            return True
        return False

    # ── 内部算法 ──────────────────────────────────────────────

    @staticmethod
    def _calc_entropy(img) -> float:
        """计算灰度图像的信息熵。"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return 5.0  # 无 cv2 时默认判定正常

        if img is None or img.size == 0:
            return 0.0
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / (hist.sum() + 1e-10)
            entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
            return entropy
        except Exception:
            return 0.0

    @staticmethod
    def _calc_variance(img) -> float:
        """计算灰度图像的方差。"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return 1000.0  # 无 cv2 时默认判定正常

        if img is None or img.size == 0:
            return 0.0
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return float(np.var(gray))
        except Exception:
            return 0.0
