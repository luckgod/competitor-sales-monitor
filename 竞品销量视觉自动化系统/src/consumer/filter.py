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

    # ── V5.0 骨架屏升级防御 ──────────────────────────────────

    def is_skeleton_screen(self, card: dict,
                           templates: list | None = None,
                           ssim_threshold: float = 0.85,
                           sobel_threshold: float = 50.0) -> bool:
        """SSIM 模板匹配 + Sobel 边缘梯度 双重拦截骨架屏。

        Args:
            card: 商品卡片 dict，含 image/sales 子图
            templates: 骨架屏模板图像列表
            ssim_threshold: SSIM 高于此值判定无效
            sobel_threshold: Sobel 梯度方差低于此值判定无效

        Returns:
            True 表示命中骨架屏特征，应丢弃
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return False

        sales_roi = card.get("sales")
        image_roi = card.get("image")

        # 防御 1：SSIM 模板匹配
        if templates and image_roi is not None:
            for template in templates:
                if self._calc_ssim(image_roi, template) > ssim_threshold:
                    logger.debug("SSIM 命中骨架屏模板")
                    return True

        # 防御 2：Sobel 边缘梯度检查
        if sales_roi is not None:
            grad_var = self._calc_sobel_variance(sales_roi)
            if grad_var < sobel_threshold:
                logger.debug("Sobel 梯度方差过低: %.1f < %.0f",
                             grad_var, sobel_threshold)
                return True

        return False

    @staticmethod
    def _calc_ssim(img1, img2) -> float:
        """计算两张图像的结构相似性（SSIM）。

        简化实现：缩放至相同大小后计算归一化互相关系数。
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return 0.0

        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # 统一尺寸
            h = min(gray1.shape[0], gray2.shape[0])
            w = min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (w, h))
            gray2 = cv2.resize(gray2, (w, h))

            # 归一化互相关系数
            mean1, mean2 = gray1.mean(), gray2.mean()
            std1, std2 = gray1.std(), gray2.std()

            if std1 < 1e-6 or std2 < 1e-6:
                return 1.0 if abs(mean1 - mean2) < 1e-6 else 0.0

            corr = np.mean((gray1 - mean1) * (gray2 - mean2)) / (std1 * std2)
            return float(max(0.0, min(1.0, corr)))
        except Exception:
            return 0.0

    @staticmethod
    def _calc_sobel_variance(img) -> float:
        """计算 Sobel 边缘梯度方差。

        渲染完毕的文字具有丰富的边缘纹理 → 高梯度方差。
        骨架屏渐变色块无清晰边缘 → 低梯度方差。
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return 1000.0

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_x_abs = cv2.convertScaleAbs(grad_x)
            return float(np.var(grad_x_abs))
        except Exception:
            return 0.0
