"""pHash 感知哈希模糊去重引擎。

算法流程（来自设计文档 3.3）：
1. 内存维护双端滑动窗口，容量 30
2. 提取区域 A（主图）→ 灰度化 → 缩放 → pHash 64 位特征码
3. 新卡片进入时：
   a. 比对标题文本相似度，> 90% 则进一步检查
   b. 计算 pHash 汉明距离，< 5 判定为重复 → 丢弃
   c. 否则为新商品，压入滑动窗口
"""
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class DedupEngine:
    """pHash 模糊去重引擎 — 滑动窗口 + 汉明距离判定。"""

    # 阈值
    DEFAULT_WINDOW_SIZE = 30
    HAMMING_THRESHOLD = 5        # 汉明距离低于此值判定重复
    TEXT_SIMILARITY_THRESHOLD = 0.90  # 标题相似度阈值

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self._window_size = window_size
        # 滑动窗口：存储 (title_hash, phash_hex) 元组
        self._window: deque[tuple[int, str]] = deque(maxlen=window_size)
        self._hit_count = 0
        self._miss_count = 0

    def is_duplicate(self, title: str, image_roi) -> bool:
        """判断当前商品卡片是否为重复项。

        Args:
            title: OCR 提取的商品标题
            image_roi: 主图区域子图（numpy 数组或 None）

        Returns:
            True 表示重复（应丢弃），False 表示新商品
        """
        if image_roi is None:
            return self._check_by_title_only(title)

        phash = self._compute_phash(image_roi)
        if phash is None:
            return self._check_by_title_only(title)

        title_hash = hash(title)

        for cached_title_hash, cached_phash in self._window:
            # 先比对标题相似度（快速路径）
            title_sim = self._title_similarity_fast(title_hash, cached_title_hash)
            if title_sim < self.TEXT_SIMILARITY_THRESHOLD:
                continue

            # 标题高度相似，再计算汉明距离
            distance = self._hamming_distance(phash, cached_phash)
            if distance < self.HAMMING_THRESHOLD:
                self._hit_count += 1
                logger.debug("去重命中: 汉明距离=%d", distance)
                return True

        # 新商品，压入窗口
        self._window.append((title_hash, phash))
        self._miss_count += 1
        return False

    def _check_by_title_only(self, title: str) -> bool:
        """仅通过标题哈希快速判定（无主图时）。"""
        title_hash = hash(title)
        for cached_title_hash, _ in self._window:
            if cached_title_hash == title_hash:
                self._hit_count += 1
                return True
        self._window.append((title_hash, ""))
        self._miss_count += 1
        return False

    # ── pHash 计算 ────────────────────────────────────────────

    @staticmethod
    def _compute_phash(img) -> Optional[str]:
        """计算图像的 64 位感知哈希值。"""
        try:
            from PIL import Image
            import imagehash
            if isinstance(img, Image.Image):
                pil_img = img
            else:
                import numpy as np
                pil_img = Image.fromarray(img)
            return str(imagehash.phash(pil_img, hash_size=8))
        except ImportError:
            # 降级：使用简单的像素均值哈希
            return DedupEngine._simple_phash(img)
        except Exception:
            logger.warning("pHash 计算失败", exc_info=True)
            return None

    @staticmethod
    def _simple_phash(img) -> Optional[str]:
        """简易均值哈希（无需 imagehash 库时的降级方案）。"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (8, 8))
            avg = resized.mean()
            bits = (resized > avg).flatten()
            hash_val = 0
            for bit in bits:
                hash_val = (hash_val << 1) | int(bit)
            return format(hash_val, '016x')
        except Exception:
            return None

    # ── 汉明距离 ──────────────────────────────────────────────

    @staticmethod
    def _hamming_distance(hash1: str, hash2: str) -> int:
        """计算两个十六进制 pHash 的汉明距离。"""
        if not hash1 or not hash2:
            return 999
        try:
            n1 = int(hash1, 16)
            n2 = int(hash2, 16)
            return (n1 ^ n2).bit_count()
        except (ValueError, AttributeError):
            # Python < 3.8 fallback
            n = int(hash1, 16) ^ int(hash2, 16)
            return bin(n).count("1")

    @staticmethod
    def _title_similarity_fast(h1: int, h2: int) -> float:
        """标题哈希快速相似度（基于哈希值的近似比较）。

        完整的文本相似度应由 difflib.SequenceMatcher 计算，
        此处为性能优化，仅比较哈希是否相等。
        实际使用中配合 _check_by_title_only 降级路径。
        """
        return 1.0 if h1 == h2 else 0.0

    # ── 指标 ──────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "window_size": len(self._window),
            "max_window_size": self._window_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "dedup_rate": (
                self._hit_count / max(self._hit_count + self._miss_count, 1)
            ),
        }
