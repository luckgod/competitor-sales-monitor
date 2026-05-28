"""动态入口寻址模块 — PaddleOCR 扫描流量探针标签，定位"实时销量"弹窗入口。

设计文档 4.2：
- 在商品详情页屏幕中下部 OCR 检索特征词
- 命中后提取文本矩形中心坐标，叠加高斯偏移点击
- 未命中则降级为月销文本抓取
"""
import logging
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# 特征匹配词库（正则）
ENTRY_FEATURES = [
    r'超\d+人加购',
    r'\d+人感兴趣',
    r'近\d+天\d+\+人逛过',
    r'近期销量飙升',
]


@dataclass
class EntryPoint:
    """入口定位结果。"""
    x: int
    y: int
    matched_feature: str
    confidence: float = 1.0


class EntryLocator:
    """动态入口寻址器。

    使用 PaddleOCR 在屏幕中下部扫描流量探针标签，
    命中后返回叠加高斯偏移的点击坐标。
    """

    def __init__(self, ocr_func=None, screen_width: int = 1080,
                 screen_height: int = 1920, search_region: str = "lower_half"):
        """
        Args:
            ocr_func: OCR 函数，签名为 (image) -> list[dict]，
                      每项含 text/bbox/confidence。
                      为 None 时使用内置 PaddleOCR（多进程隔离）。
            screen_width: 屏幕像素宽度
            screen_height: 屏幕像素高度
            search_region: 搜索区域 "lower_half" | "full"
        """
        self._ocr_func = ocr_func
        self._screen_w = screen_width
        self._screen_h = screen_height
        self._search_region = search_region
        self._hit_count = 0
        self._miss_count = 0

    def locate(self, frame) -> Optional[EntryPoint]:
        """在截图中定位实时销量入口。

        Args:
            frame: numpy 数组，全屏截图或详情页截图

        Returns:
            EntryPoint 或 None（未命中，需降级）
        """
        import re

        # 截取搜索区域
        roi = self._crop_search_region(frame)

        # OCR 提取文本
        ocr_results = self._run_ocr(roi)
        if not ocr_results:
            self._miss_count += 1
            return None

        # 扫描特征词
        for item in ocr_results:
            text = item.get("text", "")
            bbox = item.get("bbox")

            for pattern in ENTRY_FEATURES:
                if re.search(pattern, text):
                    # 提取文本中心坐标
                    cx, cy = self._bbox_center(bbox)

                    # 叠加高斯偏移（模拟右手大拇指点击习惯）
                    cx += int(random.gauss(0, 8) - 5)  # 偏左
                    cy += int(random.gauss(0, 5))

                    # 转换回全屏坐标
                    if self._search_region == "lower_half":
                        cy += self._screen_h // 2

                    self._hit_count += 1
                    logger.info("入口命中: '%s' → (%d, %d)", text, cx, cy)
                    return EntryPoint(x=cx, y=cy, matched_feature=text)

        self._miss_count += 1
        return None

    # ── 内部 ──────────────────────────────────────────────────

    def _crop_search_region(self, frame):
        """截取搜索 ROI。"""
        h, w = frame.shape[:2]
        if self._search_region == "lower_half":
            return frame[h // 2:, :]
        return frame

    def _run_ocr(self, roi) -> list[dict]:
        """执行 OCR。"""
        if self._ocr_func is not None:
            return self._ocr_func(roi)

        # 降级：使用内置 PaddleOCR
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang="ch", show_log=False)
            results = ocr.ocr(roi)
            if not results or not results[0]:
                return []
            return [
                {"text": line[1][0], "bbox": line[0], "confidence": line[1][1]}
                for line in results[0]
            ]
        except ImportError:
            return []

    @staticmethod
    def _bbox_center(bbox) -> tuple[int, int]:
        """计算边界框中心点。"""
        if not bbox or len(bbox) < 4:
            return 0, 0
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    @property
    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / max(total, 1),
        }
