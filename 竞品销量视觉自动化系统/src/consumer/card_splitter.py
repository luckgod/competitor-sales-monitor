"""OpenCV 商品卡片网格化切分引擎。

输入：1080P 全屏截图（numpy 数组）
输出：独立商品卡片子图列表，每张卡片包含 (主图区, 标题区, 销量区) 三个 ROI。

支持两种模式：
1. 固定锚点比例模式：按已知 UI 布局的百分比坐标切分
2. 轮廓检测模式：自适应检测矩形卡片边界
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CardRegions:
    """单张商品卡片的三个子区域（归一化相对坐标，0.0~1.0）。"""
    # 全卡片区域
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    # 主图区 (区域A)
    image_x: float = 0.05
    image_y: float = 0.05
    image_w: float = 0.35
    image_h: float = 0.85
    # 标题区 (区域B)
    title_x: float = 0.42
    title_y: float = 0.05
    title_w: float = 0.53
    title_h: float = 0.50
    # 销量区 (区域C)
    sales_x: float = 0.42
    sales_y: float = 0.58
    sales_w: float = 0.53
    sales_h: float = 0.35


# 默认淘宝商品列表布局：每行 2 个卡片，从上到下排列
# 每个卡片约占 1/2 宽度，高度约 1/4 ~ 1/3 屏幕
DEFAULT_LAYOUT = CardRegions(
    x=0.0, y=0.0, w=0.5, h=0.3,
    image_x=0.03, image_y=0.05, image_w=0.42, image_h=0.70,
    title_x=0.48, title_y=0.05, title_w=0.49, title_h=0.42,
    sales_x=0.48, sales_y=0.52, sales_w=0.49, sales_h=0.40,
)


class CardSplitter:
    """商品卡片网格化切分器。

    对全屏截图执行轮廓检测或按固定锚点比例切分出独立商品卡片。
    """

    def __init__(self, layout: CardRegions | None = None,
                 mode: str = "contour",
                 screen_width: int = 1080, screen_height: int = 1920):
        self._layout = layout or DEFAULT_LAYOUT
        self._mode = mode
        self._screen_width = screen_width
        self._screen_height = screen_height

    def split(self, frame) -> list[dict]:
        """将全屏帧拆分为商品卡片列表。

        每张卡片返回 dict:
          {"card": 全卡片子图, "image": 主图区, "title": 标题区, "sales": 销量区,
           "x": int, "y": int, "w": int, "h": int}
        """
        try:
            if self._mode == "contour":
                cards = self._split_by_contour(frame)
            else:
                cards = self._split_by_grid(frame)
            return cards
        except Exception:
            logger.exception("卡片切分失败")
            return []

    def _split_by_contour(self, frame) -> list[dict]:
        """V5.1 语义轮廓切分 — Y轴安全区 + 形态学膨胀 + 触线丢弃。"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return self._split_by_grid(frame)

        sh, sw = self._screen_height, self._screen_width
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 形态学操作：膨胀连接卡片边缘断点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(morph, 30, 150)
        dilated = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Y轴安全区
        SAFE_ZONE_TOP = int(sh * 0.18)
        SAFE_ZONE_BOTTOM = int(sh * 0.90)
        min_area = (sw * sh) * 0.015

        cards = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # 面积过滤
            if area < min_area or area > (sw * sh * 0.85):
                continue
            # 形状过滤：太扁的横向条（导航栏/广告横幅）
            if w > h * 1.5 and h < sh * 0.10:
                continue
            if h < sh * 0.06:
                continue

            # 【核心修复】Y轴安全区触线判定：拒绝残缺的半截卡片
            if y < SAFE_ZONE_TOP or (y + h) > SAFE_ZONE_BOTTOM:
                continue

            cards.append({"x": x, "y": y, "w": w, "h": h, "area": area})

        # 按 y 坐标排序
        avg_h = sum(c["h"] for c in cards) / max(len(cards), 1)
        row_tolerance = max(int(avg_h / 3), 10)
        cards.sort(key=lambda c: (c["y"] // row_tolerance, c["x"]))

        return [self._extract_regions(frame, c) for c in cards]

    def _split_by_grid(self, frame) -> list[dict]:
        """按固定锚点比例网格切分。"""
        L = self._layout
        sw, sh = self._screen_width, self._screen_height

        cols = max(1, int(1.0 / L.w))
        rows = max(1, int(1.0 / L.h))

        cards = []
        for row in range(rows):
            for col in range(cols):
                x = int(col * L.w * sw)
                y = int(row * L.h * sh)
                w = int(L.w * sw)
                h = int(L.h * sh)
                cards.append(self._extract_regions(frame, {"x": x, "y": y, "w": w, "h": h}))
        return cards

    def _extract_regions(self, frame, bbox: dict) -> dict:
        """从一张卡片中提取三个子区域。"""
        try:
            import cv2
        except ImportError:
            return {"card": None, "image": None, "title": None, "sales": None, **bbox}

        L = self._layout
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        def _crop(rx: float, ry: float, rw: float, rh: float):
            rx = max(0, min(1, rx))
            ry = max(0, min(1, ry))
            rw = max(0.01, min(1, rw))
            rh = max(0.01, min(1, rh))
            return frame[
                y + int(ry * h): y + int((ry + rh) * h),
                x + int(rx * w): x + int((rx + rw) * w),
            ]

        return {
            "card": frame[y:y + h, x:x + w],
            "image": _crop(L.image_x, L.image_y, L.image_w, L.image_h),
            "title": _crop(L.title_x, L.title_y, L.title_w, L.title_h),
            "sales": _crop(L.sales_x, L.sales_y, L.sales_w, L.sales_h),
            "x": x, "y": y, "w": w, "h": h,
        }
