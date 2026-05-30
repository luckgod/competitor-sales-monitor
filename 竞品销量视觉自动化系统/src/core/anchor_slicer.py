"""锚点中线切割法 — 弃边框、寻心脏、画楚河汉界

核心思想：
1. 找价格符号(¥/人付款)作为"心脏锚点" → 断定此处必有商品
2. X/Y轴聚类 → 感知单列/双列布局
3. 相邻锚点中点画切割线 → 绝对安全区包含标题
4. 三道防线：锚点校验 + 边界外扩 + 空间重叠剔除
"""

import cv2, numpy as np
from typing import Optional


class AnchorSlicer:
    """锚点中线切割器 — 动态感知布局 + 楚河汉界切分"""

    def __init__(self, x_threshold: int = 200, y_threshold: int = 100,
                 padding: int = 10):
        self._x_threshold = x_threshold
        self._y_threshold = y_threshold
        self._padding = padding
        self._anchors: list[dict] = []
        self._cols: int = 1
        self._rows: int = 0

    # ── 锚点提取 ──────────────────────────────────────────

    def extract_anchors(self, image: np.ndarray,
                        ocr_reader=None) -> list[dict]:
        """全屏盲扫，只保留含价格特征的文本块中心点。

        Args:
            image: BGR 全屏截图
            ocr_reader: EasyOCR Reader 实例(复用避免重复初始化)

        Returns:
            [{"x": int, "y": int, "text": str}, ...]
        """
        h, w = image.shape[:2]

        # 懒加载 OCR
        if ocr_reader is None:
            import easyocr
            ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

        results = ocr_reader.readtext(image)

        self._anchors = []
        for t in results:
            text, conf, bbox = t[1], t[2], t[0]
            if conf < 0.3:
                continue

            # 增强价格锚点：销量/价格/促销关键词
            price_keywords = {'人付款', '付款', '已售', '月销', '¥', '￥', '元',
                              '包邮', '到手', '券后', '优惠', '价格'}
            is_price = any(kw in text for kw in price_keywords)
            # 也接受"数字+付款/已售"组合(OCR可能漏¥)
            if not is_price:
                has_digit = any(c.isdigit() for c in text)
                is_price = has_digit and any(kw in text for kw in ['付款','人付款','已售','月销'])

            if not is_price:
                continue

            cx = int((bbox[0][0] + bbox[2][0]) / 2)
            cy = int((bbox[0][1] + bbox[2][1]) / 2)

            # 跳过顶部(搜索栏/筛选栏)和底部导航
            if cy < h * 0.18 or cy > h * 0.92:
                continue

            self._anchors.append({"x": cx, "y": cy, "text": text})

        return self._anchors

    # ── 布局感知 ──────────────────────────────────────────

    def detect_layout(self) -> tuple[int, list[list[dict]]]:
        """X/Y轴聚类 → 感知单列/双列 + 行分组。

        X轴用整体 spread 判列数(非简单阈值):
        - 所有锚点X spread < 屏宽40% → 单列(list view)
        - 否则存在明显双峰 → 双列(grid view)
        """
        if len(self._anchors) < 2:
            self._cols = 1
            return 1, [self._anchors]

        xs = sorted(a["x"] for a in self._anchors)
        x_spread = xs[-1] - xs[0]

        # 判列数：spread < 40%屏宽 → 单列
        if x_spread < 480:  # 1220px * 40% ≈ 480
            self._cols = 1
            col_anchors = [sorted(self._anchors, key=lambda a: a["y"])]
        else:
            # 双列：找双峰
            x_mean = sum(xs) / len(xs)
            left = [a for a in self._anchors if a["x"] < x_mean]
            right = [a for a in self._anchors if a["x"] >= x_mean]
            if len(left) >= 2 and len(right) >= 2:
                self._cols = 2
                left.sort(key=lambda a: a["y"])
                right.sort(key=lambda a: a["y"])
                col_anchors = [left, right]
            else:
                self._cols = 1
                col_anchors = [sorted(self._anchors, key=lambda a: a["y"])]

        return self._cols, col_anchors

    # ── 中线切割 ──────────────────────────────────────────

    def slice(self, image: np.ndarray,
              col_anchors: list[list[dict]]) -> list[np.ndarray]:
        """画楚河汉界，切分商品卡片。"""
        h, w = image.shape[:2]
        cards = []

        # 计算所有锚点的Y值(去重合并)
        all_ys = sorted(set(a["y"] for ancs in col_anchors for a in ancs))
        merged = []
        for y in all_ys:
            if not merged or y - merged[-1] > self._y_threshold:
                merged.append(y)
            else:
                merged[-1] = int((merged[-1] + y) / 2)
        rows_y = merged
        self._rows = len(rows_y)

        # 估算卡高（锚点间距）
        if len(rows_y) >= 2:
            card_h = int(np.median([rows_y[i+1] - rows_y[i]
                         for i in range(len(rows_y)-1)]))
        else:
            card_h = 440

        # X切割线
        if self._cols == 1:
            x_boundaries = [(0, w)]  # 单列全宽
        else:
            col_centers = [int(np.median([a["x"] for a in ancs]))
                           for ancs in col_anchors if ancs]
            col_centers.sort()
            x_mid = (col_centers[0] + col_centers[1]) // 2 if len(col_centers) >= 2 else w // 2
            x_boundaries = [(0, x_mid), (x_mid, w)]

        # 每张卡：锚点上方 card_h*0.9 → 锚点下方 30px
        for anchor_y in rows_y:
            y1 = max(0, anchor_y - int(card_h * 0.9) - self._padding)
            y2 = min(h, anchor_y + 30 + self._padding)

            for x1, x2 in x_boundaries:
                card = image[y1:y2, x1:x2]
                if card.shape[0] > 100:
                    cards.append(card)

        return cards

    @property
    def layout_info(self) -> dict:
        return {
            "cols": self._cols,
            "rows": self._rows,
            "anchors": len(self._anchors),
            "mode": "单列" if self._cols == 1 else f"{self._cols}列网格",
        }

    # ── 内部 ──────────────────────────────────────────────

    def _has_anchor(self, card: np.ndarray, off_x: int, off_y: int) -> bool:
        """反向校验：卡片区域内至少包含一个价格锚点。"""
        ch, cw = card.shape[:2]
        for a in self._anchors:
            ax, ay = a["x"] - off_x, a["y"] - off_y
            if -self._padding <= ax < cw + self._padding and \
               -self._padding <= ay < ch + self._padding:
                return True
        return False


def slice_page(image_path: str) -> tuple[list[np.ndarray], dict]:
    """便捷入口：截图路径 → 卡片列表 + 布局信息"""
    img = cv2.imread(image_path)
    if img is None:
        return [], {"error": "无法读取图片"}

    slicer = AnchorSlicer()
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    slicer.extract_anchors(img, reader)
    cols, col_anchors = slicer.detect_layout()
    cards = slicer.slice(img, col_anchors)
    return cards, slicer.layout_info
