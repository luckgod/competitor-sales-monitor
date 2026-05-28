"""VLM 视觉状态机 — 增量差分裁剪 + KV-Cache slot 管理。

设计文档 4.4.1：
- 弹窗滑动时 30%~50% 与上一帧重叠，全图送 VLM 造成算力浪费
- 计算 Scrollable ROI 的像素位移 Δy，仅将底部新增增量切片送 VLM
- System Prompt KV-Cache 保持复用，单次推理 400ms → 80ms
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VisualStateMachine:
    """VLM 视觉状态机 — 增量推理管理。

    维护弹窗滚动状态，对连续帧执行差分检测。
    仅将新涌入区域送入 VLM 推理，减少 Token 消耗。
    """

    def __init__(self, scrollable_roi: Tuple[int, int, int, int] | None = None,
                 diff_threshold: float = 0.05):
        """
        Args:
            scrollable_roi: 可滚动区域 (x, y, w, h)，相对屏幕坐标
            diff_threshold: 像素变动率阈值，低于此值判定为静止
        """
        self._roi = scrollable_roi
        self._diff_threshold = diff_threshold
        self._prev_frame = None
        self._delta_y = 0  # 累计滚动位移
        self._slot_id = -1
        self._frame_count = 0
        self._diff_count = 0

    def feed(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """输入一帧弹窗截图，返回增量裁剪区域或 None（无需推理）。

        Args:
            frame: 当前弹窗帧（numpy 数组）

        Returns:
            (x, y, w, h) 增量区域裁剪坐标，None 表示画面无变化或首帧
        """
        self._frame_count += 1

        if self._prev_frame is None:
            # 首帧：全图送 VLM
            self._prev_frame = frame.copy()
            h, w = frame.shape[:2]
            return (0, 0, w, h)

        # 计算像素位移 Δy
        dy = self._estimate_vertical_shift(self._prev_frame, frame)
        if dy is None or dy <= 0:
            self._prev_frame = frame.copy()
            return None  # 无变化或反向滚动，跳过

        self._delta_y += dy
        self._prev_frame = frame.copy()
        self._diff_count += 1

        h, w = frame.shape[:2]
        if dy >= h:
            # 整屏替换，送全图
            return (0, 0, w, h)

        # 仅返回底部新涌入的增量区域
        return (0, h - dy, w, dy)

    @property
    def should_use_cache(self) -> bool:
        """当前帧是否应启用 KV-Cache 复用。"""
        return self._frame_count > 1

    @property
    def slot_id(self) -> int:
        return self._slot_id

    @slot_id.setter
    def slot_id(self, value: int) -> None:
        self._slot_id = value

    @property
    def stats(self) -> dict:
        return {
            "frame_count": self._frame_count,
            "diff_count": self._diff_count,
            "delta_y_total": self._delta_y,
            "cache_enabled": self.should_use_cache,
        }

    def reset(self) -> None:
        """重置状态机（弹窗关闭/切换商品时调用）。"""
        self._prev_frame = None
        self._delta_y = 0
        self._frame_count = 0
        self._diff_count = 0

    # ── 内部 ──────────────────────────────────────────────────

    def _estimate_vertical_shift(self, prev, curr) -> Optional[int]:
        """估算两帧之间的垂直像素位移。

        使用光流法或模板匹配近似。
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None

        try:
            gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # 取上半部分做模板匹配（弹窗标题区通常不变）
            h = gray_prev.shape[0]
            template_h = min(h // 3, 200)
            template = gray_prev[:template_h, :]
            search = gray_curr[: min(template_h + 100, h), :]

            if template.shape[0] > search.shape[0]:
                return None

            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val < 0.7:
                return None  # 匹配度太低，可能是整屏切换

            return max_loc[1]  # Δy = 模板在新图中的 y 偏移

        except Exception:
            return None
