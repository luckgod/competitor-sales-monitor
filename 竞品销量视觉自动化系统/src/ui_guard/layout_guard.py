"""界面布局改版防护网 — UI Shift Kill-Switch。

设计规范（来自设计文档 12.x）：
- 每次采集开始时，对前 5 个卡片进行基准校验，建立 UI 模板基线
- 运行中检测：连续 20 个商品偏离度 > 15% → 触发数据熔断锁
- 连续 20 次 {"sales": null} → 触发熔断
- 触发后：终止滑动 + 锁死数据库写入 + 发送紧急改版告警
"""
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class LayoutGuard:
    """UI 改版数据熔断锁 — 检测目标 App 布局变动并截断脏数据写入。"""

    # 阈值
    DEVIATION_THRESHOLD = 0.15      # 宽高比偏离度 > 15%
    CONSECUTIVE_ABNORMAL = 20       # 连续异常次数
    BASELINE_SAMPLES = 5            # 基准线采样数

    def __init__(self):
        # 基准线：(avg_aspect_ratio, avg_area_ratio)
        self._baseline: Optional[tuple[float, float]] = None
        self._sample_buffer: list[tuple[float, float]] = []

        self._abnormal_count = 0
        self._null_sales_count = 0
        self._killed = False  # 熔断锁状态

        self._lock = threading.Lock()
        self._on_killswitch: list[Callable[[str], None]] = []

    # ── 回调 ──────────────────────────────────────────────────

    def on_killswitch(self, callback: Callable[[str], None]) -> None:
        self._on_killswitch.append(callback)

    # ── 基准线建立 ────────────────────────────────────────────

    def calibrate(self, card_bboxes: list[dict]) -> None:
        """建立当前 UI 版本的卡片布局基准线。

        Args:
            card_bboxes: 前 BASELINE_SAMPLES 个卡片的边界框
                         [{"w": int, "h": int}, ...]
        """
        with self._lock:
            self._sample_buffer = []
            self._baseline = None
            for bbox in card_bboxes[:self.BASELINE_SAMPLES]:
                self._sample(bbox)
            self._finalize_baseline()

    def _sample(self, bbox: dict) -> None:
        w, h = bbox.get("w", 0), bbox.get("h", 0)
        if w <= 0 or h <= 0:
            return
        aspect = w / h
        self._sample_buffer.append((aspect, w * h))

    def _finalize_baseline(self) -> None:
        if len(self._sample_buffer) < 2:
            return
        aspects = [s[0] for s in self._sample_buffer]
        areas = [s[1] for s in self._sample_buffer]
        avg_aspect = sum(aspects) / len(aspects)
        avg_area = sum(areas) / len(areas)
        self._baseline = (avg_aspect, avg_area)
        logger.info("UI 基线已建立: aspect=%.3f, area=%.0f", avg_aspect, avg_area)

    # ── 运行时检测 ────────────────────────────────────────────

    def check(self, card_bbox: dict, sales_value: Optional[int]) -> bool:
        """检查当前卡片是否符合基准布局。

        Args:
            card_bbox: 当前卡片边界框 {"w": int, "h": int}
            sales_value: 销量清洗结果（None 表示 LLM 输出 null）

        Returns:
            True 表示正常，False 表示触发熔断。
        """
        with self._lock:
            if self._killed:
                return False

            if self._baseline is None:
                return True  # 尚未建立基线，容错放行

            w, h = card_bbox.get("w", 0), card_bbox.get("h", 0)
            if w <= 0 or h <= 0:
                return True

            current_aspect = w / h
            baseline_aspect, _ = self._baseline

            deviation = abs(current_aspect - baseline_aspect) / baseline_aspect

            if deviation > self.DEVIATION_THRESHOLD:
                self._abnormal_count += 1
            else:
                self._abnormal_count = max(0, self._abnormal_count - 1)

            if sales_value is None or sales_value == -1:
                self._null_sales_count += 1
            else:
                self._null_sales_count = max(0, self._null_sales_count - 1)

            # 两个触发条件满足其一即熔断
            if self._abnormal_count >= self.CONSECUTIVE_ABNORMAL:
                self._trigger(f"连续 {self._abnormal_count} 个商品偏离度 > {self.DEVIATION_THRESHOLD*100:.0f}%")
                return False

            if self._null_sales_count >= self.CONSECUTIVE_ABNORMAL:
                self._trigger(f"连续 {self._null_sales_count} 次 sales=null")
                return False

            return True

    def _trigger(self, reason: str) -> None:
        self._killed = True
        logger.critical("UI 改版熔断锁触发: %s", reason)
        for cb in self._on_killswitch:
            try:
                cb(f"UI 布局可能改版: {reason}")
            except Exception:
                logger.exception("熔断告警回调异常")

    # ── 状态 ──────────────────────────────────────────────────

    @property
    def is_killed(self) -> bool:
        return self._killed

    @property
    def baseline_established(self) -> bool:
        return self._baseline is not None

    def reset(self) -> None:
        """手动重置熔断锁。"""
        with self._lock:
            self._killed = False
            self._abnormal_count = 0
            self._null_sales_count = 0
            self._baseline = None
            self._sample_buffer = []
