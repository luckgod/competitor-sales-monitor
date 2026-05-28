"""非预期弹窗遮挡检测器 — 泛化检测系统弹窗/验证码/推送。

设计规范（来自设计文档 9.2）：
- 连续 N 帧检测到遮挡 → 挂起系统 + 丢弃队列 + 发送告警
- 每 60 秒重检屏幕，恢复后自动续跑
- 30 分钟无人处理 → 硬重置（kill App + 重启 + 重新会话初始化）
- 连续 3 次硬重置失败 → 彻底停机
"""
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class OverlayDetector:
    """非预期弹窗遮挡检测器 — 基于关键词 + 图像异常判定。"""

    # 系统弹窗关键词（中英文）
    SYSTEM_KEYWORDS = [
        "确定", "取消", "电池", "低电量", "充电", "更新", "升级",
        "安装", "允许", "拒绝", "同意", "设置", "通知", "提醒",
        "OK", "Cancel", "Update", "Install", "Allow", "Deny",
        "验证码", "滑块", "拼图", "请点击", "滑动验证",
        "重新登录", "登录失效", "账号异常",
    ]

    ABNORMAL_THRESHOLD = 3          # 连续异常帧数 → 挂起
    RECOVERY_CHECK_INTERVAL = 60    # 恢复检查间隔 (秒)
    HARD_RESET_TIMEOUT = 1800       # 30 分钟无人处理 → 硬重置
    MAX_HARD_RESETS = 3             # 最多硬重置次数

    def __init__(self):
        self._abnormal_count = 0
        self._is_blocked = False
        self._hard_reset_count = 0
        self._blocked_since: float | None = None
        self._lock = threading.Lock()

        self._on_blocked_callbacks: list[Callable[[str], None]] = []
        self._on_recovered_callbacks: list[Callable[[], None]] = []
        self._on_hard_reset_callbacks: list[Callable[[], None]] = []

    # ── 回调 ──────────────────────────────────────────────────

    def on_blocked(self, callback: Callable[[str], None]) -> None:
        self._on_blocked_callbacks.append(callback)

    def on_recovered(self, callback: Callable[[], None]) -> None:
        self._on_recovered_callbacks.append(callback)

    def on_hard_reset(self, callback: Callable[[], None]) -> None:
        self._on_hard_reset_callbacks.append(callback)

    # ── 检测 ──────────────────────────────────────────────────

    def check(self, ocr_texts: list[str],
              card_regions_valid: bool = True) -> bool:
        """检测当前帧是否被弹窗遮挡。

        Args:
            ocr_texts: OCR 提取到的所有文本
            card_regions_valid: 卡片区域形状是否正常

        Returns:
            True 表示需要挂起系统，False 表示正常。
        """
        with self._lock:
            all_text = " ".join(ocr_texts)

            # 检查系统关键词命中
            hits = [kw for kw in self.SYSTEM_KEYWORDS if kw in all_text]

            if not card_regions_valid:
                self._abnormal_count += 1
            elif len(hits) >= 2:
                self._abnormal_count += 1
            else:
                self._abnormal_count = max(0, self._abnormal_count - 1)

            # 触发挂起条件
            if self._abnormal_count >= self.ABNORMAL_THRESHOLD and not self._is_blocked:
                self._is_blocked = True
                self._blocked_since = time.monotonic()
                reason = f"连续 {self._abnormal_count} 帧异常，命中关键词: {hits}"
                logger.warning("弹窗遮挡检测触发: %s", reason)
                self._notify_blocked(reason)
                return True

            return self._is_blocked

    def check_recovery(self, card_regions_valid: bool = True,
                        system_keywords_found: bool = False) -> bool:
        """检查屏幕是否恢复正常。

        Returns:
            True 表示已恢复，False 表示仍被遮挡。
        """
        with self._lock:
            if not self._is_blocked:
                return True

            if card_regions_valid and not system_keywords_found:
                self._abnormal_count = max(0, self._abnormal_count - 1)
                if self._abnormal_count <= 0:
                    self._is_blocked = False
                    self._blocked_since = None
                    logger.info("屏幕遮挡已清除，恢复正常")
                    self._notify_recovered()
                    return True

            # 检查是否需要硬重置
            if self._blocked_since is not None:
                elapsed = time.monotonic() - self._blocked_since
                if elapsed > self.HARD_RESET_TIMEOUT:
                    if self._hard_reset_count < self.MAX_HARD_RESETS:
                        logger.warning("遮挡超时 30min，执行硬重置 (第 %d 次)",
                                       self._hard_reset_count + 1)
                        self._hard_reset_count += 1
                        self._blocked_since = time.monotonic()  # 重置计时器
                        self._notify_hard_reset()
                    else:
                        logger.critical("连续 %d 次硬重置失败，彻底停机",
                                        self.MAX_HARD_RESETS)
                        self._notify_blocked("硬重置耗尽，系统彻底停机")

            return False

    def reset(self) -> None:
        """手动重置遮挡状态。"""
        with self._lock:
            self._abnormal_count = 0
            self._is_blocked = False
            self._blocked_since = None

    # ── 内部 ──────────────────────────────────────────────────

    def _notify_blocked(self, reason: str) -> None:
        for cb in self._on_blocked_callbacks:
            try:
                cb(reason)
            except Exception:
                logger.exception("遮挡告警回调异常")

    def _notify_recovered(self) -> None:
        for cb in self._on_recovered_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("恢复回调异常")

    def _notify_hard_reset(self) -> None:
        for cb in self._on_hard_reset_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("硬重置回调异常")

    # ── 状态 ──────────────────────────────────────────────────

    @property
    def is_blocked(self) -> bool:
        return self._is_blocked

    @property
    def abnormal_count(self) -> int:
        return self._abnormal_count
