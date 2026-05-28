"""多账号资产池 — 分摊单账号曝光频次，防黑号熔断。

设计规范（来自设计文档 11.2）：
- 每采集完 N 个店铺，清除 App 缓存，切换至下一个备用小号
- 语义级黑号熔断：连续 15 次非标准文本 → 熔断锁 + 账号隔离
- 隔离账号标记为 Isolated，触发最高级安全告警
"""
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AccountStatus(Enum):
    ACTIVE = "active"
    ISOLATED = "isolated"  # 已被平台关黑屋
    IDLE = "idle"


@dataclass
class Account:
    username: str
    password: str = ""
    status: AccountStatus = AccountStatus.IDLE
    stores_completed: int = 0
    abnormal_count: int = 0  # 连续异常文本计数


@dataclass
class PoolConfig:
    stores_per_account: int = 3          # 每个号最多采集 N 个店铺后切换
    abnormal_threshold: int = 15         # 连续异常次数 → 熔断


class AccountPool:
    """多账号资产池 — 轮询切换 + 黑号熔断。"""

    def __init__(self, config: PoolConfig | None = None):
        self._config = config or PoolConfig()
        self._accounts: list[Account] = []
        self._current_index = -1
        self._lock = threading.RLock()
        self._on_isolated_callbacks: list[Callable[[Account], None]] = []

    # ── 账号管理 ──────────────────────────────────────────────

    def add_account(self, username: str, password: str = "") -> None:
        with self._lock:
            self._accounts.append(Account(username=username, password=password))

    def load_accounts(self, accounts: list[dict]) -> None:
        """批量加载账号。

        Args:
            accounts: [{"username": "...", "password": "..."}, ...]
        """
        with self._lock:
            for acc in accounts:
                self._accounts.append(Account(
                    username=acc["username"],
                    password=acc.get("password", ""),
                ))

    def on_isolated(self, callback: Callable[["Account"], None]) -> None:
        """注册隔离告警回调。"""
        self._on_isolated_callbacks.append(callback)

    # ── 轮询逻辑 ──────────────────────────────────────────────

    @property
    def current(self) -> Optional[Account]:
        with self._lock:
            if self._current_index < 0 or self._current_index >= len(self._accounts):
                return None
            return self._accounts[self._current_index]

    def next_active(self) -> Optional[Account]:
        """获取下一个可用账号，必要时轮换。"""
        with self._lock:
            active = [a for a in self._accounts if a.status != AccountStatus.ISOLATED]
            if not active:
                logger.error("所有账号已被隔离，无可用的活跃账号")
                return None

            current = self.current
            if current is not None and current.status == AccountStatus.ACTIVE:
                # 检查是否需要切换
                if current.stores_completed < self._config.stores_per_account:
                    return current

            # 轮换至下一个
            current_idx = self._accounts.index(current) if current else -1
            for i in range(len(self._accounts)):
                idx = (current_idx + 1 + i) % len(self._accounts)
                acc = self._accounts[idx]
                if acc.status != AccountStatus.ISOLATED:
                    acc.status = AccountStatus.ACTIVE
                    acc.stores_completed = 0
                    self._current_index = idx
                    logger.info("切换至账号: %s", acc.username)
                    return acc

            return None

    def complete_store(self) -> None:
        """标记当前账号完成一个店铺的采集。"""
        with self._lock:
            acc = self.current
            if acc:
                acc.stores_completed += 1

    # ── 黑号熔断 ──────────────────────────────────────────────

    def report_abnormal_text(self) -> None:
        """报告一次异常文本（系统繁忙/验证码/加载失败）。

        连续达到阈值时触发熔断隔离。
        """
        with self._lock:
            acc = self.current
            if acc is None:
                return
            acc.abnormal_count += 1
            if acc.abnormal_count >= self._config.abnormal_threshold:
                self._isolate(acc)

    def reset_abnormal_count(self) -> None:
        """重置异常计数（正常商品文本出现后）。"""
        with self._lock:
            acc = self.current
            if acc:
                acc.abnormal_count = 0

    def _isolate(self, account: Account) -> None:
        """强行熔断：隔离当前账号。"""
        account.status = AccountStatus.ISOLATED
        logger.critical("账号 %s 已被隔离（连续 %d 次异常文本）",
                         account.username, account.abnormal_count)
        for cb in self._on_isolated_callbacks:
            try:
                cb(account)
            except Exception:
                logger.exception("隔离告警回调异常")

    # ── 指标 ──────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total": len(self._accounts),
                "active": sum(1 for a in self._accounts if a.status == AccountStatus.ACTIVE),
                "isolated": sum(1 for a in self._accounts if a.status == AccountStatus.ISOLATED),
                "idle": sum(1 for a in self._accounts if a.status == AccountStatus.IDLE),
                "current": self.current.username if self.current else None,
            }
