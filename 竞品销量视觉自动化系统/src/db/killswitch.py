# -*- coding: utf-8 -*-
"""KillSwitch: 全局熔断管理器 — 连接 LayoutGuard 和 Repository"""
import threading

_killed = threading.Event()

def is_killed() -> bool:
    """检查熔断是否已触发。Repository 写入前调用。"""
    return _killed.is_set()

def trigger(reason: str = "") -> None:
    """触发全局写入熔断。"""
    _killed.set()
    import logging
    logging.getLogger("killswitch").critical("数据库写入熔断已触发: %s", reason)

def reset() -> None:
    """重置熔断（新版本适配后调用）。"""
    _killed.clear()
