#!/usr/bin/env python3
"""随动采集启动脚本 — 加载靶点池 → 随机打散 → 轮询采集。

V5.0 双引擎动态路由：
  路径 A: 拓扑发现（高频滑动 + pHash + 早停）→ 深挖队列
  路径 B: 关键词精准直搜 → 详情页 VLM 深挖 → 24h 熔断截断
"""
import logging
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保项目根在 path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import load_config
from src.core.adb_controller import ADBController
from src.core.watchdog import Watchdog
from src.core.session import SessionManager
from src.core.target_loader import TargetLoader
from src.pipeline.queue_manager import ImageQueue
from src.producer.capture import ProducerThread


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    setup_logging()
    logger = logging.getLogger("run")

    # ── 加载靶点池 ────────────────────────────────────────────
    loader = TargetLoader("config/targets.yaml")
    targets = loader.load()
    if not targets:
        logger.error("靶点池为空，请检查 config/targets.yaml")
        sys.exit(1)

    shuffled = loader.shuffled()
    print(f"\n{'='*60}")
    print(f"  竞品靶点池: {loader.tier1_count} 家核心 + {loader.tier2_count} 家大盘")
    print(f"  执行顺序 (随机打散):")
    for i, t in enumerate(shuffled[:10]):
        print(f"    {i+1}. [{t.tier}] {t.store_name}")
    if len(shuffled) > 10:
        print(f"    ... 共 {len(shuffled)} 家")
    print(f"{'='*60}\n")

    # ── 环境初始化 ────────────────────────────────────────────
    config = load_config("config/settings.yaml")
    adb_cfg = config["adb"]
    scrcpy_cfg = config["scrcpy"]
    slide_cfg = config["slide"]

    adb = ADBController(
        adb_path=adb_cfg["adb_path"],
        scrcpy_path=adb_cfg["scrcpy_path"],
        device_serial=adb_cfg["device_serial"],
    )

    if not adb.is_device_connected():
        logger.error("未检测到设备，请连接手机后重试")
        sys.exit(1)

    logger.info("设备在线: %s", adb.list_devices()[0].serial)

    # 启动 scrcpy（无窗口模式，纯管道帧捕获）
    logger.info("启动 scrcpy 投屏...")
    adb.launch_scrcpy(
        max_size=scrcpy_cfg["max_size"],
        max_fps=scrcpy_cfg["max_fps"],
        bit_rate=scrcpy_cfg["bit_rate"],
        stay_awake=scrcpy_cfg["stay_awake"],
    )

    # ── 信号处理 ──────────────────────────────────────────────
    shutdown = False

    def _on_signal(signum, frame):
        nonlocal shutdown
        logger.info("收到退出信号")
        shutdown = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── 主循环：轮询店铺 ──────────────────────────────────────
    store_index = 0
    total_stores = len(shuffled)

    try:
        while store_index < total_stores and not shutdown:
            target = shuffled[store_index]
            store_index += 1

            print(f"\n{'─'*50}")
            print(f"  [{store_index}/{total_stores}] [{target.tier}] {target.store_name}")
            print(f"  关键词: {target.keywords or '(自动搜索)'}")
            print(f"{'─'*50}")

            logger.info("开始采集: %s", target.store_name)

            # === 路径 A: 拓扑发现（占位 — 快速扫描商品列表）===
            logger.info("  [路径A] 拓扑发现中...")
            time.sleep(2)  # 模拟：实际执行高频滑动 + pHash 切分

            # === 路径 B: 深挖（占位 — 进入详情页提取订单）===
            if target.keywords:
                logger.info("  [路径B] 关键词搜索: %s", target.keywords[0])
            else:
                logger.info("  [路径B] 按商品列表顺序深挖")
            time.sleep(1)  # 模拟：实际执行 VLM 多模态提取 + 24h 熔断

            logger.info("  完成: %s (发现 0 新品, 提取 0 笔订单)", target.store_name)

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        adb.kill_scrcpy()
        logger.info(f"采集结束。已处理 {store_index}/{total_stores} 家店铺")


if __name__ == "__main__":
    main()
