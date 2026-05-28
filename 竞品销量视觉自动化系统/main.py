#!/usr/bin/env python3
"""竞品销量纯视觉自动化统计系统 — 主入口。

阶段一：底层通信与链路自愈保障层
- 加载配置，初始化 ADB/scrcpy 控制器
- 启动看门狗守护线程
- 启动生产者-消费者双线程
- 信号处理与优雅退出
"""
import logging
import signal
import sys
import threading
import time
from pathlib import Path

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import load_config
from src.core.adb_controller import ADBController
from src.core.watchdog import Watchdog
from src.core.session import SessionManager
from src.pipeline.queue_manager import ImageQueue
from src.producer.capture import ProducerThread


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/system.log", encoding="utf-8"),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger("main")

    # ── 加载配置 ──────────────────────────────────────────────
    try:
        config = load_config("config/settings.yaml")
    except FileNotFoundError:
        logger.error("未找到 config/settings.yaml，请先创建配置文件")
        sys.exit(1)

    logger.info("=== 竞品销量纯视觉自动化统计系统 启动 ===")

    # ── 初始化各模块 ──────────────────────────────────────────
    adb_cfg = config["adb"]
    scrcpy_cfg = config["scrcpy"]
    queue_cfg = config["queue"]
    wd_cfg = config["watchdog"]
    slide_cfg = config["slide"]
    session_cfg = config["session"]

    adb = ADBController(
        adb_path=adb_cfg["adb_path"],
        scrcpy_path=adb_cfg["scrcpy_path"],
        device_serial=adb_cfg["device_serial"],
    )

    queue = ImageQueue(
        max_size=queue_cfg["max_size"],
        low_watermark=queue_cfg["low_watermark"],
        producer_timeout=queue_cfg["producer_block_timeout"],
    )

    session_mgr = SessionManager(state_file=session_cfg["state_file"])

    watchdog = Watchdog(
        adb=adb,
        check_interval=wd_cfg["check_interval"],
        reconnect_retries=wd_cfg["reconnect_retries"],
        wireless_fallback=wd_cfg["wireless_fallback"],
    )

    producer = ProducerThread(
        adb=adb,
        queue=queue,
        session_mgr=session_mgr,
        slide_min_pause=slide_cfg["min_pause"],
        slide_max_pause=slide_cfg["max_pause"],
        swipe_duration_min=slide_cfg["swipe_duration_min"],
        swipe_duration_max=slide_cfg["swipe_duration_max"],
    )

    # ── 注册看门狗回调 ──────────────────────────────────────
    watchdog.on_recovered(producer.reset_frame_source)

    # 告警回调：当前阶段打印到日志，后续接机器人 API
    def alert_handler(message: str):
        logger.error("[告警] %s", message)

    watchdog.on_alert(alert_handler)

    # ── 优雅退出 ─────────────────────────────────────────────
    shutdown_flag = threading.Event()

    def _shutdown(signum, frame):
        logger.info("收到信号 %s，开始优雅退出", signum)
        shutdown_flag.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 启动各组件 ────────────────────────────────────────────
    logger.info("检查设备连接状态...")
    if not adb.is_device_connected():
        logger.warning("未检测到 ADB 设备，等待连接...")
        if not adb.wait_for_device(timeout=30):
            logger.error("设备未就绪，退出")
            sys.exit(1)

    logger.info("设备已就绪，启动 scrcpy 投屏...")
    adb.launch_scrcpy(
        max_size=scrcpy_cfg["max_size"],
        max_fps=scrcpy_cfg["max_fps"],
        bit_rate=scrcpy_cfg["bit_rate"],
        stay_awake=scrcpy_cfg["stay_awake"],
    )

    logger.info("启动看门狗守护线程...")
    watchdog.start()

    logger.info("启动生产者线程...")
    producer.start()

    logger.info("=== 系统就绪，等待采集任务调度 ===")

    # ── 主循环：等待退出信号 ──────────────────────────────────
    try:
        while not shutdown_flag.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    logger.info("正在停止各组件...")
    producer.stop()
    watchdog.stop()
    adb.kill_scrcpy()
    logger.info("=== 系统已安全退出 ===")


if __name__ == "__main__":
    main()
