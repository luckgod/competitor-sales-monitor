"""消费者线程 — 图像解析、模糊去重、两道式清洗网关。

线程 B 职责：
1. 从队列中提取大图 → OpenCV 卡片切分
2. 懒加载无效帧过滤（信息熵/灰度方差）
3. pHash 汉明距离模糊去重
4. 正则优先 → LLM 兜底 两道式销量清洗
5. 消费完成后通知队列（背压释放）
"""
import logging
import threading
import time
from typing import Optional

from src.core.session import SessionManager
from src.pipeline.queue_manager import ImageQueue
from src.consumer.card_splitter import CardSplitter, CardRegions
from src.consumer.filter import LazyLoadFilter
from src.consumer.dedup import DedupEngine
from src.ocr.sales_extractor import SalesExtractor

logger = logging.getLogger(__name__)


class ConsumerThread:
    """消费者线程 — 视觉解析流水线。

    处理链路：
    Queue.get() → 卡片切分 → 无效帧过滤 → pHash去重 → Regex清洗 → LLM兜底 → DB入库
    """

    def __init__(self, queue: ImageQueue, session_mgr: SessionManager,
                 dedup_window_size: int = 30,
                 splitter_layout: CardRegions | None = None,
                 splitter_mode: str = "grid",
                 sales_extractor: SalesExtractor | None = None):
        self._queue = queue
        self._session_mgr = session_mgr

        # 阶段三核心模块
        self._splitter = CardSplitter(layout=splitter_layout, mode=splitter_mode)
        self._lazy_filter = LazyLoadFilter()
        self._dedup = DedupEngine(window_size=dedup_window_size)
        self._sales_extractor = sales_extractor or SalesExtractor()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.clear()

        # 处理耗时统计
        self._total_processed = 0
        self._total_discarded_lazy = 0
        self._total_discarded_duplicate = 0
        self._lock = threading.Lock()

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=False, name="consumer")
        self._thread.start()
        logger.info("消费者线程已启动")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def stats(self) -> dict:
        base = {
            "total_processed": self._total_processed,
            "total_discarded_lazy": self._total_discarded_lazy,
            "total_discarded_duplicate": self._total_discarded_duplicate,
        }
        dedup_stats = self._dedup.stats
        extractor_stats = self._sales_extractor.stats
        return {**base, **dedup_stats, **extractor_stats}

    # ── 主循环 ────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            while self._paused.is_set() and not self._stop_event.is_set():
                time.sleep(0.5)

            frame = self._queue.get(timeout=2.0)
            if frame is None:
                continue

            process_start = time.monotonic()

            try:
                cards = self._splitter.split(frame)
                for card in cards:
                    if self._stop_event.is_set():
                        break

                    # 第一道：无效帧过滤（懒加载白块/未渲染）
                    if self._lazy_filter.is_invalid(card):
                        with self._lock:
                            self._total_discarded_lazy += 1
                        continue

                    # OCR 文本提取
                    raw_title = self._extract_text(card.get("title"))
                    raw_sales = self._extract_text(card.get("sales"))

                    # 文本级无效帧检查
                    if self._lazy_filter.is_invalid_by_text(raw_title):
                        with self._lock:
                            self._total_discarded_lazy += 1
                        continue

                    # 第二道：pHash 模糊去重
                    if self._dedup.is_duplicate(raw_title, card.get("image")):
                        with self._lock:
                            self._total_discarded_duplicate += 1
                        continue

                    # 第三道：Regex → LLM 销量清洗
                    sales_value = self._sales_extractor.extract_with_fallback(raw_sales)

                    # 入库（阶段四实现）
                    self._persist_card(card, raw_title, sales_value)

                    with self._lock:
                        self._total_processed += 1

            except Exception:
                logger.exception("消费者处理帧异常")

            elapsed = time.monotonic() - process_start
            self._queue.task_done(elapsed)

    # ── 处理步骤 ──────────────────────────────────────────────

    def _extract_text(self, region) -> str:
        """从图像区域提取文本。当前返回占位，阶段三完整实现需 PaddleOCR。"""
        if region is None:
            return ""
        # 占位：返回非空字符串使得滤网不会误判
        return "placeholder_text"

    def _persist_card(self, card: dict, title: str, sales_value: int) -> None:
        """数据入库。阶段四实现。"""
        pass
