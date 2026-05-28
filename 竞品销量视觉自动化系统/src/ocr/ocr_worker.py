"""PaddleOCR 多进程隔离 Worker — 绕过 GIL 锁。

设计文档 4.4.3：
- PaddleOCR 是 CPU 密集型任务，必须在独立子进程中运行
- 通过 multiprocessing.Queue 与主进程通信
- 主进程发送图像，子进程返回 OCR 结果
"""
import logging
import multiprocessing as mp
from typing import Optional

logger = logging.getLogger(__name__)


class OCRWorkerProcess:
    """PaddleOCR 子进程封装 — 物理隔离 GIL。

    用法:
        worker = OCRWorkerProcess()
        worker.start()
        results = worker.ocr(image)  # 非阻塞，通过 Queue 通信
        worker.stop()
    """

    def __init__(self, lang: str = "ch", timeout: float = 30.0):
        self._lang = lang
        self._timeout = timeout
        self._input_queue: Optional[mp.Queue] = None
        self._output_queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        ctx = mp.get_context("spawn")
        self._input_queue = ctx.Queue(maxsize=50)
        self._output_queue = ctx.Queue()

        self._process = ctx.Process(
            target=_ocr_worker_loop,
            args=(self._input_queue, self._output_queue, self._lang),
            daemon=True,
        )
        self._process.start()
        self._started = True
        logger.info("PaddleOCR 子进程已启动 (pid=%d)", self._process.pid)

    def stop(self) -> None:
        if not self._started:
            return
        self._input_queue.put(("__STOP__", None))
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
        self._started = False
        logger.info("PaddleOCR 子进程已停止")

    def ocr(self, image) -> list[dict]:
        """发送图像到子进程执行 OCR。

        Returns:
            [{"text": str, "bbox": list, "confidence": float}, ...]
        """
        if not self._started:
            raise RuntimeError("OCR 子进程未启动，请先调用 start()")

        # 生成请求 ID
        import uuid
        req_id = uuid.uuid4().hex[:8]

        self._input_queue.put((req_id, image))

        try:
            result_id, result = self._output_queue.get(timeout=self._timeout)
            if result_id != req_id:
                logger.warning("OCR 响应 ID 不匹配: %s != %s", result_id, req_id)
            return result or []
        except Exception:
            logger.exception("OCR 子进程通信超时")
            return []


def _ocr_worker_loop(input_q: mp.Queue, output_q: mp.Queue, lang: str):
    """OCR 子进程主循环。"""
    ocr_instance = None

    try:
        from paddleocr import PaddleOCR
        ocr_instance = PaddleOCR(lang=lang, show_log=False)
    except ImportError:
        while True:
            req_id, img = input_q.get()
            if req_id == "__STOP__":
                break
            output_q.put((req_id, []))
        return

    while True:
        try:
            req_id, img = input_q.get()
            if req_id == "__STOP__":
                break

            results = ocr_instance.ocr(img)
            if not results or not results[0]:
                output_q.put((req_id, []))
            else:
                parsed = [
                    {"text": line[1][0], "bbox": line[0], "confidence": line[1][1]}
                    for line in results[0]
                ]
                output_q.put((req_id, parsed))

        except Exception:
            output_q.put((req_id, []))
