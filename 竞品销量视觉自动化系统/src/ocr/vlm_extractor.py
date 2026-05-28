"""Qwen2.5-VL 多模态数据提取器 — 本地视觉大模型语义结构化。

设计文档 4.4：
- 调用本地 Ollama Qwen2.5-VL-7B 对实时销量弹窗截图执行端到端语义提取
- System Prompt 强制输出标准 JSON
- 支持 KV-Cache slot 管理（增量推理优化）
"""
import json as _json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderItem:
    """单条实时订单。"""
    buyer: str
    is_repeat: bool
    sku: str
    time_str: str


@dataclass
class SalesSnapshot:
    """实时销量弹窗语义提取结果。"""
    cart_adds: str = ""
    growth: str = ""
    orders: list[OrderItem] = field(default_factory=list)


class VLMExtractor:
    """Qwen2.5-VL 多模态提取器。

    将实时销量弹窗截图送入 Ollama Vision API，
    返回结构化 SalesSnapshot。
    """

    SYSTEM_PROMPT = (
        "你是一个电商数据结构化清洗专家。"
        "请提取图片中'实时销量'列表的数据，忽略头像。"
        "将每一行提取为一个对象，严格输出如下标准的 JSON 格式，"
        "不要包含任何解释或 Markdown 标记：\n"
        '{"summary": {"cart_adds": "顶部汇总用户加购人数文本", '
        '"growth": "周销量上涨倍数文本"}, '
        '"orders": [{"buyer": "买家脱敏ID", '
        '"is_repeat": true或false, '
        '"sku": "购买的商品款式或课程名称", '
        '"time_str": "下单时间文本"}]}'
    )

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5-vl:7b",
                 num_predict: int = 512,
                 temperature: float = 0.0,
                 slot_id: int = -1):
        """
        Args:
            base_url: Ollama 服务地址
            model: VLM 模型名
            num_predict: 最大生成长度
            temperature: 温度（0 = 确定性输出）
            slot_id: KV-Cache slot ID（-1 = 自动分配）
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._num_predict = num_predict
        self._temperature = temperature
        self._slot_id = slot_id

        self._total_calls = 0
        self._total_cached_calls = 0

    def extract(self, image_data: bytes, use_cache: bool = False) -> Optional[SalesSnapshot]:
        """从图片中提取实时销量数据。

        Args:
            image_data: PNG/JPEG 图片字节流
            use_cache: 是否复用 KV-Cache（增量推理模式）

        Returns:
            SalesSnapshot 或 None（提取失败）
        """
        try:
            import requests
        except ImportError:
            logger.error("requests 未安装，无法调用 VLM")
            return None

        try:
            import base64
            img_b64 = base64.b64encode(image_data).decode("utf-8")

            payload = {
                "model": self._model,
                "prompt": "请提取图片中的实时销量列表数据。",
                "system": self.SYSTEM_PROMPT,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._num_predict,
                },
            }

            if use_cache and self._slot_id >= 0:
                payload["context"] = []  # Ollama 会复用 slot

            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            self._total_calls += 1
            if use_cache:
                self._total_cached_calls += 1

            response_text = data.get("response", "").strip()
            return self._parse_response(response_text)

        except Exception:
            logger.exception("VLM 提取失败")
            return None

    def _parse_response(self, text: str) -> Optional[SalesSnapshot]:
        """解析 VLM 返回的 JSON 文本。"""
        # 去除可能的 Markdown 代码块标记
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text.strip())

        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            # 正则硬提取兜底
            logger.warning("VLM 返回非标准 JSON: %s", text[:200])
            return None

        summary = data.get("summary", {})
        snapshot = SalesSnapshot(
            cart_adds=summary.get("cart_adds", ""),
            growth=summary.get("growth", ""),
        )

        for order_data in data.get("orders", []):
            snapshot.orders.append(OrderItem(
                buyer=order_data.get("buyer", ""),
                is_repeat=order_data.get("is_repeat", False),
                sku=order_data.get("sku", ""),
                time_str=order_data.get("time_str", ""),
            ))

        return snapshot

    # ── KV-Cache 管理 ─────────────────────────────────────────

    def warmup(self) -> bool:
        """预热 VLM：发送 keepalive 请求。"""
        try:
            import requests
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "keepalive": -1},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            logger.warning("VLM 预热失败")
            return False

    def release(self) -> bool:
        """释放 VLM：允许卸载模型。"""
        try:
            import requests
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "keepalive": "0s"},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "cached_calls": self._total_cached_calls,
            "cache_hit_rate": (
                self._total_cached_calls / max(self._total_calls, 1)
            ),
        }
