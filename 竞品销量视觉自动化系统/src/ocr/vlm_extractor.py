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
        """解析 VLM 返回的 JSON 文本 — 自带 JSON 修复网关。

        VLM 千分之一的概率吐出未转义引号、缺失闭合括号等非标准 JSON。
        严禁直接抛 JSONDecodeError，自动修复后重试。
        """
        # 去除可能的 Markdown 代码块标记
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text.strip())

        data = self._try_parse_json(text)
        if data is None:
            # JSON 修复网关：自动补齐缺失括号、剥离非法字符
            repaired = self._repair_json(text)
            data = self._try_parse_json(repaired)

        if data is None:
            logger.warning("VLM JSON 不可修复: %s", text[:200])
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

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        try:
            return _json.loads(text)
        except (_json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _repair_json(text: str) -> str:
        """JSON 修复网关：自动补齐闭合括号，修复未转义引号。"""
        # 1. 补齐尾部缺失的 }
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces > close_braces:
            text += "}" * (open_braces - close_braces)
        # 2. 补齐尾部缺失的 ]
        open_brackets = text.count("[")
        close_brackets = text.count("]")
        if open_brackets > close_brackets:
            text += "]" * (open_brackets - close_brackets)
        # 3. 尝试移除非法控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text

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

    # ── V5.0 优化：Tile 格栅裁剪 ──────────────────────────────

    # ── V5.0 优化：固定几何尺寸归一化 ─────────────────────────

    @staticmethod
    def normalize_to_fixed(image, target_w: int = 512, target_h: int = 512):
        """固定尺寸归一化 — 等比例缩放 + 黑边填充，杜绝显存碎片。

        RTX 4060 8GB 显存防御：强制所有 VLM 输入统一为 512×512。
        降采样用 INTER_AREA（保小字锐度），放大用 INTER_CUBIC（保边缘平滑）。
        """
        try:
            import cv2
            import numpy as np
            h, w = image.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            # 降采样 vs 放大 → 选择最优插值
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_off = (target_h - new_h) // 2
            x_off = (target_w - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            return canvas
        except ImportError:
            return image

    @staticmethod
    def resize_for_vlm(image, max_pixels: int = 512):
        """VLM 推理前强制下采样，限制 Token 数量。

        1080P 弹窗截图 → 等比例缩放至 max_pixels 边长，
        防止 ViT 动态分 Tile 撑爆显存。
        """
        try:
            import cv2
            h, w = image.shape[:2]
            scale = max_pixels / max(h, w, 1)
            if scale < 1.0:
                return cv2.resize(image, (int(w * scale), int(h * scale)))
        except ImportError:
            pass
        return image

    @staticmethod
    def encode_image_for_vlm(image, max_pixels: int = 512) -> bytes:
        """将图像编码为 VLM API 可接受的 base64 字节流。

        先下采样再编码为 JPEG（压缩质量 85%），平衡清晰度与传输体积。
        """
        import base64
        import cv2

        resized = VLMExtractor.resize_for_vlm(image, max_pixels)
        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "cached_calls": self._total_cached_calls,
            "cache_hit_rate": (
                self._total_cached_calls / max(self._total_calls, 1)
            ),
        }
