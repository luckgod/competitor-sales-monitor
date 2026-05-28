"""两道式高吞吐销量清洗网关 — Regex 优先，LLM 兜底。

设计原则（来自设计文档 3.4.1）：
- 第一道（规则正则）：微秒级，覆盖 ~90% 常见格式
- 第二道（本地模型兜底）：百毫秒级，仅处理正则失败的 ~10% 异常文本
- 第三道：极端异常标记为脏数据 (-1)

预期收益：
- 5000 单品总耗时：~30min（原方案）→ ~3min（Regex 优先）
- LLM 调用次数：5000 → ~500（仅异常文本）
"""
import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SalesExtractor:
    """销量文本解析器 — 正则主闸门 + LLM 降级通道。"""

    # 常见销量格式正则（按优先级排序，来自设计文档 3.4.1）
    PATTERNS = [
        re.compile(r'月销\s*([\d.]+)\s*万', re.IGNORECASE),   # "月销 1万+" / "月销 2.5万+"
        re.compile(r'已售\s*(\d+)\s*件', re.IGNORECASE),      # "已售 100件"
        re.compile(r'(\d+)\+?\s*人付款', re.IGNORECASE),      # "4200+人付款"
        re.compile(r'付款\s*(\d+)', re.IGNORECASE),            # "付款 4200+"
        re.compile(r'销量\s*(\d+)', re.IGNORECASE),            # "销量 5000+"
        re.compile(r'^(\d+)$'),                                # 纯数字兜底
    ]

    WAN_MULTIPLIER = 10_000  # "万" 换算系数

    def __init__(self, llm_func: Optional[Callable[[str], Optional[int]]] = None):
        """初始化提取器。

        Args:
            llm_func: LLM 兜底函数，签名为 (raw_text: str) -> int | None。
                     为 None 时禁用 LLM 兜底。
        """
        self._llm_func = llm_func
        self._regex_hits = 0
        self._llm_hits = 0
        self._failures = 0

    def extract(self, raw_text: str) -> Optional[int]:
        """正则主闸门 — 微秒级。

        Returns:
            解析出的整数值，None 表示正则未能匹配。
        """
        if not raw_text:
            return None

        text = raw_text.strip()

        for pattern in self.PATTERNS:
            match = pattern.search(text)
            if not match:
                continue

            try:
                num = float(match.group(1))
            except ValueError:
                continue

            if '万' in text:
                result = int(num * self.WAN_MULTIPLIER)
            else:
                result = int(num)

            self._regex_hits += 1
            return result

        return None

    def extract_with_fallback(self, raw_text: str) -> int:
        """完整的带 LLM 兜底的解析入口。

        Returns:
            解析出的整数值。无法解析时返回 -1（脏数据标记）。
        """
        # 第一道：正则主闸门
        result = self.extract(raw_text)
        if result is not None:
            return result

        # 第二道：LLM 语义兜底
        if self._llm_func is not None:
            try:
                llm_result = self._llm_func(raw_text)
                if llm_result is not None:
                    self._llm_hits += 1
                    return llm_result
            except Exception:
                logger.warning("LLM 兜底调用异常: %s", raw_text)

        # 第三道：标记脏数据
        self._failures += 1
        logger.error("无法解析销量文本: '%s'", raw_text)
        return -1

    # ── 指标 ──────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total = self._regex_hits + self._llm_hits + self._failures
        return {
            "regex_hits": self._regex_hits,
            "llm_fallbacks": self._llm_hits,
            "failures": self._failures,
            "regex_coverage": self._regex_hits / max(total, 1),
            "total_processed": total,
        }


# ────────────────────────────────────────────────────────────
# Ollama 网关（LLM 兜底）
# ────────────────────────────────────────────────────────────

class OllamaGateway:
    """本地 Ollama LLM 网关 — 语义清理兜底。

    约束（来自设计文档 5.x Token 熔断）：
    - temperature: 0（确定输出）
    - num_predict: 32（最大生成长度，防废话）
    - 输出必须是纯 JSON {"sales": int}
    """

    SYSTEM_PROMPT = (
        "你是一个数据转换网关。输入任意电商销量文本，你必须换算并只输出形如 "
        '{"sales": 整数} 的纯 JSON 字符串，不要包含 Markdown 标记、不要包含解释。'
    )

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b",
                 num_predict: int = 32,
                 temperature: float = 0.0):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._num_predict = num_predict
        self._temperature = temperature

    def __call__(self, raw_text: str) -> Optional[int]:
        """调用 Ollama 解析销量文本。"""
        return self.parse(raw_text)

    def parse(self, raw_text: str) -> Optional[int]:
        """调用 Ollama API 解析文本，返回整数或 None。"""
        import json as _json

        try:
            import requests
        except ImportError:
            logger.warning("requests 未安装，无法调用 Ollama")
            return None

        payload = {
            "model": self._model,
            "prompt": f"输入文本: {raw_text}\n请转换:",
            "system": self.SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "").strip()

            # 尝试解析 JSON
            result = _json.loads(response_text)
            if isinstance(result, dict) and "sales" in result:
                return int(result["sales"])

            # 尝试直接解析为数字
            return int(response_text)

        except (_json.JSONDecodeError, ValueError, KeyError):
            # 大模型返回了非标准 JSON，尝试正则硬提取
            numbers = re.findall(r'\d+', response_text if 'response_text' in dir() else "")
            if numbers:
                return int(numbers[0])
            logger.warning("Ollama 返回格式异常: %s", response_text)
            return None
        except Exception:
            logger.exception("Ollama 调用失败")
            return None

    def warmup(self) -> bool:
        """预热请求 — 强制模型在跑盘周期内常驻。

        发送 keepalive: -1 请求，阻止 Ollama 因 5 分钟无请求而卸载模型。
        返回 True 表示预热成功。
        """
        import json as _json
        try:
            import requests
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "keepalive": -1},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            logger.warning("Ollama 预热失败，首次请求将有冷启动延迟")
            return False

    def release(self) -> bool:
        """释放模型 — 允许 Ollama 按默认策略卸载以节省功耗。"""
        import json as _json
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
