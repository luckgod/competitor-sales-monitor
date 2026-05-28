"""YAML 配置加载器 — 支持环境变量覆盖 ${ENV_VAR:default}。"""
import os
import re
from pathlib import Path
import yaml


_ENV_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env(value: str) -> str:
    def _replace(m: re.Match) -> str:
        var = m.group(1)
        default = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(var, default)
    return _ENV_RE.sub(_replace, value)


def _walk_and_resolve(obj):
    """递归遍历配置对象，替换环境变量占位符。"""
    if isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_resolve(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_env(obj)
    return obj


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """加载 YAML 配置并解析环境变量占位符。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _walk_and_resolve(raw)
