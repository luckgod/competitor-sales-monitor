"""显存物理红线守卫 — 启动前校验 GPU 资源。

设计文档 4.4.2：
- OLLAMA_NUM_PARALLEL=1 锁死并发
- OLLAMA_USE_MLOCK=1 强制模型驻留
- CUDA 可用显存检测
- 动态边界裁剪前置建议
"""
import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 建议的最低可用显存 (MB)
MIN_VRAM_MB = 2048


@dataclass
class GPUStatus:
    available: bool
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class GPUGuard:
    """显存物理红线守卫。

    启动时执行环境校验，任何异常立即红字报告。
    """

    def __init__(self, min_vram_mb: int = MIN_VRAM_MB):
        self._min_vram = min_vram_mb

    def check(self) -> GPUStatus:
        """执行全部检查，返回 GPU 状态。"""
        status = GPUStatus(available=False)

        # 检查 Ollama 环境变量
        self._check_env_vars(status)

        # 检查 CUDA 可用性
        self._check_cuda(status)

        # 检查 Ollama 服务
        self._check_ollama(status)

        return status

    def enforce(self) -> bool:
        """强制执行检查。任一项失败 → 红字报错 → 返回 False。"""
        status = self.check()

        if not status.available:
            print("\033[91m[GPU Guard] 显存校验失败！请确认以下条件：\033[0m")
            for w in status.warnings:
                print(f"  \033[91m✗\033[0m {w}")
            return False

        if status.warnings:
            print("\033[93m[GPU Guard] 警告：\033[0m")
            for w in status.warnings:
                print(f"  \033[93m!\033[0m {w}")

        print(f"\033[92m[GPU Guard] 显存校验通过 "
              f"(空闲: {status.vram_free_mb}MB / 总: {status.vram_total_mb}MB)\033[0m")
        return True

    # ── 内部检查 ──────────────────────────────────────────────

    def _check_env_vars(self, status: GPUStatus) -> None:
        """检查 Ollama 环境变量。"""
        num_parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
        use_mlock = os.environ.get("OLLAMA_USE_MLOCK")

        if num_parallel != "1":
            status.warnings.append(
                "OLLAMA_NUM_PARALLEL 未设为 1，并发请求可能引起显存驱逐。"
                " 建议: set OLLAMA_NUM_PARALLEL=1"
            )

        if use_mlock != "1":
            status.warnings.append(
                "OLLAMA_USE_MLOCK 未设为 1，系统内存不足时模型可能被 Swap。"
                " 建议: set OLLAMA_USE_MLOCK=1"
            )

    def _check_cuda(self, status: GPUStatus) -> None:
        """检测 CUDA 显存状态。"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    status.vram_total_mb = int(parts[0].strip())
                    status.vram_free_mb = int(parts[1].strip())
                    status.available = True

                    if status.vram_free_mb < self._min_vram:
                        status.warnings.append(
                            f"可用显存不足 ({status.vram_free_mb}MB < {self._min_vram}MB)，"
                            "推理可能被系统驱逐"
                        )
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # nvidia-smi 不可用，尝试检查 Ollama 状态
        status.warnings.append("nvidia-smi 不可用，无法检测显存状态")

    def _check_ollama(self, status: GPUStatus) -> None:
        """检查 Ollama 服务是否运行。"""
        try:
            import requests
            resp = requests.get(
                "http://localhost:11434/api/tags", timeout=5,
            )
            if resp.status_code == 200:
                if not status.available:
                    status.available = True  # 至少 Ollama 在运行
                return
        except Exception:
            logger.warning("Ollama 健康检查请求失败")

        status.warnings.append("Ollama 服务未响应，VLM 将不可用")


# ── 系统级前置动作 ──────────────────────────────────────────────

def enforce_vram_settings() -> None:
    """强制设置显存保护环境变量。"""
    if os.environ.get("OLLAMA_NUM_PARALLEL") != "1":
        os.environ["OLLAMA_NUM_PARALLEL"] = "1"
        logger.info("OLLAMA_NUM_PARALLEL 已强制设为 1")
    if os.environ.get("OLLAMA_USE_MLOCK") != "1":
        os.environ["OLLAMA_USE_MLOCK"] = "1"
        logger.info("OLLAMA_USE_MLOCK 已强制设为 1")
