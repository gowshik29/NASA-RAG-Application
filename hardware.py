"""Automatic CUDA/CPU selection with conservative VRAM checks."""

from __future__ import annotations

import os


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_free_gb() -> float | None:
    if not cuda_available():
        return None
    try:
        import torch

        free, _total = torch.cuda.mem_get_info()
        return free / (1024**3)
    except Exception:
        return None


def get_device(min_free_gb: float = 1.25) -> str:
    """Return 'cuda' or 'cpu'.

    Override with ORBIT_DEVICE=cuda|cpu. On 4 GB class GPUs, Ollama may already
    occupy VRAM; embeddings fall back to CPU unless enough free memory remains.
    """
    override = os.getenv("ORBIT_DEVICE", "").strip().lower()
    if override in {"cuda", "cpu"}:
        if override == "cuda" and not cuda_available():
            return "cpu"
        return override
    if not cuda_available():
        return "cpu"
    free = cuda_free_gb()
    if free is not None and free < min_free_gb:
        return "cpu"
    return "cuda"


def device_report() -> dict:
    info = {
        "requested": os.getenv("ORBIT_DEVICE", "auto"),
        "selected": get_device(),
        "cuda_available": cuda_available(),
        "cuda_free_gb": cuda_free_gb(),
        "torch_version": None,
        "torch_cuda": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    return info
