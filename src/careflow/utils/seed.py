"""Global seeding and device selection.

Caveat: some MPS kernels have no deterministic implementation. With
`warn_only=True` PyTorch warns instead of failing, so bitwise reproducibility
is guaranteed on CPU only; on MPS we report mean ± std over several seeds.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)  # also seeds MPS and CUDA generators
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "mps") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
