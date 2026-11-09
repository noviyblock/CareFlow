"""Cascade latency metrics from SATER (Shen et al., arXiv:2510.05164).

AGL (Average Generation Latency): latency when the reply is completed by the small model M_s.
AROL (Average Routing Overhead Latency): extra latency when M_s fails/abstains and the system falls
back to M_l, compared with calling M_l directly. SATER uses output tokens as the latency proxy; we
report both tokens (hardware-independent) and seconds (M5 Pro, MLX) under the same definitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CascadeStep:
    small_cost: float       # tokens or seconds spent by M_s on this item (incl. an abstention)
    accepted: bool          # True if M_s's reply was kept
    large_cost: float = 0.0  # tokens or seconds of M_l if it was called (fallback); ignored when accepted


def agl(steps: Sequence[CascadeStep]) -> float:
    vals = [s.small_cost for s in steps if s.accepted]
    return float(np.mean(vals)) if vals else float("nan")


def arol(steps: Sequence[CascadeStep]) -> float:
    """Fallback items cost small_cost + large_cost instead of large_cost: the overhead is small_cost."""
    vals = [s.small_cost for s in steps if not s.accepted]
    return float(np.mean(vals)) if vals else float("nan")


def cascade_summary(steps: Sequence[CascadeStep]) -> dict[str, float]:
    total = [s.small_cost + (0.0 if s.accepted else s.large_cost) for s in steps]
    return {
        "agl": agl(steps),
        "arol": arol(steps),
        "small_share": float(np.mean([s.accepted for s in steps])) if steps else float("nan"),
        "mean_cost": float(np.mean(total)) if total else float("nan"),
    }
