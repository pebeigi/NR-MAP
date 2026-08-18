"""Load calibrated utility parameters for residual MARL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from RL._paths import REPO_ROOT
from utility_model import (
    DEFAULT_KERNEL_PARAMS,
    DEFAULT_SIGMA_LAT,
    DEFAULT_SIGMA_LONG,
)

DEFAULT_CALIBRATION_PATH = REPO_ROOT / "Calibration" / "utility_calibration.json"

# Residual policy modulates these terms (weights + collision-kernel scales).
# Kept here so the RL package does not depend on mutating utility_model's clip box.
RESIDUAL_PARAM_KEYS = (
    "S_v",
    "S_theta",
    "S_d",
    "w_c",
    "xi_i",
    "gamma",
    "w_ell",
    "sigma_long",
    "sigma_lat",
)

# Absolute ΔΘ bounds large enough to matter at calibrated magnitudes.
DEFAULT_RESIDUAL_SCALES: dict[str, float] = {
    "S_v": 1.5,
    "S_theta": 1.5,
    "S_d": 2.0,
    "w_c": 250.0,
    "xi_i": 1.0,
    "gamma": 1.0,
    "w_ell": 15.0,
    "sigma_long": 1.5,
    "sigma_lat": 0.6,
}

# Clip bounds aligned with calibrated / near-optimal ranges (not the old GSA box).
RL_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "S_theta": (0.05, 12.0),
    "S_v": (0.05, 12.0),
    "xi_i": (1.0, 10.0),
    "S_d": (0.05, 12.0),
    "gamma": (0.05, 10.0),
    "w_x": (0.05, 15.0),
    "w_y": (0.05, 15.0),
    "w_c": (0.01, 1200.0),
    "w_ell": (0.1, 300.0),
    "beta": (0.01, 12.0),
    "sigma_long": (0.3, 6.0),
    "sigma_lat": (0.2, 3.0),
}


def residual_vector_to_dict(residual: Any) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(residual, dtype=float).reshape(-1)
    if arr.size != len(RESIDUAL_PARAM_KEYS):
        raise ValueError(f"Expected {len(RESIDUAL_PARAM_KEYS)} residuals, got {arr.size}")
    return dict(zip(RESIDUAL_PARAM_KEYS, arr.astype(float)))


def clip_params_rl(params: dict[str, float]) -> dict[str, float]:
    import numpy as np

    out = {**DEFAULT_KERNEL_PARAMS, **params}
    for key, (lo, hi) in RL_PARAM_BOUNDS.items():
        if key in out:
            out[key] = float(np.clip(out[key], lo, hi))
    return out


def apply_residual(
    base_params: dict[str, float],
    delta_theta: dict[str, float] | None,
) -> dict[str, float]:
    """Θ_i = Θ_base + ΔΘ_i, then clip with RL-aligned bounds."""
    merged = {**DEFAULT_KERNEL_PARAMS, **base_params}
    if delta_theta:
        for key in RESIDUAL_PARAM_KEYS:
            merged[key] = float(merged.get(key, 0.0)) + float(delta_theta.get(key, 0.0))
    return clip_params_rl(merged)


def load_base_params(
    path: Path | None = None,
    prefer: str = "robust",
) -> dict[str, float]:
    """
    Load Θ_base from a calibration JSON.

    prefer: "robust" (default) | "best"

    Older calibration files without sigma_* get vehicle-scale defaults filled in.
    """
    path = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if prefer == "best" and "best_params" in payload:
        params = payload["best_params"]
    elif "robust_params" in payload:
        params = payload["robust_params"]
    elif "best_params" in payload:
        params = payload["best_params"]
    else:
        params = payload

    out = {str(k): float(v) for k, v in params.items()}
    out.setdefault("sigma_long", DEFAULT_SIGMA_LONG)
    out.setdefault("sigma_lat", DEFAULT_SIGMA_LAT)
    return out
