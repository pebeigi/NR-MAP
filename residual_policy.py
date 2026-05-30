"""Residual policy π(o) -> ΔΘ (Paper Eqs. 14, 18)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utility_model import RESIDUAL_PARAM_KEYS


class ResidualPolicy:
    """
    Small MLP policy mapping local observations to bounded utility residuals.

    Output order matches RESIDUAL_PARAM_KEYS:
    [ΔS_v, ΔS_θ, ΔS_d, ΔW_cc, Δξ_i, Δγ, Δw_p]
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        residual_scale: float = 0.25,
        seed: int | None = None,
    ):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale
        self.rng = np.random.default_rng(seed)

        # Xavier-style init for tanh MLP: obs -> hidden -> residual_dim
        self.w1 = self.rng.normal(0, np.sqrt(2 / obs_dim), (obs_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = self.rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, len(RESIDUAL_PARAM_KEYS)))
        self.b2 = np.zeros(len(RESIDUAL_PARAM_KEYS))

    def forward(self, obs: np.ndarray) -> np.ndarray:
        x = np.tanh(obs @ self.w1 + self.b1)
        raw = np.tanh(x @ self.w2 + self.b2)
        return raw * self.residual_scale

    def act(self, obs: np.ndarray) -> dict[str, float]:
        delta = self.forward(obs)
        return dict(zip(RESIDUAL_PARAM_KEYS, delta.astype(float)))

    def act_batch(self, observations: list[np.ndarray]) -> list[dict[str, float]]:
        return [self.act(obs) for obs in observations]

    def parameters(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in self.parameters()])

    def set_flat_params(self, flat: np.ndarray) -> None:
        idx = 0
        for param in self.parameters():
            size = param.size
            param[:] = flat[idx : idx + size].reshape(param.shape)
            idx += size

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "obs_dim": self.obs_dim,
            "hidden_dim": self.hidden_dim,
            "residual_scale": self.residual_scale,
            "flat_params": self.get_flat_params().tolist(),
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "ResidualPolicy":
        payload = json.loads(Path(path).read_text())
        policy = cls(
            obs_dim=payload["obs_dim"],
            hidden_dim=payload["hidden_dim"],
            residual_scale=payload["residual_scale"],
        )
        policy.set_flat_params(np.array(payload["flat_params"], dtype=float))
        return policy
