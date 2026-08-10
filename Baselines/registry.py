"""Name -> controller factory, so the benchmark can be driven from the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import Baselines._paths  # noqa: F401
from Baselines.controllers import Controller


def _utility_pt(**kwargs: Any) -> Controller:
    from Baselines.utility_prior import UtilityPriorController

    return UtilityPriorController(**kwargs)


def _utility_pt_logit(**kwargs: Any) -> Controller:
    from Baselines.utility_prior import UtilityPriorController

    kwargs.setdefault("temperature", 1.0)
    kwargs.setdefault("name", "utility_pt_logit")
    return UtilityPriorController(**kwargs)


def _residual_marl(**kwargs: Any) -> Controller:
    from Baselines.residual_marl import ResidualMARLController

    return ResidualMARLController(**kwargs)


def _residual_sigma_frozen(**kwargs: Any) -> Controller:
    """Same residual checkpoint, but Δσ_long / Δσ_lat are forced to zero."""
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("freeze_keys", ("sigma_long", "sigma_lat"))
    kwargs.setdefault("name", "residual_sigma_frozen")
    return ResidualMARLController(**kwargs)


def _residual_collpen(**kwargs: Any) -> Controller:
    """Residual trained with an OBB collision penalty (sparse training by default)."""
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("checkpoint", Path("RL/checkpoints/residual_collpen_policy.pt"))
    kwargs.setdefault("name", "residual_collpen")
    return ResidualMARLController(**kwargs)


def _residual_collpen_dense(**kwargs: Any) -> Controller:
    """Collision-penalty residual trained under dense spawn (stress distribution)."""
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("checkpoint", Path("RL/checkpoints/residual_collpen_dense_policy.pt"))
    kwargs.setdefault("name", "residual_collpen_dense")
    return ResidualMARLController(**kwargs)


def _orca(**kwargs: Any) -> Controller:
    from Baselines.orca import ORCAController

    return ORCAController(**kwargs)


def _social_force(**kwargs: Any) -> Controller:
    from Baselines.social_force import SocialForceController

    return SocialForceController(**kwargs)


def _dwa(**kwargs: Any) -> Controller:
    from Baselines.dwa import DWAController

    return DWAController(**kwargs)


def _mppi(**kwargs: Any) -> Controller:
    from Baselines.mppi import MPPIController

    return MPPIController(**kwargs)


def _frenet(**kwargs: Any) -> Controller:
    from Baselines.frenet_planner import FrenetPlannerController

    return FrenetPlannerController(**kwargs)


def _pure_rl(**kwargs: Any) -> Controller:
    from Baselines.pure_rl import PureRLController

    return PureRLController(**kwargs)


def _pure_rl_safe(**kwargs: Any) -> Controller:
    """Pure RL trained with an explicit collision penalty added to the shared reward."""
    from Baselines.pure_rl import PureRLController

    kwargs.setdefault("checkpoint", Path("Baselines/checkpoints/pure_rl_safe_policy.pt"))
    kwargs.setdefault("name", "pure_rl_safe")
    return PureRLController(**kwargs)


def _marl(algo: str) -> Callable[..., Controller]:
    def factory(**kwargs: Any) -> Controller:
        from Baselines.marl import MARLController

        kwargs.setdefault("algo", algo)
        kwargs.setdefault("name", algo)
        return MARLController(**kwargs)

    return factory


REGISTRY: dict[str, Callable[..., Controller]] = {
    "utility_pt": _utility_pt,
    "utility_pt_logit": _utility_pt_logit,
    "residual_marl": _residual_marl,
    "residual_sigma_frozen": _residual_sigma_frozen,
    "residual_collpen": _residual_collpen,
    "residual_collpen_dense": _residual_collpen_dense,
    "orca": _orca,
    "social_force": _social_force,
    "dwa": _dwa,
    "mppi": _mppi,
    "frenet": _frenet,
    "pure_rl": _pure_rl,
    "pure_rl_safe": _pure_rl_safe,
    "mappo": _marl("mappo"),
    "happo": _marl("happo"),
    "hatrpo": _marl("hatrpo"),
}

DEFAULT_MODELS = [
    "orca",
    "social_force",
    "dwa",
    "mppi",
    "frenet",
    "pure_rl",
    "mappo",
    "happo",
    "hatrpo",
    "utility_pt",
    "residual_marl",
]

# Human-readable labels for tables and figures.
LABELS = {
    "utility_pt": "Utility prior (PT)",
    "utility_pt_logit": "Utility prior (logit choice)",
    "residual_marl": "Residual MARL (ours)",
    "residual_sigma_frozen": "Residual (sigma frozen)",
    "residual_collpen": "Residual + coll. penalty",
    "residual_collpen_dense": "Residual + coll. pen. (dense)",
    "orca": "ORCA",
    "social_force": "Social force (SDP)",
    "dwa": "DWA",
    "mppi": "MPPI",
    "frenet": "Frenet planner",
    "pure_rl": "IPPO (no prior)",
    "pure_rl_safe": "IPPO + collision penalty",
    "mappo": "MAPPO",
    "happo": "HAPPO",
    "hatrpo": "HATRPO",
}


def build_controller(name: str, **kwargs: Any) -> Controller:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def controller_kwargs(
    name: str,
    residual_checkpoint: Path | None = None,
    pure_rl_checkpoint: Path | None = None,
    calibration: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """CLI-level wiring of checkpoints and calibration files."""
    if name in {"residual_marl", "residual_sigma_frozen"}:
        kwargs: dict[str, Any] = {"calibration": calibration}
        if residual_checkpoint is not None:
            kwargs["checkpoint"] = residual_checkpoint
        return kwargs
    if name in {"residual_collpen", "residual_collpen_dense"}:
        kwargs = {"calibration": calibration}
        if residual_checkpoint is not None and "collpen" in residual_checkpoint.name:
            kwargs["checkpoint"] = residual_checkpoint
        return kwargs
    if name in {"utility_pt", "utility_pt_logit"}:
        return {"calibration": calibration}
    if name == "pure_rl":
        return {"checkpoint": pure_rl_checkpoint} if pure_rl_checkpoint is not None else {}
    if name in {"mappo", "happo", "hatrpo"} and checkpoint_dir is not None:
        return {"checkpoint": checkpoint_dir / f"{name}_policy.pt"}
    return {}
