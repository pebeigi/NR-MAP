"""Paper ablations and denser stress tests on matched seeds.

(1) Ablation — same scenarios, residual variants vs prior (and MAPPO):
      utility_pt | residual_marl | residual_sigma_frozen | residual_collpen | mappo

(2) Stress — denser spawn (more agents, tighter packing) on the same models.

    python -m Baselines.ablation_stress --mode ablation
    python -m Baselines.ablation_stress --mode stress
    python -m Baselines.ablation_stress --mode both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401
from Baselines.metrics import aggregate, metrics_frame
from Baselines.plots import plot_metric_bars, plot_trajectory_grid_frenet
from Baselines.registry import LABELS, build_controller, controller_kwargs
from Baselines.runner import RolloutResult, rollout
from Baselines.scenario import Scenario, build_scenario
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID

ABLATION_MODELS = [
    "utility_pt",
    "residual_marl",
    "residual_sigma_frozen",
    "residual_collpen",
    "mappo",
]

DEFAULT_OUTPUT = Path("Baselines/results")

# Dense traffic: ~2x agents packed into a shorter spawn window.
STRESS_SPAWN = {
    "spawn_s_range": (20.0, 80.0),
    "spawn_lateral_frac": 0.55,
    "min_initial_spacing": 5.0,
}


def _available_models(models: list[str]) -> list[str]:
    """Skip residual_collpen if its checkpoint has not been trained yet."""
    out = []
    for name in models:
        if name == "residual_collpen":
            ckpt = Path("RL/checkpoints/residual_collpen_policy.pt")
            if not ckpt.exists():
                print(f"[skip] {name}: {ckpt} not found (train with --collision-penalty first)")
                continue
        out.append(name)
    return out


def _build_scenarios(args: argparse.Namespace, dense: bool) -> list[Scenario]:
    seeds = list(range(args.seed, args.seed + args.scenarios))
    num_agents = args.stress_agents if dense else args.num_agents
    kwargs: dict = {
        "num_agents": num_agents,
        "max_steps": args.max_steps,
        "dt": args.dt,
        "run_id": args.run_id,
        "lane_kf": args.lane_kf,
    }
    if dense:
        kwargs.update(STRESS_SPAWN)
    return [build_scenario(seed=s, **kwargs) for s in seeds]


def _run_suite(
    label: str,
    scenarios: list[Scenario],
    models: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n=== {label} ===\n"
        f"Corridor run_id={args.run_id}, lane_kf={args.lane_kf} | "
        f"{len(scenarios)} scenarios x {scenarios[0].num_agents} agents x "
        f"{args.max_steps} steps"
    )

    results: dict[str, list[RolloutResult]] = {}
    for model in models:
        kwargs = controller_kwargs(
            model,
            residual_checkpoint=args.residual_checkpoint,
            calibration=args.calibration,
            checkpoint_dir=args.checkpoint_dir,
        )
        controller = build_controller(model, **kwargs)
        model_results = [rollout(scenario, controller) for scenario in scenarios]
        results[model] = model_results
        collisions = np.mean([r.collision_events for r in model_results])
        arrivals = np.mean([(r.arrival_step >= 0).mean() for r in model_results])
        print(
            f"  {LABELS.get(model, model):<32} collisions={collisions:6.2f} | "
            f"arrival={arrivals:5.2f}"
        )

    flat = [r for model_results in results.values() for r in model_results]
    frame = metrics_frame(flat)
    frame.to_csv(output_dir / f"{label}_raw.csv", index=False)
    summary = aggregate(frame)
    summary.to_csv(output_dir / f"{label}_summary.csv", index=False)

    show = summary.set_index("model")
    mean_cols = [
        c
        for c in (
            "mean_collision_events",
            "mean_offroad_rate",
            "mean_arrival_rate",
            "mean_mean_travel_time_s",
            "mean_mean_speed_mps",
            "mean_rms_jerk",
        )
        if c in show.columns
    ]
    print(show[mean_cols].round(3).to_string())

    if not args.no_figures:
        try:
            plot_metric_bars(frame, output_dir / f"{label}_metrics.png")
            first_results = [results[m][0] for m in models if results[m]]
            plot_trajectory_grid_frenet(
                first_results,
                scenarios[0],
                output_dir / f"{label}_trajectories_frenet.png",
            )
        except Exception as exc:
            print(f"[figures] skipped: {exc}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual ablations and dense stress tests")
    parser.add_argument("--mode", choices=("ablation", "stress", "both"), default="both")
    parser.add_argument("--models", nargs="+", default=ABLATION_MODELS)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=10, help="agents for standard ablation")
    parser.add_argument("--stress-agents", type=int, default=18, help="agents for dense stress")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--residual-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    models = _available_models(list(args.models))
    if not models:
        raise SystemExit("No models available to evaluate.")

    summaries = {}
    if args.mode in {"ablation", "both"}:
        scenarios = _build_scenarios(args, dense=False)
        summaries["ablation"] = _run_suite(
            "ablation", scenarios, models, args, args.output_dir / "ablation"
        )

    if args.mode in {"stress", "both"}:
        scenarios = _build_scenarios(args, dense=True)
        summaries["stress"] = _run_suite(
            "stress", scenarios, models, args, args.output_dir / "stress"
        )

    # Side-by-side delta table for paper: residual − prior under each regime.
    rows = []
    for regime, summary in summaries.items():
        indexed = summary.set_index("model") if "model" in summary.columns else summary
        if "utility_pt" not in indexed.index or "residual_marl" not in indexed.index:
            continue
        for metric in ("collision_events", "arrival_rate", "offroad_rate", "rms_jerk"):
            col = f"mean_{metric}"
            if col not in indexed.columns:
                continue
            prior = float(indexed.loc["utility_pt", col])
            residual = float(indexed.loc["residual_marl", col])
            rows.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "utility_pt": prior,
                    "residual_marl": residual,
                    "delta_residual_minus_prior": residual - prior,
                }
            )
    if rows:
        delta = pd.DataFrame(rows)
        path = args.output_dir / "ablation_stress_delta.csv"
        delta.to_csv(path, index=False)
        print(f"\nWrote {path}")
        print(delta.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
