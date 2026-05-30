#!/usr/bin/env python
"""Visualize utility-only vs residual-modulated traffic rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from traffic_env import EnvConfig, MultiAgentTrafficEnv
from utility_model import RESIDUAL_PARAM_KEYS

try:
    import torch
    from train_residual_marl import TorchResidualPolicy
except ImportError:
    torch = None
    TorchResidualPolicy = None


AGENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#003f5c",
    "#ffa600",
]
OUTPUT_DIR = Path("figures/simulation")


def load_policy(checkpoint: Path) -> Any | None:
    if torch is None or not checkpoint.exists():
        return None
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    policy = TorchResidualPolicy(payload["obs_dim"], hidden_dim=payload["hidden_dim"])
    policy.residual_scale = payload["residual_scale"]
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy


def record_rollout(
    env: MultiAgentTrafficEnv,
    policy=None,
    explore_std: float = 0.0,
) -> dict[str, Any]:
    """Run one episode and record positions, velocities, and residuals."""
    obs_list = env.reset()
    n_agents = len(env.agents)
    positions: list[list[np.ndarray]] = [[] for _ in range(n_agents)]
    velocities: list[list[np.ndarray]] = [[] for _ in range(n_agents)]
    residuals: list[list[dict[str, float]]] = [[] for _ in range(n_agents)]
    controls: list[list[dict[str, float]]] = [[] for _ in range(n_agents)]
    destinations = [a.dest.copy() for a in env.agents]
    start_positions = [a.pos.copy() for a in env.agents]

    for i in range(n_agents):
        positions[i].append(env.agents[i].pos.copy())
        velocities[i].append(env.agents[i].vel.copy())
        residuals[i].append({})
        controls[i].append({"accel": 0.0, "steering": 0.0})

    done = False
    while not done:
        residual_actions = None
        if policy is not None:
            residual_actions = []
            for obs in obs_list:
                action, _ = policy.act(obs, explore_std=explore_std)
                residual_actions.append(action)

        obs_list, _, done, _ = env.step(residual_actions)

        for i in range(n_agents):
            positions[i].append(env.agents[i].pos.copy())
            velocities[i].append(env.agents[i].vel.copy())
            controls[i].append(dict(env.agents[i].prev_control))
            if residual_actions is not None:
                residuals[i].append(residual_actions[i])
            else:
                residuals[i].append({})

    return {
        "positions": positions,
        "velocities": velocities,
        "residuals": residuals,
        "controls": controls,
        "destinations": destinations,
        "starts": start_positions,
        "metric": env.rollout_metric(),
        "collisions": env.collision_count,
        "steps": env.step_count,
        "road_y": (
            env.config.sim_config["road_y_min"],
            env.config.sim_config["road_y_max"],
        ),
        "highway_length": env.config.highway_length,
    }


def _stack_positions(positions: list[list[np.ndarray]]) -> list[np.ndarray]:
    return [np.array(p) for p in positions]


def plot_trajectories(
    baseline: dict[str, Any],
    residual: dict[str, Any] | None,
    output_path: Path,
) -> None:
    """Side-by-side 2D trajectory plots."""
    n_panels = 2 if residual is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    rolls = [("Utility-only baseline", baseline)]
    if residual is not None:
        rolls.append(("Residual policy", residual))

    for ax, (title, roll) in zip(axes, rolls):
        y_min, y_max = roll["road_y"]
        ax.axhspan(y_min, y_max, color="#e8e8e8", alpha=0.5, zorder=0)
        ax.axhline(y_min, color="#c9a227", lw=2, ls="-")
        ax.axhline(y_max, color="#c9a227", lw=2, ls="-")
        ax.axhline(0, color="white", lw=1, ls="--", alpha=0.8)

        pos_list = _stack_positions(roll["positions"])
        for i, traj in enumerate(pos_list):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=2, alpha=0.85, label=f"Agent {i}")
            ax.scatter(traj[0, 0], traj[0, 1], s=80, c=color, marker="o", edgecolors="k", zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c=color, marker="s", edgecolors="k", zorder=5)
            dest = roll["destinations"][i]
            ax.scatter(dest[0], dest[1], s=120, c=color, marker="*", edgecolors="k", zorder=5)

        ax.set_title(
            f"{title}\nmetric={roll['metric']:.2f}, collisions={roll['collisions']}, steps={roll['steps']}",
            fontsize=12,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_speed_profiles(baseline: dict[str, Any], residual: dict[str, Any] | None, output_path: Path) -> None:
    """Speed vs time for each agent."""
    n_agents = len(baseline["positions"])
    fig, axes = plt.subplots(n_agents, 1, figsize=(10, 3 * n_agents), sharex=True, constrained_layout=True)
    if n_agents == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        b_vel = np.array(baseline["velocities"][i])
        b_speed = np.linalg.norm(b_vel, axis=1)
        t = np.arange(len(b_speed)) * 0.5
        ax.plot(t, b_speed, "-", color=AGENT_COLORS[i], lw=2, label="Baseline")
        if residual is not None:
            r_vel = np.array(residual["velocities"][i])
            r_speed = np.linalg.norm(r_vel, axis=1)
            ax.plot(t, r_speed, "--", color=AGENT_COLORS[i], lw=2, alpha=0.8, label="Residual")
        ax.set_ylabel(f"Agent {i} speed (m/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Agent speed profiles", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_control_profiles(baseline: dict[str, Any], residual: dict[str, Any] | None, output_path: Path) -> None:
    """Acceleration and steering vs time for each agent."""
    n_agents = len(baseline["positions"])
    fig, axes = plt.subplots(n_agents, 2, figsize=(12, 2.5 * n_agents), sharex=True, constrained_layout=True)
    if n_agents == 1:
        axes = np.array([axes])

    dt = 0.5
    for i in range(n_agents):
        t = np.arange(len(baseline["controls"][i])) * dt
        b_ctrl = baseline["controls"][i]
        b_accel = [c.get("accel", 0.0) for c in b_ctrl]
        b_steer = [c.get("steering", 0.0) for c in b_ctrl]
        axes[i, 0].plot(t, b_accel, "-", color=AGENT_COLORS[i], lw=2, label="Baseline")
        axes[i, 1].plot(t, b_steer, "-", color=AGENT_COLORS[i], lw=2, label="Baseline")
        if residual is not None and "controls" in residual:
            r_ctrl = residual["controls"][i]
            r_accel = [c.get("accel", 0.0) for c in r_ctrl]
            r_steer = [c.get("steering", 0.0) for c in r_ctrl]
            axes[i, 0].plot(t, r_accel, "--", color=AGENT_COLORS[i], lw=2, alpha=0.8, label="Residual")
            axes[i, 1].plot(t, r_steer, "--", color=AGENT_COLORS[i], lw=2, alpha=0.8, label="Residual")
        axes[i, 0].set_ylabel(f"Agent {i}\naccel (m/s²)")
        axes[i, 1].set_ylabel(f"Agent {i}\nsteer (rad)")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend(fontsize=8)
            axes[i, 1].legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Utility-selected acceleration and steering", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_residual_heatmap(residual: dict[str, Any], output_path: Path) -> None:
    """Heatmap of |ΔΘ| components over time per agent."""
    n_agents = len(residual["residuals"])
    keys = list(RESIDUAL_PARAM_KEYS)
    fig, axes = plt.subplots(n_agents, 1, figsize=(11, 2.8 * n_agents), constrained_layout=True)
    if n_agents == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        series = residual["residuals"][i]
        mat = np.zeros((len(keys), max(len(series) - 1, 1)))
        for t, delta in enumerate(series[1:], start=0):
            for k, key in enumerate(keys):
                if isinstance(delta, dict):
                    mat[k, t] = abs(delta.get(key, 0.0))
                else:
                    mat[k, t] = abs(float(np.asarray(delta).reshape(-1)[k]))

        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", origin="lower")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([r"$\Delta$" + k.replace("_", r"\_") for k in keys], fontsize=9)
        ax.set_xlabel("Step")
        ax.set_title(f"Agent {i}: utility parameter residuals")
        fig.colorbar(im, ax=ax, label="|ΔΘ|")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_animation(roll: dict[str, Any], output_path: Path, title: str, fps: int = 4) -> None:
    """Animated 2D rollout (GIF)."""
    pos_list = _stack_positions(roll["positions"])
    n_frames = max(len(p) for p in pos_list)
    y_min, y_max = roll["road_y"]

    all_x = np.concatenate([p[:, 0] for p in pos_list])
    all_y = np.concatenate([p[:, 1] for p in pos_list])
    pad = 2.0
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (min(all_y.min(), y_min) - pad, max(all_y.max(), y_max) + pad)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.axhspan(y_min, y_max, color="#e8e8e8", alpha=0.5)
    ax.axhline(y_min, color="#c9a227", lw=2)
    ax.axhline(y_max, color="#c9a227", lw=2)

    lines = []
    points = []
    dest_markers = []
    for i, dest in enumerate(roll["destinations"]):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        (ln,) = ax.plot([], [], "-", color=color, lw=2, alpha=0.85)
        (pt,) = ax.plot([], [], "o", color=color, ms=10, markeredgecolor="k")
        (dm,) = ax.plot([dest[0]], [dest[1]], "*", color=color, ms=14, markeredgecolor="k")
        lines.append(ln)
        points.append(pt)
        dest_markers.append(dm)

    ax.set_title(title)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AGENT_COLORS[i], label=f"Agent {i}", markersize=8)
        for i in range(len(pos_list))
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    def update(frame: int):
        for i, traj in enumerate(pos_list):
            sub = traj[: frame + 1]
            lines[i].set_data(sub[:, 0], sub[:, 1])
            points[i].set_data([sub[-1, 0]], [sub[-1, 1]])
        ax.set_title(f"{title} — step {frame}/{n_frames - 1}")
        return lines + points

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize traffic rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/residual_policy.pt"))
    parser.add_argument(
        "--rllib-checkpoint",
        type=Path,
        default=Path("checkpoints/rllib_ppo/checkpoint_final"),
        help="RLlib checkpoint directory (preferred if it exists)",
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF animation export")
    args = parser.parse_args()

    env_cfg = EnvConfig(max_steps=args.max_steps, num_agents=args.num_agents)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Recording baseline rollout...")
    baseline_env = MultiAgentTrafficEnv(env_cfg, seed=args.seed)
    baseline = record_rollout(baseline_env, policy=None)

    residual = None
    if args.rllib_checkpoint.exists() or list(args.rllib_checkpoint.parent.glob("checkpoint_*")):
        print(f"Recording RLlib residual rollout from {args.rllib_checkpoint}...")
        from eval_rllib import find_latest_checkpoint, load_algorithm, record_rollout as record_rllib

        ckpt = args.rllib_checkpoint
        if ckpt.is_dir() and not (ckpt / "rllib_checkpoint.json").exists():
            ckpt = find_latest_checkpoint(ckpt.parent if ckpt.name.startswith("checkpoint_") else ckpt)
        algo = load_algorithm(ckpt)
        from traffic_gym_env import TrafficMARLEnv

        rllib_env = TrafficMARLEnv(
            {"seed": args.seed, "max_steps": args.max_steps, "num_agents": args.num_agents}
        )
        residual = record_rllib(rllib_env, algo, explore=False)
    else:
        policy = load_policy(args.checkpoint)
        if policy is not None:
            print("Recording legacy PyTorch residual rollout...")
            residual_env = MultiAgentTrafficEnv(env_cfg, seed=args.seed)
            residual = record_rollout(residual_env, policy=policy, explore_std=0.0)
        else:
            print("No checkpoint found — plotting baseline only.")

    plot_trajectories(
        baseline,
        residual,
        OUTPUT_DIR / "trajectories_compare.png",
    )
    print(f"Saved {OUTPUT_DIR / 'trajectories_compare.png'}")

    plot_speed_profiles(baseline, residual, OUTPUT_DIR / "speed_profiles.png")
    print(f"Saved {OUTPUT_DIR / 'speed_profiles.png'}")

    plot_control_profiles(baseline, residual, OUTPUT_DIR / "control_profiles.png")
    print(f"Saved {OUTPUT_DIR / 'control_profiles.png'}")

    if residual is not None:
        plot_residual_heatmap(residual, OUTPUT_DIR / "residual_heatmap.png")
        print(f"Saved {OUTPUT_DIR / 'residual_heatmap.png'}")

    if not args.no_gif:
        print("Rendering baseline animation...")
        save_animation(baseline, OUTPUT_DIR / "baseline_rollout.gif", "Utility-only baseline")
        print(f"Saved {OUTPUT_DIR / 'baseline_rollout.gif'}")
        if residual is not None:
            print("Rendering residual animation...")
            save_animation(residual, OUTPUT_DIR / "residual_rollout.gif", "Residual policy")
            print(f"Saved {OUTPUT_DIR / 'residual_rollout.gif'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
