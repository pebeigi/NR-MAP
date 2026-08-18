"""IEEE-style appendix figures: three-site geometry and urban calibration-quality montage."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "Appendix"
OUT.mkdir(parents=True, exist_ok=True)

TRAJ = "#5b5b5b"
CENTER = "#1a1a1a"
TUBE_LO = "#c0392b"
TUBE_HI = "#1e8449"
TUBE_FILL = "#9ec9e8"
OUTER = "#1f4e79"
ISLAND = "#a93226"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.55,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=seed)


def _label(ax, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
    )


def _clean(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.tick_params(length=2.5, width=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.55)
    ax.grid(False)


def plot_highway(ax) -> None:
    bound = pd.read_csv(
        ROOT / "data" / "Lebanon_Highway" / "derived_highway_boundaries" / "highway_boundaries.csv"
    )
    traj = pd.read_csv(
        ROOT / "data" / "Lebanon_Highway" / "Final_Lebanon_Data.csv",
        usecols=["xloc_kf", "yloc_kf", "class"],
    )
    cars = traj[traj["class"] == 1.0]
    s = _sample(cars, 90_000)
    ax.scatter(s["xloc_kf"], s["yloc_kf"], s=0.35, alpha=0.12, color=TRAJ, linewidths=0, rasterized=True)
    for (run_id, lane), g in bound.groupby(["run_id", "lane_kf"], sort=True):
        g = g.sort_values("point_index")
        ax.fill(
            np.r_[g["lower_x"].to_numpy(), g["upper_x"].to_numpy()[::-1]],
            np.r_[g["lower_y"].to_numpy(), g["upper_y"].to_numpy()[::-1]],
            color=TUBE_FILL,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ls = "-" if int(run_id) == 1 else "--"
        ax.plot(g["center_x"], g["center_y"], color=CENTER, lw=1.15, ls=ls, zorder=3)
        ax.plot(g["lower_x"], g["lower_y"], color=TUBE_LO, lw=0.65, ls=ls, zorder=2)
        ax.plot(g["upper_x"], g["upper_y"], color=TUBE_HI, lw=0.65, ls=ls, zorder=2)
    _clean(ax)
    _label(ax, "(a) Lebanon Highway")
    ax.legend(
        handles=[
            Line2D([0], [0], color=CENTER, lw=1.15, label="PCA centerline"),
            Line2D([0], [0], color=TUBE_LO, lw=0.8, label="path bounds"),
            Line2D([0], [0], color=TRAJ, marker="o", ls="none", markersize=3, label="class-1 trajectories"),
        ],
        loc="lower right",
        frameon=False,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.25,
    )


def plot_jounieh(ax) -> None:
    bound = pd.read_csv(ROOT / "data" / "Lebanon_Jounieh" / "Jounieh_Road_Boundaries.csv")
    traj = pd.read_csv(
        ROOT / "data" / "Lebanon_Jounieh" / "prepared" / "trajectories_calibration.csv",
        usecols=["xloc_kf", "yloc_kf", "class"],
    )
    cars = traj[traj["class"] == 2.0]
    s = _sample(cars, 80_000)
    ax.scatter(s["xloc_kf"], s["yloc_kf"], s=0.7, alpha=0.16, color=TRAJ, linewidths=0, rasterized=True)
    for (kind, pid), g in bound.groupby(["kind", "polygon_id"], sort=True):
        g = g.sort_values("vertex_index")
        xy = g[["x", "y"]].to_numpy(float)
        color = OUTER if kind == "outer" else ISLAND
        ax.plot(
            np.r_[xy[:, 0], xy[0, 0]],
            np.r_[xy[:, 1], xy[0, 1]],
            lw=1.6 if kind == "outer" else 1.25,
            color=color,
            zorder=3,
        )
    _clean(ax)
    _label(ax, "(b) Jounieh roundabout")
    ax.legend(
        handles=[
            Line2D([0], [0], color=OUTER, lw=1.6, label="outer curb"),
            Line2D([0], [0], color=ISLAND, lw=1.25, label="islands"),
            Line2D([0], [0], color=TRAJ, marker="o", ls="none", markersize=3, label="class-2 trajectories"),
        ],
        loc="lower left",
        frameon=False,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.25,
    )


def plot_tgsim(ax) -> None:
    bound = pd.read_csv(
        ROOT / "data" / "TGSIM FB" / "derived_boundaries" / "street_boundaries.csv"
    )
    traj = pd.read_csv(
        ROOT / "data" / "TGSIM FB" / "prepared" / "trajectories_calibration.csv",
        usecols=["xloc_kf", "yloc_kf", "class"],
    )
    cars = traj[traj["class"] == 3.0]
    s = _sample(cars, 110_000)
    ax.scatter(s["xloc_kf"], s["yloc_kf"], s=0.35, alpha=0.10, color=TRAJ, linewidths=0, rasterized=True)
    for _, part in bound.groupby("part_index"):
        xy = part.sort_values("vertex_index")[["x", "y"]].to_numpy(float)
        ax.plot(xy[:, 0], xy[:, 1], lw=0.9, color=OUTER, zorder=3)
    _clean(ax)
    _label(ax, "(c) TGSIM Foggy Bottom")
    ax.legend(
        handles=[
            Line2D([0], [0], color=OUTER, lw=1.2, label="street curb"),
            Line2D([0], [0], color=TRAJ, marker="o", ls="none", markersize=3, label="class-3 trajectories"),
        ],
        loc="upper right",
        frameon=False,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.25,
    )


def write_site_figure() -> Path:
    fig = plt.figure(figsize=(7.16, 6.85))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.18], hspace=0.18, wspace=0.16)
    ax_h = fig.add_subplot(gs[0, :])
    ax_j = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[1, 1])
    plot_highway(ax_h)
    plot_jounieh(ax_j)
    plot_tgsim(ax_t)
    out = OUT / "fig_datasets_sites.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


def write_fit_montage() -> Path:
    paths = [
        (
            ROOT / "Calibration" / "diagnostics_jounieh" / "calibration_quality.png",
            "(a) Jounieh roundabout",
        ),
        (
            ROOT / "Calibration" / "diagnostics_tgsim" / "calibration_quality.png",
            "(b) TGSIM Foggy Bottom",
        ),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 5.6))
    for ax, (path, lab) in zip(axes, paths):
        im = Image.open(path).convert("RGB")
        ax.imshow(np.asarray(im))
        ax.axis("off")
        ax.set_title(lab, loc="left", fontsize=8, fontweight="bold", pad=3)
    fig.subplots_adjust(hspace=0.08, left=0.01, right=0.99, top=0.96, bottom=0.01)
    out = OUT / "fig_env_calib_fit.png"
    fig.savefig(out, dpi=280, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


if __name__ == "__main__":
    _style()
    p1 = write_site_figure()
    p2 = write_fit_montage()
    print("wrote", p1)
    print("wrote", p2)
