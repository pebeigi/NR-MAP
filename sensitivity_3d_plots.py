#!/usr/bin/env python
"""Surface-based sensitivity visualizations for utility components.

With three varied parameters plus a utility value, a single ordinary 3D surface
is not possible because that would require four dimensions. The response-surface
standard is to plot two parameters on x/y, utility on z, and show the third
parameter as low/mid/high slices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("figures/sensitivity/3d")


def directional_utility(s_theta: np.ndarray, theta_error_deg: np.ndarray) -> np.ndarray:
    """Paper Eq. 5: U_dir = S_theta * cos(theta - theta_goal)."""
    return s_theta * np.cos(np.deg2rad(theta_error_deg))


def speed_utility(s_v: np.ndarray, rho_g: np.ndarray, xi_i: np.ndarray) -> np.ndarray:
    """Paper Eq. 6."""
    rho_g = np.maximum(rho_g, 1e-6)
    return s_v * (rho_g / (1 + rho_g ** ((xi_i - 1) / 2)))


def proximity_utility(
    s_d: np.ndarray,
    gamma: np.ndarray,
    d_eff: np.ndarray,
    h_p: np.ndarray,
) -> np.ndarray:
    """Paper Eq. 7."""
    h_p = np.maximum(h_p, 1e-6)
    return s_d / (1 + (d_eff / h_p) ** gamma)


def collision_utility(w_c: np.ndarray, d: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Paper Eqs. 10-12 for a single neighbor with isotropic 2D covariance."""
    sigma = np.maximum(sigma, 1e-6)
    prob_coll = (1.0 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * (d / sigma) ** 2)
    return -w_c * prob_coll


def path_adherence_utility(
    w_ell: np.ndarray,
    beta: np.ndarray,
    ell_i: np.ndarray,
) -> np.ndarray:
    """Paper Eq. 13 as a negative utility penalty."""
    return -w_ell * (1 - np.exp(-beta * ell_i**2))


def plot_response_surface_slices(
    x_values: np.ndarray,
    y_values: np.ndarray,
    slice_values: list[float],
    utility_fn,
    labels: tuple[str, str, str],
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    show: bool = False,
) -> None:
    """Plot utility response surfaces for low/mid/high values of a third parameter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    utilities = [utility_fn(x_grid, y_grid, slice_value) for slice_value in slice_values]
    vmin = min(float(np.nanmin(utility)) for utility in utilities)
    vmax = max(float(np.nanmax(utility)) for utility in utilities)

    fig = plt.figure(figsize=(18, 5.8))
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_clim(vmin, vmax)

    for index, (slice_value, utility_grid) in enumerate(zip(slice_values, utilities), start=1):
        ax = fig.add_subplot(1, len(slice_values), index, projection="3d")
        surface = ax.plot_surface(
            x_grid,
            y_grid,
            utility_grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            antialiased=True,
            alpha=0.96,
        )
        ax.contour(
            x_grid,
            y_grid,
            utility_grid,
            zdir="z",
            offset=vmin,
            cmap=cmap,
            levels=10,
            linewidths=0.7,
        )
        ax.set_title(f"{labels[2]} = {slice_value:g}", fontsize=12)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel("Utility")
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=28, azim=-135)
        mappable = surface

    fig.suptitle(title, fontsize=16)
    colorbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.72, pad=0.04)
    colorbar.set_label("Utility value")
    fig.subplots_adjust(left=0.03, right=0.9, bottom=0.08, top=0.86, wspace=0.12)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_speed_3d(show: bool = False) -> None:
    s_v = np.linspace(0.2, 1.0, 80)
    rho_g = np.linspace(0.2, 2.5, 80)
    xi_slices = [1.5, 2.5, 3.5]
    plot_response_surface_slices(
        s_v,
        rho_g,
        xi_slices,
        utility_fn=lambda s_v_grid, rho_grid, xi: speed_utility(s_v_grid, rho_grid, xi),
        labels=(r"$S_v$", r"$\rho_g$", r"$\xi_i$"),
        title="Speed Utility Response Surfaces",
        output_path=OUTPUT_DIR / "speed_utility_surface_slices.png",
        cmap="viridis",
        show=show,
    )


def plot_proximity_3d(show: bool = False) -> None:
    gamma = np.linspace(0.5, 3.0, 80)
    d_eff = np.linspace(1.0, 150.0, 80)
    h_p_slices = [20.0, 60.0, 100.0]
    plot_response_surface_slices(
        gamma,
        d_eff,
        h_p_slices,
        utility_fn=lambda gamma_grid, d_grid, h_p: proximity_utility(
            1.0,
            gamma_grid,
            d_grid,
            h_p,
        ),
        labels=(r"$\gamma$", r"$d_{\mathrm{eff}}$", r"$H_p$"),
        title="Proximity Utility Response Surfaces",
        output_path=OUTPUT_DIR / "proximity_utility_surface_slices.png",
        cmap="plasma",
        show=show,
    )


def plot_collision_3d(show: bool = False) -> None:
    # Avoid very small sigma values; they create a near-singular spike at d=0.
    w_c = np.linspace(0.1, 20.0, 80)
    d = np.linspace(0.0, 12.0, 80)
    sigma_slices = [1.0, 3.5, 6.0]
    plot_response_surface_slices(
        w_c,
        d,
        sigma_slices,
        utility_fn=lambda w_grid, d_grid, sigma: collision_utility(w_grid, d_grid, sigma),
        labels=(r"$w_c$", r"$\|\vec r_g-\vec \mu_j\|$", r"$\sigma$"),
        title="Collision Utility Response Surfaces",
        output_path=OUTPUT_DIR / "collision_utility_surface_slices.png",
        cmap="magma",
        show=show,
    )


def plot_path_3d(show: bool = False) -> None:
    w_ell = np.linspace(5.0, 50.0, 80)
    beta = np.linspace(0.1, 5.0, 80)
    ell_slices = [0.5, 2.5, 5.0]
    plot_response_surface_slices(
        w_ell,
        beta,
        ell_slices,
        utility_fn=lambda w_grid, beta_grid, ell_i: path_adherence_utility(
            w_grid,
            beta_grid,
            ell_i,
        ),
        labels=(r"$w_{\ell}$", r"$\beta$", r"$\ell_i$"),
        title="Path-Adherence Utility Response Surfaces",
        output_path=OUTPUT_DIR / "path_adherence_utility_surface_slices.png",
        cmap="cividis",
        show=show,
    )


def main(show: bool = False) -> None:
    plot_speed_3d(show=show)
    plot_proximity_3d(show=show)
    plot_collision_3d(show=show)
    plot_path_3d(show=show)
    print(f"Saved 3D sensitivity plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main(show=False)
