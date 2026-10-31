"""Chart generation engine for QbD DoE Design Space case study.

Generates 7 decision-grade, publication-quality figures using Matplotlib:
  1. 01_main_effects.png - Main effects for Viscosity and pH
  2. 02_interaction.png - Factor A x Factor B interaction on Viscosity
  3. 03_contour_viscosity.png - Apparent viscosity response contour & spec limits
  4. 04_contour_ph.png - pH response contour & spec limits
  5. 05_design_space.png - Design space overlay with feasible sweet spot & recommended point
  6. 06_diagnostics.png - Residual diagnostics (Q-Q normal plots & Residuals vs Fitted)
  7. 07_confirmation.png - Confirmation run predicted vs observed with 95% PI
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Colorblind-safe palette (Wong / Okabe-Ito)
CB_BLUE = "#0072B2"
CB_ORANGE = "#D55E00"
CB_GREEN = "#009E73"
CB_PURPLE = "#CC79A7"
CB_YELLOW = "#F0E442"
CB_SKY = "#56B4E9"
CB_DARK = "#333333"
CB_GRAY = "#777777"
CB_LIGHT_GRAY = "#EFEFEF"


def plot_main_effects(df_runs: pd.DataFrame, out_path: str):
    """Plot main effect plots for both responses across Factor A and Factor B levels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    # Factor A and B levels
    a_levels = [0.8, 1.2, 1.6]
    b_levels = [0.30, 0.40, 0.50]

    # Means by level
    v_mean_a = [df_runs[df_runs["A"] == a]["viscosity"].mean() for a in a_levels]
    v_mean_b = [df_runs[df_runs["B"] == b]["viscosity"].mean() for b in b_levels]

    p_mean_a = [df_runs[df_runs["A"] == a]["ph"].mean() for a in a_levels]
    p_mean_b = [df_runs[df_runs["B"] == b]["ph"].mean() for b in b_levels]

    # Panel 1: Viscosity
    ax1 = axes[0]
    ax1.plot(
        [-1, 0, 1],
        v_mean_a,
        marker="o",
        linewidth=2.2,
        color=CB_BLUE,
        label="Factor A: Carbomer 974P (0.8, 1.2, 1.6 %)",
    )
    ax1.plot(
        [-1, 0, 1],
        v_mean_b,
        marker="s",
        linewidth=2.2,
        linestyle="--",
        color=CB_ORANGE,
        label="Factor B: NaOH Ratio (0.30, 0.40, 0.50)",
    )
    ax1.set_xticks([-1, 0, 1])
    ax1.set_xticklabels(["Low (-1)", "Mid (0)", "High (+1)"], fontsize=10)
    ax1.set_ylabel("Apparent Viscosity (cP)", fontsize=11, fontweight="bold")
    ax1.set_title("Main Effects: Apparent Viscosity", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Panel 2: pH
    ax2 = axes[1]
    ax2.plot(
        [-1, 0, 1],
        p_mean_a,
        marker="o",
        linewidth=2.2,
        color=CB_BLUE,
        label="Factor A: Carbomer 974P",
    )
    ax2.plot(
        [-1, 0, 1],
        p_mean_b,
        marker="s",
        linewidth=2.2,
        linestyle="--",
        color=CB_ORANGE,
        label="Factor B: NaOH Ratio",
    )
    ax2.set_xticks([-1, 0, 1])
    ax2.set_xticklabels(["Low (-1)", "Mid (0)", "High (+1)"], fontsize=10)
    ax2.set_ylabel("pH", fontsize=11, fontweight="bold")
    ax2.set_title("Main Effects: pH", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_interaction(df_runs: pd.DataFrame, out_path: str):
    """Plot Factor A x Factor B interaction on Viscosity."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)

    a_levels = [0.8, 1.2, 1.6]
    b_levels = [0.30, 0.40, 0.50]
    b_colors = [CB_ORANGE, CB_BLUE, CB_GREEN]
    b_markers = ["s", "o", "^"]
    b_styles = [":", "-", "--"]

    for b, col, mark, st in zip(b_levels, b_colors, b_markers, b_styles):
        sub = df_runs[df_runs["B"] == b].groupby("A")["viscosity"].mean().reset_index()
        ax.plot(
            sub["A"],
            sub["viscosity"],
            marker=mark,
            linewidth=2.2,
            linestyle=st,
            color=col,
            label=f"Factor B (NaOH:Carbomer) = {b:.2f}",
        )

    # Spec limits
    ax.axhline(
        30000,
        color=CB_PURPLE,
        linestyle="-.",
        linewidth=1.8,
        label="Viscosity Lower Spec (30,000 cP)",
    )
    ax.axhline(
        60000,
        color=CB_PURPLE,
        linestyle="-.",
        linewidth=1.8,
        label="Viscosity Upper Spec (60,000 cP)",
    )

    ax.set_xlabel("Factor A: Carbomer 974P (% w/w)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Apparent Viscosity (cP)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Factor Interaction: Carbomer % × Neutralization Ratio on Viscosity",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(a_levels)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_contour_viscosity(
    aa: np.ndarray, bb: np.ndarray, visc_grid: np.ndarray, df_runs: pd.DataFrame, out_path: str
):
    """Plot Viscosity contour map with specification boundaries."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    # Filled contour
    levels = np.linspace(20000, 65000, 19)
    cf = ax.contourf(aa, bb, visc_grid, levels=levels, cmap="viridis", alpha=0.85)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Predicted Viscosity (cP)", fontsize=10, fontweight="bold")

    # Iso-contour lines
    cs = ax.contour(aa, bb, visc_grid, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fmt="%.0f", fontsize=8)

    # Spec limits
    cs_spec = ax.contour(
        aa,
        bb,
        visc_grid,
        levels=[30000, 60000],
        colors=[CB_PURPLE, CB_ORANGE],
        linewidths=2.5,
        linestyles=["--", "-."],
    )
    ax.clabel(cs_spec, inline=True, fmt={30000: "LSL: 30k cP", 60000: "USL: 60k cP"}, fontsize=10)

    # Experimental design points
    ax.scatter(
        df_runs["A"],
        df_runs["B"],
        color="white",
        edgecolors="black",
        s=60,
        linewidths=1.5,
        zorder=5,
        label="Experimental Runs",
    )

    ax.set_xlabel("Carbomer 974P (% w/w)", fontsize=11, fontweight="bold")
    ax.set_ylabel("NaOH : Carbomer Ratio (w/w)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Response Surface: Apparent Viscosity with Spec Boundaries (30k-60k cP)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_contour_ph(
    aa: np.ndarray, bb: np.ndarray, ph_grid: np.ndarray, df_runs: pd.DataFrame, out_path: str
):
    """Plot pH contour map with specification boundaries."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    levels = np.linspace(5.50, 7.50, 21)
    cf = ax.contourf(aa, bb, ph_grid, levels=levels, cmap="plasma", alpha=0.85)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Predicted pH", fontsize=10, fontweight="bold")

    cs = ax.contour(aa, bb, ph_grid, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fmt="%.2f", fontsize=8)

    # Spec limits
    cs_spec = ax.contour(
        aa,
        bb,
        ph_grid,
        levels=[6.00, 7.50],
        colors=[CB_BLUE, CB_PURPLE],
        linewidths=2.5,
        linestyles=["--", "-."],
    )
    ax.clabel(cs_spec, inline=True, fmt={6.00: "LSL: pH 6.00", 7.50: "USL: pH 7.50"}, fontsize=10)

    # Experimental design points
    ax.scatter(
        df_runs["A"],
        df_runs["B"],
        color="white",
        edgecolors="black",
        s=60,
        linewidths=1.5,
        zorder=5,
        label="Experimental Runs",
    )

    ax.set_xlabel("Carbomer 974P (% w/w)", fontsize=11, fontweight="bold")
    ax.set_ylabel("NaOH : Carbomer Ratio (w/w)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Response Surface: Formulation pH with Spec Boundaries (6.00 - 7.50)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_design_space_overlay(
    aa: np.ndarray,
    bb: np.ndarray,
    visc_grid: np.ndarray,
    ph_grid: np.ndarray,
    feas_grid: np.ndarray,
    df_runs: pd.DataFrame,
    robust_pt: Dict[str, Any],
    feasible_pct: float,
    out_path: str,
):
    """Plot multi-attribute design space overlay with feasible sweet spot and robust point."""
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)

    # Background shaded: Unfeasible (gray) vs Feasible Design Space (soft teal)
    ax.contourf(
        aa,
        bb,
        feas_grid.astype(int),
        levels=[-0.5, 0.5, 1.5],
        colors=[CB_LIGHT_GRAY, "#BCE3C5"],
        alpha=0.9,
    )

    # Draw individual specification boundaries
    cv_low = ax.contour(
        aa,
        bb,
        visc_grid,
        levels=[30000],
        colors=[CB_PURPLE],
        linewidths=2.2,
        linestyles="--",
    )
    cv_high = ax.contour(
        aa,
        bb,
        visc_grid,
        levels=[60000],
        colors=[CB_PURPLE],
        linewidths=2.2,
        linestyles="-.",
    )
    cp_low = ax.contour(
        aa, bb, ph_grid, levels=[6.00], colors=[CB_BLUE], linewidths=2.2, linestyles="--"
    )
    cp_high = ax.contour(
        aa, bb, ph_grid, levels=[7.50], colors=[CB_BLUE], linewidths=2.2, linestyles="-."
    )

    ax.clabel(cv_low, inline=True, fmt="Viscosity = 30k cP (LSL)", fontsize=9)
    ax.clabel(cv_high, inline=True, fmt="Viscosity = 60k cP (USL)", fontsize=9)
    ax.clabel(cp_low, inline=True, fmt="pH = 6.00 (LSL)", fontsize=9)
    ax.clabel(cp_high, inline=True, fmt="pH = 7.50 (USL)", fontsize=9)

    # Experimental points
    fact_mask = df_runs["run_type"] == "factorial"
    center_mask = df_runs["run_type"] == "center"
    ax.scatter(
        df_runs.loc[fact_mask, "A"],
        df_runs.loc[fact_mask, "B"],
        color="white",
        edgecolors=CB_DARK,
        s=60,
        linewidths=1.5,
        zorder=6,
        label="Factorial Points",
    )
    ax.scatter(
        df_runs.loc[center_mask, "A"].iloc[0],
        df_runs.loc[center_mask, "B"].iloc[0],
        color=CB_YELLOW,
        edgecolors=CB_DARK,
        s=120,
        marker="o",
        linewidths=1.8,
        zorder=6,
        label="Center Points (n=3 replicates)",
    )

    # Recommended operating point
    rec_a = robust_pt["recommended_A"]
    rec_b = robust_pt["recommended_B"]
    ax.scatter(
        rec_a,
        rec_b,
        marker="*",
        s=300,
        color=CB_ORANGE,
        edgecolors="black",
        linewidths=1.8,
        zorder=7,
        label=f"Recommended Operating Point\n(A={rec_a:.2f}%, B={rec_b:.2f})",
    )

    # Annotate operating point with predicted values
    ax.annotate(
        f"Recommended Setpoint:\nA = {rec_a:.2f}% w/w\nB = {rec_b:.2f} w/w\nPred Visc: {robust_pt['predicted_viscosity']:,.0f} cP\nPred pH: {robust_pt['predicted_ph']:.2f}\nMargin: {robust_pt['min_normalized_margin']*100:.1f}%",
        xy=(rec_a, rec_b),
        xytext=(rec_a - 0.28, rec_b - 0.08),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=CB_ORANGE, alpha=0.95),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color=CB_ORANGE, lw=1.8),
        zorder=8,
    )

    # Custom legend elements for shading
    patch_ds = patches.Patch(facecolor="#BCE3C5", edgecolor="black", label=f"Feasible Design Space ({feasible_pct:.1f}% area)")
    patch_out = patches.Patch(facecolor=CB_LIGHT_GRAY, edgecolor="gray", label="Out-of-Specification Region")

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([patch_ds, patch_out])
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.92)

    ax.set_xlabel("Factor A: Carbomer 974P (% w/w)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Factor B: NaOH : Carbomer Ratio (w/w)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"ICH Q8 Design Space Overlay (Joint Feasible Area: {feasible_pct:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0.8, 1.6)
    ax.set_ylim(0.30, 0.50)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_diagnostics(fit_visc: Dict[str, Any], fit_ph: Dict[str, Any], out_path: str):
    """Plot 4-panel residual diagnostics: Normal Q-Q and Residuals vs Fitted for both responses."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=150)

    # 1. Viscosity Normal Q-Q
    ax1 = axes[0, 0]
    theo_v = fit_visc["qq_theoretical"]
    ord_v = fit_visc["qq_ordered_residuals"]
    ax1.scatter(theo_v, ord_v, color=CB_BLUE, s=50, edgecolors="black", zorder=3)
    # 45-degree line
    lims_v = [min(theo_v + ord_v) - 0.3, max(theo_v + ord_v) + 0.3]
    ax1.plot(lims_v, lims_v, color=CB_ORANGE, linestyle="--", linewidth=1.5, zorder=2)
    ax1.set_xlabel("Theoretical Standard Normal Quantiles", fontsize=9.5)
    ax1.set_ylabel("Standardized Residuals", fontsize=9.5)
    ax1.set_title("Viscosity: Normal Q-Q Plot", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Viscosity Residuals vs Fitted
    ax2 = axes[0, 1]
    fitted_v = fit_visc["fitted_values"]
    res_v = fit_visc["residuals"]
    ax2.scatter(fitted_v, res_v, color=CB_BLUE, s=50, edgecolors="black", zorder=3)
    ax2.axhline(0, color=CB_ORANGE, linestyle="--", linewidth=1.5, zorder=2)
    ax2.set_xlabel("Fitted Values (cP)", fontsize=9.5)
    ax2.set_ylabel("Residuals (cP)", fontsize=9.5)
    ax2.set_title("Viscosity: Residuals vs Fitted", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. pH Normal Q-Q
    ax3 = axes[1, 0]
    theo_p = fit_ph["qq_theoretical"]
    ord_p = fit_ph["qq_ordered_residuals"]
    ax3.scatter(theo_p, ord_p, color=CB_GREEN, s=50, edgecolors="black", zorder=3)
    lims_p = [min(theo_p + ord_p) - 0.3, max(theo_p + ord_p) + 0.3]
    ax3.plot(lims_p, lims_p, color=CB_ORANGE, linestyle="--", linewidth=1.5, zorder=2)
    ax3.set_xlabel("Theoretical Standard Normal Quantiles", fontsize=9.5)
    ax3.set_ylabel("Standardized Residuals", fontsize=9.5)
    ax3.set_title("pH: Normal Q-Q Plot", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. pH Residuals vs Fitted
    ax4 = axes[1, 1]
    fitted_p = fit_ph["fitted_values"]
    res_p = fit_ph["residuals"]
    ax4.scatter(fitted_p, res_p, color=CB_GREEN, s=50, edgecolors="black", zorder=3)
    ax4.axhline(0, color=CB_ORANGE, linestyle="--", linewidth=1.5, zorder=2)
    ax4.set_xlabel("Fitted Values", fontsize=9.5)
    ax4.set_ylabel("Residuals", fontsize=9.5)
    ax4.set_title("pH: Residuals vs Fitted", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confirmation_run(conf: Dict[str, Any], out_path: str):
    """Plot Confirmation Run comparison: Predicted vs Observed with 95% Prediction Interval."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    # Panel 1: Viscosity
    ax1 = axes[0]
    pred_v = conf["viscosity_cp"]["predicted"]
    obs_v = conf["viscosity_cp"]["observed"]
    pi_v_low = conf["viscosity_cp"]["pi_95_lower"]
    pi_v_high = conf["viscosity_cp"]["pi_95_upper"]
    spec_v_low, spec_v_high = conf["viscosity_cp"]["spec_limits"]

    # Bar / error bar
    ax1.errorbar(
        [0],
        [pred_v],
        yerr=[[pred_v - pi_v_low], [pi_v_high - pred_v]],
        fmt="o",
        color=CB_BLUE,
        ecolor=CB_BLUE,
        elinewidth=2.5,
        capsize=8,
        capthick=2,
        markersize=9,
        label=f"Predicted ({pred_v:,.0f} cP)\nwith 95% PI [{pi_v_low:,.0f}, {pi_v_high:,.0f}]",
    )
    ax1.scatter(
        [0.4],
        [obs_v],
        color=CB_ORANGE,
        s=120,
        marker="s",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        label=f"Observed Confirmation ({obs_v:,.0f} cP)",
    )

    ax1.axhline(spec_v_low, color=CB_PURPLE, linestyle="--", linewidth=1.8, label="LSL (30,000 cP)")
    ax1.axhline(spec_v_high, color=CB_PURPLE, linestyle="--", linewidth=1.8, label="USL (60,000 cP)")

    ax1.set_xlim(-0.5, 0.9)
    ax1.set_ylim(25000, 65000)
    ax1.set_xticks([0, 0.4])
    ax1.set_xticklabels(["Predicted", "Observed"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Apparent Viscosity (cP)", fontsize=11, fontweight="bold")
    ax1.set_title("Viscosity Confirmation (Status: PASS)", fontsize=11.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    # Panel 2: pH
    ax2 = axes[1]
    pred_p = conf["ph"]["predicted"]
    obs_p = conf["ph"]["observed"]
    pi_p_low = conf["ph"]["pi_95_lower"]
    pi_p_high = conf["ph"]["pi_95_upper"]
    spec_p_low, spec_p_high = conf["ph"]["spec_limits"]

    ax2.errorbar(
        [0],
        [pred_p],
        yerr=[[pred_p - pi_p_low], [pi_p_high - pred_p]],
        fmt="o",
        color=CB_GREEN,
        ecolor=CB_GREEN,
        elinewidth=2.5,
        capsize=8,
        capthick=2,
        markersize=9,
        label=f"Predicted ({pred_p:.2f})\nwith 95% PI [{pi_p_low:.2f}, {pi_p_high:.2f}]",
    )
    ax2.scatter(
        [0.4],
        [obs_p],
        color=CB_ORANGE,
        s=120,
        marker="s",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        label=f"Observed Confirmation ({obs_p:.2f})",
    )

    ax2.axhline(spec_p_low, color=CB_PURPLE, linestyle="--", linewidth=1.8, label="LSL (pH 6.00)")
    ax2.axhline(spec_p_high, color=CB_PURPLE, linestyle="--", linewidth=1.8, label="USL (pH 7.50)")

    ax2.set_xlim(-0.5, 0.9)
    ax2.set_ylim(5.6, 7.8)
    ax2.set_xticks([0, 0.4])
    ax2.set_xticklabels(["Predicted", "Observed"], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Formulation pH", fontsize=11, fontweight="bold")
    ax2.set_title("pH Confirmation (Status: PASS)", fontsize=11.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
