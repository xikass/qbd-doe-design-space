"""Master analysis orchestration script for QbD DoE Design Space.

Integrates:
  - SQLite data extraction
  - Second-order OLS modeling & ANOVA decomposition (with lack-of-fit)
  - Feasible design space mapping & robust setpoint optimization
  - Seeded confirmation run & 95% prediction interval validation
  - 7 Decision-grade scientific figures
  - Structured output serialization (CSV & JSON)
"""

import json
import os
import sqlite3
import numpy as np
import pandas as pd

from src.anova import fit_ols, compute_anova
from src.design_space import (
    evaluate_grid,
    find_robust_operating_point,
    execute_confirmation_run,
)
from src.charts import (
    plot_main_effects,
    plot_interaction,
    plot_contour_viscosity,
    plot_contour_ph,
    plot_design_space_overlay,
    plot_diagnostics,
    plot_confirmation_run,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
SQL_DIR = os.path.join(BASE_DIR, "sql")
DB_PATH = os.path.join(DATA_DIR, "doe.db")


def load_data_from_sqlite() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract experimental runs and FMEA records from SQLite database."""
    conn = sqlite3.connect(DB_PATH)

    # Load runs using design_matrix.sql query
    with open(os.path.join(SQL_DIR, "design_matrix.sql"), "r") as f:
        sql_query = f.read()

    df_sql = pd.read_sql_query(sql_query, conn)
    # Ensure column names match expected factor labels A and B
    df_runs = df_sql.rename(columns={"carbomer_pct": "A", "naoh_ratio": "B"})

    # Load FMEA
    df_fmea = pd.read_sql_query("SELECT * FROM fmea ORDER BY rpn DESC;", conn)

    conn.close()
    return df_runs, df_fmea


def run_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("=" * 65)
    print(" QbD DoE Design Space: Execution Pipeline")
    print("=" * 65)

    # 1. Load data
    print("\n[Step 1/6] Loading data from SQLite database...")
    df_runs, df_fmea = load_data_from_sqlite()
    print(f"  Loaded {len(df_runs)} runs (3 center replicates, 8 factorial).")
    print(f"  FMEA risk factors loaded ({len(df_fmea)} entries).")

    # 2. Fit OLS models and ANOVA
    print("\n[Step 2/6] Fitting response surface models & computing ANOVA...")
    fit_visc = fit_ols(df_runs, "viscosity")
    anova_visc = compute_anova(df_runs, fit_visc)
    anova_visc_path = os.path.join(OUT_DIR, "anova_viscosity.csv")
    anova_visc.to_csv(anova_visc_path, index=False)
    print(f"  Viscosity R2 = {fit_visc['r2']:.4f}, Adj-R2 = {fit_visc['adj_r2']:.4f}")
    lof_p_v = anova_visc.loc[anova_visc["Source"] == "  Lack of Fit", "p_value"].values[0]
    print(f"  Viscosity Lack-of-Fit p-value = {lof_p_v:.4f} (Model adequate: p > 0.05)")

    fit_ph = fit_ols(df_runs, "ph")
    anova_ph = compute_anova(df_runs, fit_ph)
    anova_ph_path = os.path.join(OUT_DIR, "anova_ph.csv")
    anova_ph.to_csv(anova_ph_path, index=False)
    print(f"  pH R2 = {fit_ph['r2']:.4f}, Adj-R2 = {fit_ph['adj_r2']:.4f}")
    lof_p_p = anova_ph.loc[anova_ph["Source"] == "  Lack of Fit", "p_value"].values[0]
    print(f"  pH Lack-of-Fit p-value = {lof_p_p:.4f} (Model adequate: p > 0.05)")

    # 3. Design Space Mapping
    print("\n[Step 3/6] Mapping Design Space & calculating robust setpoint...")
    aa, bb, visc_grid, ph_grid, feas_grid, feasible_pct = evaluate_grid(
        np.array(fit_visc["coefficients"]),
        np.array(fit_ph["coefficients"]),
        resolution=101,
    )
    print(f"  Feasible Design Space Area = {feasible_pct:.2f}% of experimental domain.")

    robust_pt = find_robust_operating_point(aa, bb, visc_grid, ph_grid, feas_grid)
    print(f"  Recommended Operating Point: A = {robust_pt['recommended_A']:.2f}% w/w, B = {robust_pt['recommended_B']:.2f} w/w")
    print(f"  Predicted Viscosity = {robust_pt['predicted_viscosity']:,.1f} cP (spec [30k, 60k])")
    print(f"  Predicted pH = {robust_pt['predicted_ph']:.2f} (spec [6.0, 7.5])")
    print(f"  Min Normalized Margin = {robust_pt['min_normalized_margin']*100:.2f}%")

    # Export grid sample to CSV
    grid_df = pd.DataFrame(
        {
            "Carbomer_pct": aa.ravel(),
            "NaOH_ratio": bb.ravel(),
            "predicted_viscosity": np.round(visc_grid.ravel(), 1),
            "predicted_ph": np.round(ph_grid.ravel(), 2),
            "feasible": feas_grid.ravel(),
        }
    )
    grid_csv_path = os.path.join(OUT_DIR, "design_space_grid.csv")
    grid_df.to_csv(grid_csv_path, index=False)
    print(f"  Saved full design space grid ({len(grid_df)} points) to {grid_csv_path}")

    # 4. Confirmation Run
    print("\n[Step 4/6] Conducting synthetic confirmation run & 95% PI validation...")
    conf = execute_confirmation_run(robust_pt, fit_visc, fit_ph, seed=101)
    print(f"  Confirmation Status: {conf['confirmation_status']}")
    print(
        f"  Viscosity: Pred = {conf['viscosity_cp']['predicted']:,.0f} cP, "
        f"Obs = {conf['viscosity_cp']['observed']:,.0f} cP, "
        f"95% PI = [{conf['viscosity_cp']['pi_95_lower']:,.0f}, {conf['viscosity_cp']['pi_95_upper']:,.0f}] cP "
        f"(In PI: {conf['viscosity_cp']['in_prediction_interval']}, In Spec: {conf['viscosity_cp']['in_specification']})"
    )
    print(
        f"  pH: Pred = {conf['ph']['predicted']:.2f}, "
        f"Obs = {conf['ph']['observed']:.2f}, "
        f"95% PI = [{conf['ph']['pi_95_lower']:.2f}, {conf['ph']['pi_95_upper']:.2f}] "
        f"(In PI: {conf['ph']['in_prediction_interval']}, In Spec: {conf['ph']['in_specification']})"
    )

    # 5. Generate Figures
    print("\n[Step 5/6] Generating decision-grade figures (DPI=150)...")
    c1 = os.path.join(CHARTS_DIR, "01_main_effects.png")
    plot_main_effects(df_runs, c1)
    print(f"  Created: {c1}")

    c2 = os.path.join(CHARTS_DIR, "02_interaction.png")
    plot_interaction(df_runs, c2)
    print(f"  Created: {c2}")

    c3 = os.path.join(CHARTS_DIR, "03_contour_viscosity.png")
    plot_contour_viscosity(aa, bb, visc_grid, df_runs, c3)
    print(f"  Created: {c3}")

    c4 = os.path.join(CHARTS_DIR, "04_contour_ph.png")
    plot_contour_ph(aa, bb, ph_grid, df_runs, c4)
    print(f"  Created: {c4}")

    c5 = os.path.join(CHARTS_DIR, "05_design_space.png")
    plot_design_space_overlay(
        aa, bb, visc_grid, ph_grid, feas_grid, df_runs, robust_pt, feasible_pct, c5
    )
    print(f"  Created: {c5}")

    c6 = os.path.join(CHARTS_DIR, "06_diagnostics.png")
    plot_diagnostics(fit_visc, fit_ph, c6)
    print(f"  Created: {c6}")

    c7 = os.path.join(CHARTS_DIR, "07_confirmation.png")
    plot_confirmation_run(conf, c7)
    print(f"  Created: {c7}")

    # 6. Structured JSON Export
    print("\n[Step 6/6] Compiling and saving results.json...")
    fmea_ranking = df_fmea.to_dict(orient="records")

    results_data = {
        "case_study": "Model Oral Gel (500 g Batch) QbD Design Space",
        "cqas": {
            "apparent_viscosity_cp": {"lower_limit": 30000.0, "upper_limit": 60000.0, "target": 45000.0},
            "ph": {"lower_limit": 6.00, "upper_limit": 7.50, "target": 6.80},
        },
        "factors": {
            "A_carbomer_974p_pct": {"min": 0.8, "center": 1.2, "max": 1.6, "unit": "% w/w"},
            "B_naoh_ratio": {"min": 0.30, "center": 0.40, "max": 0.50, "unit": "w/w"},
        },
        "fixed_process_controls": {
            "glycerol_pct": 20.0,
            "mixing_speed_rpm": 800,
            "temperature_celsius": 25.0,
        },
        "coefficients": {
            "viscosity": {
                term: float(c)
                for term, c in zip(
                    ["Intercept", "A", "B", "A*B", "A^2", "B^2"],
                    fit_visc["coefficients"],
                )
            },
            "ph": {
                term: float(c)
                for term, c in zip(
                    ["Intercept", "A", "B", "A*B", "A^2", "B^2"],
                    fit_ph["coefficients"],
                )
            },
        },
        "r2": {
            "viscosity": round(fit_visc["r2"], 4),
            "ph": round(fit_ph["r2"], 4),
        },
        "adj_r2": {
            "viscosity": round(fit_visc["adj_r2"], 4),
            "ph": round(fit_ph["adj_r2"], 4),
        },
        "anova_stats": {
            "viscosity": anova_visc.to_dict(orient="records"),
            "ph": anova_ph.to_dict(orient="records"),
        },
        "lack_of_fit_p": {
            "viscosity": round(float(lof_p_v), 4),
            "ph": round(float(lof_p_p), 4),
        },
        "feasible_area_pct": round(feasible_pct, 2),
        "recommended_point": robust_pt,
        "confirmation_run": conf,
        "fmea_ranking": fmea_ranking,
    }

    results_path = os.path.join(OUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved to {results_path}")
    print("\nAnalysis successfully completed.")


if __name__ == "__main__":
    run_analysis()
