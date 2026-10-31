"""Self-check test suite for QbD DoE Design Space pipeline.

Validates:
  1. Clean dataset contains exactly 11 runs
  2. Exactly 3 center-point runs
  3. Exactly 9 unique factor combinations
  4. Response surface model has exactly 6 terms
  5. ANOVA p-values are in valid range [0, 1]
  6. R² and Adjusted R² values are in valid range [0, 1]
  7. Feasible design space area > 0%
  8. Recommended operating point is inside both CQA specifications
  9. All 7 decision-grade charts exist and exceed 10 KB
  10. results.json contains all required keys and valid confirmation status
"""

import json
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")


def run_tests():
    print("Running QbD DoE self-checks...")
    errors = []

    # 1. Check clean runs dataset
    runs_csv = os.path.join(DATA_DIR, "runs.csv")
    if not os.path.exists(runs_csv):
        errors.append(f"Missing {runs_csv}")
        return errors

    df = pd.read_csv(runs_csv)

    # 1.1 Exactly 11 runs
    if len(df) != 11:
        errors.append(f"Expected 11 runs, got {len(df)}")

    # 1.2 Exactly 3 center runs
    center_count = int(np_sum := (df["run_type"] == "center").sum())
    if center_count != 3:
        errors.append(f"Expected 3 center runs, got {center_count}")

    # 1.3 Exactly 9 unique factor combinations
    unique_combos = len(df.drop_duplicates(subset=["A", "B"]))
    if unique_combos != 9:
        errors.append(f"Expected 9 unique factor combinations, got {unique_combos}")

    # 2. Check ANOVA files & model terms
    for resp in ["viscosity", "ph"]:
        anova_path = os.path.join(OUT_DIR, f"anova_{resp}.csv")
        if not os.path.exists(anova_path):
            errors.append(f"Missing {anova_path}")
            continue

        anova_df = pd.read_csv(anova_path)
        # Check ANOVA p-values in [0, 1]
        valid_p_vals = anova_df["p_value"].dropna()
        for p in valid_p_vals:
            if not (0.0 <= p <= 1.0):
                errors.append(f"Invalid ANOVA p-value in {resp}: {p}")

    # 3. Check results.json
    results_path = os.path.join(OUT_DIR, "results.json")
    if not os.path.exists(results_path):
        errors.append(f"Missing {results_path}")
        return errors

    with open(results_path, "r") as f:
        res = json.load(f)

    required_keys = [
        "case_study",
        "cqas",
        "factors",
        "fixed_process_controls",
        "coefficients",
        "r2",
        "adj_r2",
        "anova_stats",
        "lack_of_fit_p",
        "feasible_area_pct",
        "recommended_point",
        "confirmation_run",
        "fmea_ranking",
    ]
    for key in required_keys:
        if key not in res:
            errors.append(f"Missing key in results.json: {key}")

    # 4. Model terms count (6 terms: Intercept, A, B, A*B, A^2, B^2)
    for resp in ["viscosity", "ph"]:
        n_terms = len(res["coefficients"].get(resp, {}))
        if n_terms != 6:
            errors.append(f"Expected 6 model terms for {resp}, got {n_terms}")

        r2_val = res["r2"].get(resp, -1)
        if not (0.0 <= r2_val <= 1.0):
            errors.append(f"Invalid R2 for {resp}: {r2_val}")

        adj_r2_val = res["adj_r2"].get(resp, -1)
        if not (0.0 <= adj_r2_val <= 1.0):
            errors.append(f"Invalid Adj-R2 for {resp}: {adj_r2_val}")

    # 5. Feasible area > 0%
    feas_area = res.get("feasible_area_pct", 0)
    if feas_area <= 0.0:
        errors.append(f"Expected feasible_area_pct > 0, got {feas_area}")

    # 6. Recommended point inside specs
    rec_pt = res.get("recommended_point", {})
    pred_v = rec_pt.get("predicted_viscosity", 0)
    pred_p = rec_pt.get("predicted_ph", 0)

    if not (30000.0 <= pred_v <= 60000.0):
        errors.append(f"Recommended viscosity {pred_v} out of spec [30k, 60k]")

    if not (6.00 <= pred_p <= 7.50):
        errors.append(f"Recommended pH {pred_p} out of spec [6.00, 7.50]")

    # 7. Confirmation run validation
    conf = res.get("confirmation_run", {})
    if conf.get("confirmation_status") != "PASS":
        errors.append(f"Confirmation run status is {conf.get('confirmation_status')}, expected PASS")

    # 8. Check 7 charts exist and exceed 10 KB
    chart_names = [
        "01_main_effects.png",
        "02_interaction.png",
        "03_contour_viscosity.png",
        "04_contour_ph.png",
        "05_design_space.png",
        "06_diagnostics.png",
        "07_confirmation.png",
    ]
    for cname in chart_names:
        cpath = os.path.join(CHARTS_DIR, cname)
        if not os.path.exists(cpath):
            errors.append(f"Missing chart: {cname}")
        else:
            size_kb = os.path.getsize(cpath) / 1024.0
            if size_kb < 10.0:
                errors.append(f"Chart {cname} is too small ({size_kb:.1f} KB < 10 KB)")

    return errors


if __name__ == "__main__":
    errs = run_tests()
    if errs:
        print("FAILED with errors:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED: 100% self-check assertions verified successfully.")
        sys.exit(0)
