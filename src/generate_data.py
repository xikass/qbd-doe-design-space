"""Generate synthetic QbD formulation data and audit raw records.

Case study: Model Oral Gel (500 g laboratory batch).
CQAs:
  - Apparent Viscosity: 30,000 to 60,000 cP
  - pH: 6.00 to 7.50

Design: 3x3 Full Factorial with 3 center-point replicates (11 runs).
Factors:
  - Factor A: Carbomer 974P % w/w (0.8, 1.2, 1.6)
  - Factor B: Neutralization ratio NaOH:carbomer w/w (0.30, 0.40, 0.50)
Fixed controls:
  - Glycerol: 20.0% w/w
  - Agitation: 800 rpm
  - Temperature: 25.0 °C
"""

import json
import os
import sqlite3
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
DB_PATH = os.path.join(DATA_DIR, "doe.db")


def generate_fmea_data() -> pd.DataFrame:
    """Generate risk assessment matrix (ICH Q9 FMEA)."""
    fmea_data = [
        {
            "factor": "Carbomer 974P % w/w",
            "role": "Primary gelling & bioadhesive matrix polymer",
            "failure_mode": "Viscosity collapse (<30k cP) or excessive rigidity (>60k cP) impairing mucosal flow",
            "severity": 5,
            "occurrence": 4,
            "detection": 4,
            "rpn": 80,
            "action": "Selected for DoE (Factor A: 0.8% - 1.6% w/w)",
        },
        {
            "factor": "NaOH : Carbomer Ratio (w/w)",
            "role": "Polymer neutralizing agent",
            "failure_mode": "Under-neutralization causes acidity and incomplete uncoiling; excess causes counter-ion shielding",
            "severity": 5,
            "occurrence": 4,
            "detection": 4,
            "rpn": 80,
            "action": "Selected for DoE (Factor B: 0.30 - 0.50 w/w)",
        },
        {
            "factor": "Mixing Speed (rpm)",
            "role": "Dispersion shear rate",
            "failure_mode": "Air entrapment at high rpm or incomplete polymer hydration at low rpm",
            "severity": 3,
            "occurrence": 2,
            "detection": 2,
            "rpn": 12,
            "action": "Fixed control at 800 rpm (sufficient vortex without vortex aeration)",
        },
        {
            "factor": "Glycerol % w/w",
            "role": "Humectant & plasticizer",
            "failure_mode": "Phase syneresis or tackiness deviation",
            "severity": 3,
            "occurrence": 2,
            "detection": 2,
            "rpn": 12,
            "action": "Fixed control at 20.0% w/w (validated in pre-formulation screening)",
        },
    ]
    df_fmea = pd.DataFrame(fmea_data)
    return df_fmea


def generate_clean_runs(seed: int = 42) -> pd.DataFrame:
    """Generate 11 runs (9 factorial combinations + 2 replicate center points)."""
    np.random.seed(seed)

    a_levels = [0.8, 1.2, 1.6]
    b_levels = [0.30, 0.40, 0.50]

    runs = []
    run_id = 1
    for a in a_levels:
        for b in b_levels:
            run_type = "center" if (a == 1.2 and b == 0.40) else "factorial"
            runs.append({"run_id": run_id, "A": a, "B": b, "run_type": run_type})
            run_id += 1

    # Add 2 additional center-point replicates for pure error determination
    runs.append({"run_id": run_id, "A": 1.2, "B": 0.40, "run_type": "center"})
    run_id += 1
    runs.append({"run_id": run_id, "A": 1.2, "B": 0.40, "run_type": "center"})

    df = pd.DataFrame(runs)

    # Coded variables for response surface generation:
    # x1 in [-1, 0, 1] for A in [0.8, 1.2, 1.6]
    # x2 in [-1, 0, 1] for B in [0.30, 0.40, 0.50]
    x1 = (df["A"] - 1.2) / 0.40
    x2 = (df["B"] - 0.40) / 0.10

    # Quadratic response surface with realistic experimental noise
    noise_v = np.random.normal(0, 400.0, size=len(df))
    noise_p = np.random.normal(0, 0.035, size=len(df))

    # Viscosity model (cP):
    y_visc = (
        45200.0
        + 15500.0 * x1
        + 4500.0 * x2
        + 2000.0 * x1 * x2
        - 1800.0 * (x1**2)
        - 4200.0 * (x2**2)
        + noise_v
    )

    # pH model:
    y_ph = (
        6.78
        - 0.20 * x1
        + 0.60 * x2
        - 0.04 * x1 * x2
        - 0.06 * (x1**2)
        - 0.22 * (x2**2)
        + noise_p
    )

    df["viscosity"] = np.round(y_visc, 1)
    df["ph"] = np.round(y_ph, 2)
    return df


def generate_raw_dirty_runs(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Create raw acquisition CSV with realistic dirty lab artifacts."""
    raw_df = clean_df.copy()

    # Artifact 1: Accidental duplicate row (e.g. Run 5 entered twice by lab operator)
    dup_row = raw_df[raw_df["run_id"] == 5].copy()
    dup_row["run_id"] = 12

    # Artifact 2: Incomplete record with sensor drop / missing viscosity
    null_row = pd.DataFrame(
        [
            {
                "run_id": 13,
                "A": 1.2,
                "B": 0.40,
                "run_type": "center",
                "viscosity": np.nan,
                "ph": 6.80,
            }
        ]
    )

    raw_dirty = pd.concat([raw_df, dup_row, null_row], ignore_index=True)
    return raw_dirty


def audit_and_clean_data(raw_csv_path: str) -> tuple[pd.DataFrame, dict]:
    """Audit raw CSV records, detect data anomalies, and return clean dataset."""
    df_raw = pd.read_csv(raw_csv_path)
    total_raw_rows = len(df_raw)

    # Audit check 1: Missing / null response values
    null_mask = df_raw["viscosity"].isna() | df_raw["ph"].isna()
    null_rows = df_raw[null_mask].to_dict(orient="records")

    # Audit check 2: Duplicate runs (same A and B beyond planned center replicates)
    # Factorial points must have exactly 1 run; center points must have exactly 3 runs.
    df_no_nulls = df_raw[~null_mask].copy()

    # Track duplicates
    seen_center = 0
    clean_rows = []
    dropped_duplicates = []

    for _, row in df_no_nulls.iterrows():
        a_val = round(row["A"], 2)
        b_val = round(row["B"], 2)
        is_center = (a_val == 1.20) and (b_val == 0.40)

        if is_center:
            if seen_center < 3:
                seen_center += 1
                clean_rows.append(row)
            else:
                dropped_duplicates.append(row.to_dict())
        else:
            # Factorial point: check if already in clean_rows
            already_present = any(
                round(r["A"], 2) == a_val and round(r["B"], 2) == b_val
                for r in clean_rows
            )
            if not already_present:
                clean_rows.append(row)
            else:
                dropped_duplicates.append(row.to_dict())

    df_clean = pd.DataFrame(clean_rows).reset_index(drop=True)
    df_clean["run_id"] = np.arange(1, len(df_clean) + 1)

    audit_report = {
        "raw_record_count": total_raw_rows,
        "clean_record_count": len(df_clean),
        "anomalies_detected": {
            "incomplete_records_null_response": len(null_rows),
            "duplicate_records": len(dropped_duplicates),
        },
        "flagged_null_records": null_rows,
        "flagged_duplicate_records": dropped_duplicates,
        "audit_disposition": "Rejected invalid artifacts; preserved standard 3x3 full factorial + 3 center replicates (11 runs total).",
    }
    return df_clean, audit_report


def build_sqlite_db(df_runs: pd.DataFrame, df_fmea: pd.DataFrame, db_path: str):
    """Write verified datasets to SQLite database."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    df_runs.to_sql("runs", conn, index=False)
    df_fmea.to_sql("fmea", conn, index=False)

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX idx_runs_type ON runs(run_type)")
    cursor.execute("CREATE INDEX idx_runs_factors ON runs(A, B)")
    conn.commit()
    conn.close()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] Generating ICH Q9 FMEA Risk Assessment...")
    df_fmea = generate_fmea_data()
    fmea_csv = os.path.join(DATA_DIR, "fmea.csv")
    df_fmea.to_csv(fmea_csv, index=False)
    print(f"      Saved FMEA to {fmea_csv}")

    print("[2/4] Generating experimental runs with seeded noise...")
    df_clean_gen = generate_clean_runs(seed=42)
    df_raw = generate_raw_dirty_runs(df_clean_gen)
    raw_csv = os.path.join(DATA_DIR, "runs_raw.csv")
    df_raw.to_csv(raw_csv, index=False)
    print(f"      Saved raw lab data (with dirty artifacts) to {raw_csv}")

    print("[3/4] Executing data audit and anomaly rejection...")
    df_clean, audit_rep = audit_and_clean_data(raw_csv)
    clean_csv = os.path.join(DATA_DIR, "runs.csv")
    df_clean.to_csv(clean_csv, index=False)
    audit_json = os.path.join(OUT_DIR, "data_audit_report.json")
    with open(audit_json, "w") as f:
        json.dump(audit_rep, f, indent=2)
    print(f"      Audit complete: {audit_rep['anomalies_detected']}")
    print(f"      Saved clean dataset ({len(df_clean)} runs) to {clean_csv}")
    print(f"      Saved audit report to {audit_json}")

    print("[4/4] Building SQLite database...")
    build_sqlite_db(df_clean, df_fmea, DB_PATH)
    print(f"      Populated SQLite database at {DB_PATH}")


if __name__ == "__main__":
    main()
