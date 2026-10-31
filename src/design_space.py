"""Design space determination, robust optimization, and confirmation run engine.

Calculates:
  - Multi-attribute feasible design space over factor domain
  - Feasible area percentage
  - Robust operating point (maximizing distance to specification boundaries)
  - Confirmation run simulation with 95% prediction intervals (PI)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

SPEC_VISCOSITY_LOWER = 30000.0
SPEC_VISCOSITY_UPPER = 60000.0
SPEC_PH_LOWER = 6.00
SPEC_PH_UPPER = 7.50


def evaluate_grid(
    beta_visc: np.ndarray,
    beta_ph: np.ndarray,
    a_range: Tuple[float, float] = (0.8, 1.6),
    b_range: Tuple[float, float] = (0.30, 0.50),
    resolution: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate response surfaces and CQA feasibility over factor grid."""
    a_vec = np.linspace(a_range[0], a_range[1], resolution)
    b_vec = np.linspace(b_range[0], b_range[1], resolution)
    aa, bb = np.meshgrid(a_vec, b_vec)

    # Design matrix on grid: [1, A, B, A*B, A^2, B^2]
    ones = np.ones(aa.size)
    grid_x = np.column_stack(
        [
            ones,
            aa.ravel(),
            bb.ravel(),
            (aa * bb).ravel(),
            (aa**2).ravel(),
            (bb**2).ravel(),
        ]
    )

    pred_visc = grid_x @ beta_visc
    pred_ph = grid_x @ beta_ph

    # Check feasibility against CQAs
    feas_visc = (pred_visc >= SPEC_VISCOSITY_LOWER) & (
        pred_visc <= SPEC_VISCOSITY_UPPER
    )
    feas_ph = (pred_ph >= SPEC_PH_LOWER) & (pred_ph <= SPEC_PH_UPPER)
    feas_joint = feas_visc & feas_ph

    feasible_pct = float(100.0 * np.sum(feas_joint) / aa.size)

    return aa, bb, pred_visc.reshape(aa.shape), pred_ph.reshape(aa.shape), feas_joint.reshape(aa.shape), feasible_pct


def find_robust_operating_point(
    aa: np.ndarray,
    bb: np.ndarray,
    visc_grid: np.ndarray,
    ph_grid: np.ndarray,
    feas_grid: np.ndarray,
) -> Dict[str, Any]:
    """Find operating point maximizing the minimum normalized distance to CQA boundaries."""
    visc_flat = visc_grid.ravel()
    ph_flat = ph_grid.ravel()
    feas_flat = feas_grid.ravel()

    # Normalized margins for Viscosity (range = 30,000 cP)
    v_range = SPEC_VISCOSITY_UPPER - SPEC_VISCOSITY_LOWER
    m_v_lower = (visc_flat - SPEC_VISCOSITY_LOWER) / v_range
    m_v_upper = (SPEC_VISCOSITY_UPPER - visc_flat) / v_range
    margin_v = np.minimum(m_v_lower, m_v_upper)

    # Normalized margins for pH (range = 1.50)
    p_range = SPEC_PH_UPPER - SPEC_PH_LOWER
    m_p_lower = (ph_flat - SPEC_PH_LOWER) / p_range
    m_p_upper = (SPEC_PH_UPPER - ph_flat) / p_range
    margin_p = np.minimum(m_p_lower, m_p_upper)

    # Joint minimum margin across both responses
    margin_joint = np.where(feas_flat, np.minimum(margin_v, margin_p), -1.0)
    best_idx = int(np.argmax(margin_joint))

    best_a = float(aa.ravel()[best_idx])
    best_b = float(bb.ravel()[best_idx])
    pred_v = float(visc_flat[best_idx])
    pred_p = float(ph_flat[best_idx])
    best_margin = float(margin_joint[best_idx])

    return {
        "recommended_A": round(best_a, 3),
        "recommended_B": round(best_b, 3),
        "predicted_viscosity": round(pred_v, 1),
        "predicted_ph": round(pred_p, 2),
        "min_normalized_margin": round(best_margin, 4),
        "viscosity_margin_normalized": round(float(margin_v[best_idx]), 4),
        "ph_margin_normalized": round(float(margin_p[best_idx]), 4),
    }


def compute_prediction_interval(
    x_point: np.ndarray,
    beta: np.ndarray,
    xtx_inv: np.ndarray,
    ms_res: float,
    df_res: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Compute predicted value and 95% prediction interval (PI) for a new single observation."""
    pred = float(x_point @ beta)
    # Variance of a new prediction: s^2 * (1 + x_0 * (X'X)^-1 * x_0')
    leverage = float(x_point @ xtx_inv @ x_point.T)
    var_pred = ms_res * (1.0 + leverage)
    se_pred = float(np.sqrt(var_pred))

    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_res))

    pi_lower = pred - t_crit * se_pred
    pi_upper = pred + t_crit * se_pred
    return pred, se_pred, pi_lower, pi_upper


def execute_confirmation_run(
    robust_pt: Dict[str, Any],
    fit_visc: Dict[str, Any],
    fit_ph: Dict[str, Any],
    seed: int = 101,
) -> Dict[str, Any]:
    """Execute synthetic confirmation run at the recommended point with prediction intervals."""
    a_rec = robust_pt["recommended_A"]
    b_rec = robust_pt["recommended_B"]
    x0 = np.array([1.0, a_rec, b_rec, a_rec * b_rec, a_rec**2, b_rec**2])

    beta_v = np.array(fit_visc["coefficients"])
    beta_p = np.array(fit_ph["coefficients"])

    pred_v, se_v, pi_v_low, pi_v_high = compute_prediction_interval(
        x0, beta_v, fit_visc["xtx_inv"], fit_visc["ms_residual"], fit_visc["df_residual"]
    )
    pred_p, se_p, pi_p_low, pi_p_high = compute_prediction_interval(
        x0, beta_p, fit_ph["xtx_inv"], fit_ph["ms_residual"], fit_ph["df_residual"]
    )

    # Seeded observation with true experimental noise
    np.random.seed(seed)
    noise_v = float(np.random.normal(0, np.sqrt(fit_visc["ms_residual"])))
    noise_p = float(np.random.normal(0, np.sqrt(fit_ph["ms_residual"])))

    obs_v = round(pred_v + noise_v, 1)
    obs_p = round(pred_p + noise_p, 2)

    # Check whether observation is within PI
    pi_v_pass = bool(pi_v_low <= obs_v <= pi_v_high)
    pi_p_pass = bool(pi_p_low <= obs_p <= pi_p_high)

    # Check whether observation is within specification
    spec_v_pass = bool(SPEC_VISCOSITY_LOWER <= obs_v <= SPEC_VISCOSITY_UPPER)
    spec_p_pass = bool(SPEC_PH_LOWER <= obs_p <= SPEC_PH_UPPER)

    overall_pass = bool(pi_v_pass and pi_p_pass and spec_v_pass and spec_p_pass)

    return {
        "confirmation_factors": {
            "Carbomer_pct": a_rec,
            "NaOH_ratio": b_rec,
        },
        "viscosity_cp": {
            "predicted": round(pred_v, 1),
            "observed": obs_v,
            "se_prediction": round(se_v, 1),
            "pi_95_lower": round(pi_v_low, 1),
            "pi_95_upper": round(pi_v_high, 1),
            "spec_limits": [SPEC_VISCOSITY_LOWER, SPEC_VISCOSITY_UPPER],
            "in_prediction_interval": pi_v_pass,
            "in_specification": spec_v_pass,
        },
        "ph": {
            "predicted": round(pred_p, 2),
            "observed": obs_p,
            "se_prediction": round(se_p, 3),
            "pi_95_lower": round(pi_p_low, 2),
            "pi_95_upper": round(pi_p_high, 2),
            "spec_limits": [SPEC_PH_LOWER, SPEC_PH_UPPER],
            "in_prediction_interval": pi_p_pass,
            "in_specification": spec_p_pass,
        },
        "confirmation_status": "PASS" if overall_pass else "FAIL",
    }
