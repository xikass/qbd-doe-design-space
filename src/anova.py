"""Ordinary Least Squares (OLS) regression and ANOVA engine for response surfaces.

Implements second-order polynomial modeling without statsmodels:
  y = beta_0 + beta_1*A + beta_2*B + beta_12*(A*B) + beta_11*(A^2) + beta_22*(B^2) + e

Decomposes ANOVA into:
  - Model (df=5)
    - Linear (A, B) (df=2)
    - Interaction (AB) (df=1)
    - Quadratic (A^2, B^2) (df=2)
  - Residual Error (df=5)
    - Lack of Fit (df=3)
    - Pure Error (df=2)
  - Total (df=10)
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def build_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Construct second-order polynomial model matrix X.

    Columns: [1, A, B, A*B, A^2, B^2]
    """
    a = df["A"].values
    b = df["B"].values
    ones = np.ones(len(df))
    x = np.column_stack([ones, a, b, a * b, a**2, b**2])
    return x


def fit_ols(df: pd.DataFrame, response_col: str) -> Dict[str, Any]:
    """Fit second-order polynomial OLS model and compute coefficients, SE, t, and p-values."""
    x = build_design_matrix(df)
    y = df[response_col].values
    n = len(y)
    p = x.shape[1] - 1  # 5 model parameters excluding intercept
    df_res = n - p - 1  # 11 - 5 - 1 = 5

    # OLS parameter estimates
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x.T @ y)

    y_pred = x @ beta
    residuals = y - y_pred

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residuals**2))
    ss_mod = ss_tot - ss_res

    ms_res = ss_res / df_res
    ms_mod = ss_mod / p

    r2 = ss_mod / ss_tot
    adj_r2 = 1.0 - (ms_res / (ss_tot / (n - 1)))
    f_mod = ms_mod / ms_res
    p_mod = float(stats.f.sf(f_mod, p, df_res))

    # Variance-covariance of beta
    cov_beta = ms_res * xtx_inv
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
    t_stats = beta / se_beta
    p_values = float(2.0) * stats.t.sf(np.abs(t_stats), df=df_res)

    t_crit = stats.t.ppf(0.975, df=df_res)
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta

    terms = ["Intercept", "A", "B", "A*B", "A^2", "B^2"]
    coef_df = pd.DataFrame(
        {
            "term": terms,
            "coefficient": beta,
            "std_error": se_beta,
            "t_statistic": t_stats,
            "p_value": p_values,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
        }
    )

    # Standardized residuals
    std_residuals = residuals / np.sqrt(ms_res)

    # Normal probability plot coordinates (Q-Q)
    sorted_idx = np.argsort(std_residuals)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))

    return {
        "response": response_col,
        "n": n,
        "df_model": p,
        "df_residual": df_res,
        "df_total": n - 1,
        "coefficients": beta.tolist(),
        "coef_df": coef_df,
        "ss_total": ss_tot,
        "ss_model": ss_mod,
        "ss_residual": ss_res,
        "ms_model": ms_mod,
        "ms_residual": ms_res,
        "f_model": f_mod,
        "p_model": p_mod,
        "r2": r2,
        "adj_r2": adj_r2,
        "residuals": residuals.tolist(),
        "std_residuals": std_residuals.tolist(),
        "fitted_values": y_pred.tolist(),
        "qq_theoretical": theoretical_quantiles.tolist(),
        "qq_ordered_residuals": std_residuals[sorted_idx].tolist(),
        "xtx_inv": xtx_inv,
    }


def compute_anova(df: pd.DataFrame, fit_results: Dict[str, Any]) -> pd.DataFrame:
    """Construct full sequential/orthogonal ANOVA table with Lack-of-Fit and Pure Error."""
    y = df[fit_results["response"]].values
    n = len(y)
    y_mean = np.mean(y)
    ss_tot = fit_results["ss_total"]
    ss_mod = fit_results["ss_model"]
    ss_res = fit_results["ss_residual"]
    ms_res = fit_results["ms_residual"]

    # 1. Linear terms: A, B (df=2)
    x_lin = np.column_stack([np.ones(n), df["A"].values, df["B"].values])
    b_lin = np.linalg.pinv(x_lin.T @ x_lin) @ (x_lin.T @ y)
    y_lin = x_lin @ b_lin
    ss_lin = float(np.sum((y_lin - y_mean) ** 2))

    # 2. Interaction term: A*B (df=1)
    x_int = np.column_stack([x_lin, (df["A"] * df["B"]).values])
    b_int = np.linalg.pinv(x_int.T @ x_int) @ (x_int.T @ y)
    y_int = x_int @ b_int
    ss_int = float(np.sum((y_int - y_mean) ** 2)) - ss_lin

    # 3. Quadratic terms: A^2, B^2 (df=2)
    ss_quad = float(ss_mod - (ss_lin + ss_int))

    # 4. Pure error from replicated center points (df=2)
    center_mask = df["run_type"] == "center"
    center_y = df.loc[center_mask, fit_results["response"]].values
    n_center = len(center_y)
    df_pe = n_center - 1  # 3 - 1 = 2
    ss_pe = float(np.sum((center_y - np.mean(center_y)) ** 2))
    ms_pe = ss_pe / df_pe if df_pe > 0 else 0.0

    # 5. Lack of fit (df=3)
    df_lof = fit_results["df_residual"] - df_pe  # 5 - 2 = 3
    ss_lof = float(ss_res - ss_pe)
    ms_lof = ss_lof / df_lof if df_lof > 0 else 0.0

    f_lof = ms_lof / ms_pe if ms_pe > 0 else np.nan
    p_lof = float(stats.f.sf(f_lof, df_lof, df_pe)) if ms_pe > 0 else np.nan

    # Mean squares for model components
    ms_lin = ss_lin / 2.0
    ms_int = ss_int / 1.0
    ms_quad = ss_quad / 2.0

    f_lin = ms_lin / ms_res
    p_lin = float(stats.f.sf(f_lin, 2, 5))
    f_int = ms_int / ms_res
    p_int = float(stats.f.sf(f_int, 1, 5))
    f_quad = ms_quad / ms_res
    p_quad = float(stats.f.sf(f_quad, 2, 5))

    anova_rows = [
        {"Source": "Model", "DF": 5, "SS": ss_mod, "MS": fit_results["ms_model"], "F_value": fit_results["f_model"], "p_value": fit_results["p_model"]},
        {"Source": "  Linear (A, B)", "DF": 2, "SS": ss_lin, "MS": ms_lin, "F_value": f_lin, "p_value": p_lin},
        {"Source": "  Interaction (A*B)", "DF": 1, "SS": ss_int, "MS": ms_int, "F_value": f_int, "p_value": p_int},
        {"Source": "  Quadratic (A^2, B^2)", "DF": 2, "SS": ss_quad, "MS": ms_quad, "F_value": f_quad, "p_value": p_quad},
        {"Source": "Residual Error", "DF": 5, "SS": ss_res, "MS": ms_res, "F_value": np.nan, "p_value": np.nan},
        {"Source": "  Lack of Fit", "DF": 3, "SS": ss_lof, "MS": ms_lof, "F_value": f_lof, "p_value": p_lof},
        {"Source": "  Pure Error", "DF": 2, "SS": ss_pe, "MS": ms_pe, "F_value": np.nan, "p_value": np.nan},
        {"Source": "Total", "DF": 10, "SS": ss_tot, "MS": np.nan, "F_value": np.nan, "p_value": np.nan},
    ]

    df_anova = pd.DataFrame(anova_rows)
    return df_anova
