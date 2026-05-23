"""
baselines.py
============
Altman Z-Score and Beneish M-Score reimplemented as classifiers
so we can benchmark them on the EXACT same test set as our models.

Why reimplement instead of using our metrics.py versions?
---------------------------------------------------------
metrics.py computes Z/M for display purposes on a single company.
Here we need them to run on the full labeled DataFrame (30,000 rows),
output a probability-like score, and integrate with sklearn's
roc_auc_score / average_precision_score for fair comparison.

Fair comparison rules
---------------------
1. Same test set, same years, same companies
2. Same evaluation metrics (AUC-ROC, AUC-PR)
3. Altman evaluated on BOTH tasks (fraud + distress) even though
   it was designed for bankruptcy — shows the domain mismatch
4. Beneish evaluated on BOTH tasks
5. No threshold tuning for baselines (use raw scores as rankings)
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


# ── Altman Z-Score ────────────────────────────────────────────────────────────
def altman_z_scores(df: pd.DataFrame) -> pd.Series:
    """
    Compute Altman Z for every row in the feature DataFrame.
    Returns raw Z scores (higher = safer, lower = more distressed).

    X1 = Working Capital / Total Assets      (f_z_x1)
    X2 = Retained Earnings / Total Assets    (f_z_x2)
    X3 = EBIT / Total Assets                 (f_z_x3)
    X4 = Market Cap / Total Liabilities      (f_z_x4 — often NaN, use 1.0 fallback)
    X5 = Revenue / Total Assets              (f_z_x5)
    """
    x1 = df.get("f_z_x1", pd.Series(np.nan, index=df.index))
    x2 = df.get("f_z_x2", pd.Series(np.nan, index=df.index))
    x3 = df.get("f_z_x3", pd.Series(np.nan, index=df.index))
    x4 = df.get("f_z_x4", pd.Series(1.0,    index=df.index)).fillna(1.0)
    x5 = df.get("f_z_x5", pd.Series(np.nan, index=df.index))

    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
    return z


def altman_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Convert Z to a risk score [0, 1] where 1 = highest risk.
    Invert Z (lower Z = higher risk) and normalise.
    """
    z = altman_z_scores(df)
    # Sigmoid-like inversion: Z=1.81 → 0.5, Z=2.99 → 0.27, Z=0 → 0.86
    risk = 1 / (1 + np.exp(z - 1.81))
    return risk.clip(0, 1)


# ── Beneish M-Score ───────────────────────────────────────────────────────────
def beneish_m_scores(df: pd.DataFrame) -> pd.Series:
    """
    Compute Beneish M-Score for every row.
    Returns raw M scores (higher = more likely manipulator).
    M > -2.22 → likely manipulator.
    """
    DSRI = df.get("f_b_dsri", pd.Series(1.0, index=df.index)).fillna(1.0)
    GMI  = df.get("f_b_gmi",  pd.Series(1.0, index=df.index)).fillna(1.0)
    AQI  = df.get("f_b_aqi",  pd.Series(1.0, index=df.index)).fillna(1.0)
    SGI  = df.get("f_b_sgi",  pd.Series(1.0, index=df.index)).fillna(1.0)
    DEPI = df.get("f_b_depi", pd.Series(1.0, index=df.index)).fillna(1.0)
    SGAI = df.get("f_b_sgai", pd.Series(1.0, index=df.index)).fillna(1.0)
    LVGI = df.get("f_b_lvgi", pd.Series(1.0, index=df.index)).fillna(1.0)
    TATA = df.get("f_b_tata", pd.Series(0.0, index=df.index)).fillna(0.0)

    M = (-4.84
         + 0.920 * DSRI
         + 0.528 * GMI
         + 0.404 * AQI
         + 0.892 * SGI
         + 0.115 * DEPI
         - 0.172 * SGAI
         + 4.679 * TATA
         - 0.327 * LVGI)
    return M


def beneish_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Convert M to a risk score [0, 1] where 1 = highest manipulation risk.
    M = -2.22 → 0.5 (threshold), higher M = higher risk.
    """
    m = beneish_m_scores(df)
    risk = 1 / (1 + np.exp(-1.5 * (m + 2.22)))   # sigmoid centred at -2.22
    return risk.clip(0, 1)


# ── Benchmark evaluation ──────────────────────────────────────────────────────
def benchmark(
    test_df:    pd.DataFrame,
    our_scores: pd.Series | None = None,
    task:       str = "fraud",          # "fraud" or "distress"
) -> pd.DataFrame:
    """
    Evaluate Altman, Beneish, and (optionally) our model on the same test set.

    Parameters
    ----------
    test_df     : feature DataFrame with label columns
    our_scores  : our ensemble scores (Series aligned with test_df index)
    task        : "fraud" → use is_fraud label; "distress" → use is_bankrupt

    Returns
    -------
    DataFrame with columns [model, auc_roc, auc_pr, recall_at_5pct]
    """
    label_col = "is_fraud" if task == "fraud" else "is_bankrupt"
    assert label_col in test_df.columns, f"'{label_col}' column missing"

    y = test_df[label_col].astype(int)

    if y.sum() == 0:
        logger.warning(f"No positive labels in test set for task='{task}'")
        return pd.DataFrame()

    rows = []

    def _eval(name: str, scores: pd.Series) -> dict:
        scores = scores.fillna(scores.median())
        # Handle constant score edge case
        if scores.nunique() == 1:
            return {"model": name, "auc_roc": 0.5, "auc_pr": float(y.mean()),
                    "recall_at_5pct": np.nan}

        auc_roc = roc_auc_score(y, scores)
        auc_pr  = average_precision_score(y, scores)

        # Recall @ top 5% flagged
        threshold = scores.quantile(0.95)
        flagged   = (scores >= threshold)
        recall_5  = (y[flagged].sum() / y.sum()) if y.sum() > 0 else np.nan

        return {
            "model":          name,
            "auc_roc":        round(auc_roc, 4),
            "auc_pr":         round(auc_pr,  4),
            "recall_at_5pct": round(recall_5, 4),
            "n_test":         len(y),
            "n_positive":     int(y.sum()),
            "positive_rate":  round(float(y.mean()), 4),
        }

    rows.append(_eval("Altman Z-Score",   altman_risk_score(test_df)))
    rows.append(_eval("Beneish M-Score",  beneish_risk_score(test_df)))

    if our_scores is not None:
        label = "Our Model (XGB + IsoForest)"
        rows.append(_eval(label, our_scores))

    result = pd.DataFrame(rows).set_index("model")

    logger.info(f"\n{'='*55}")
    logger.info(f"BENCHMARK RESULTS — Task: {task.upper()}")
    logger.info(f"{'='*55}")
    logger.info(result[["auc_roc", "auc_pr", "recall_at_5pct"]].to_string())

    return result


# ── Lift curve data ───────────────────────────────────────────────────────────
def lift_curve_data(
    test_df:    pd.DataFrame,
    our_scores: pd.Series,
    task:       str = "fraud",
) -> pd.DataFrame:
    """
    Compute lift curve data for all three models.
    X axis: % of companies reviewed (top-K%)
    Y axis: % of actual frauds caught (recall)

    Returns DataFrame suitable for plotting.
    """
    label_col = "is_fraud" if task == "fraud" else "is_bankrupt"
    y = test_df[label_col].astype(int).values
    n = len(y)

    models = {
        "Altman Z-Score":   altman_risk_score(test_df).values,
        "Beneish M-Score":  beneish_risk_score(test_df).values,
        "Our Model":        our_scores.values,
    }

    rows = []
    pct_range = np.arange(0.01, 1.01, 0.01)

    for model_name, scores in models.items():
        order     = np.argsort(-scores)     # descending
        y_sorted  = y[order]
        cum_fraud = np.cumsum(y_sorted)
        total_fraud = y.sum()

        for pct in pct_range:
            k        = max(1, int(pct * n))
            recall   = cum_fraud[k-1] / total_fraud if total_fraud > 0 else 0
            rows.append({
                "model":   model_name,
                "pct_reviewed": round(pct * 100, 1),
                "recall":  round(recall, 4),
            })

    return pd.DataFrame(rows)
