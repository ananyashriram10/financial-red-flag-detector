"""
evaluate.py
===========
Runs the full model evaluation pipeline and produces the benchmark report.

Called from the Kaggle notebook after training both models.

Outputs
-------
  1. Walk-forward CV results per model (fold-level + aggregate)
  2. Head-to-head benchmark table vs Altman + Beneish
  3. Lift curve data (for plotting in the notebook)
  4. Feature importance table (SHAP mean absolute values)
  5. All results saved to data/processed/eval_results.json
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from models.baselines import benchmark, lift_curve_data
from models.fraud_model import FraudModel
from models.distress_model import DistressModel

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def run_evaluation(
    labeled_df:     pd.DataFrame,
    fraud_model:    FraudModel,
    distress_model: DistressModel,
    test_year_cutoff: int = 2016,
) -> dict:
    """
    Full evaluation suite. Returns a results dict.

    Parameters
    ----------
    labeled_df        : output of labeler.build_labeled_dataset()
    fraud_model       : fitted FraudModel instance
    distress_model    : fitted DistressModel instance
    test_year_cutoff  : years >= this are used as the hold-out test set
    """
    # ── Split into train / test ───────────────────────────────────────────────
    test_df  = labeled_df[labeled_df["year"] >= test_year_cutoff].copy()
    train_df = labeled_df[labeled_df["year"] <  test_year_cutoff].copy()

    logger.info(f"Test set: {len(test_df):,} rows "
                f"({test_df['year'].min()}–{test_df['year'].max()}), "
                f"fraud={test_df['is_fraud'].sum()}, "
                f"bankrupt={test_df['is_bankrupt'].sum()}")

    results = {}

    # ── 1. Walk-forward CV results ────────────────────────────────────────────
    logger.info("\n=== Walk-Forward CV: Fraud Model ===")
    fraud_cv = fraud_model.walk_forward_evaluate(labeled_df)
    results["fraud_cv"] = {
        "oof_auc_roc": fraud_cv.get("oof_auc_roc"),
        "oof_auc_pr":  fraud_cv.get("oof_auc_pr"),
        "folds":       fraud_cv.get("fold_metrics", pd.DataFrame()).to_dict("records"),
    }

    logger.info("\n=== Walk-Forward CV: Distress Model ===")
    distress_cv = distress_model.walk_forward_evaluate(labeled_df)
    results["distress_cv"] = {
        "oof_auc_roc": distress_cv.get("oof_auc_roc"),
        "oof_auc_pr":  distress_cv.get("oof_auc_pr"),
        "folds":       distress_cv.get("fold_metrics", pd.DataFrame()).to_dict("records"),
    }

    # ── 2. Get model predictions on test set ──────────────────────────────────
    fraud_preds   = fraud_model.predict(test_df)
    distress_preds= distress_model.predict(test_df)

    fraud_scores   = fraud_preds["score"]
    distress_scores= distress_preds["score"]

    # ── 3. Benchmark vs Altman + Beneish ──────────────────────────────────────
    logger.info("\n=== Benchmark: Fraud Task ===")
    fraud_bench = benchmark(test_df, fraud_scores, task="fraud")

    logger.info("\n=== Benchmark: Distress Task ===")
    distress_bench = benchmark(test_df, distress_scores, task="distress")

    results["fraud_benchmark"]   = fraud_bench.reset_index().to_dict("records")
    results["distress_benchmark"]= distress_bench.reset_index().to_dict("records")

    # ── 4. Lift curves ─────────────────────────────────────────────────────────
    fraud_lift   = lift_curve_data(test_df, fraud_scores,    "fraud")
    distress_lift= lift_curve_data(test_df, distress_scores, "distress")

    results["fraud_lift"]   = fraud_lift.to_dict("records")
    results["distress_lift"]= distress_lift.to_dict("records")

    # ── 5. Feature importance (mean |SHAP|) ───────────────────────────────────
    results["fraud_feature_importance"]   = _shap_importance(fraud_model,   test_df)
    results["distress_feature_importance"]= _shap_importance(distress_model, test_df)

    # ── 6. Print summary ──────────────────────────────────────────────────────
    _print_summary(results)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out = PROC_DIR / "eval_results.json"
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    logger.info(f"\nResults saved → {out}")

    return results


def _shap_importance(model, df: pd.DataFrame, n: int = 20) -> list[dict]:
    """Top N features by mean absolute SHAP value."""
    X, feat_cols = model._prep(df)
    X = X[model._feature_cols].fillna(0)
    shap_vals = model.explainer.shap_values(X)
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    idxs      = np.argsort(-mean_abs)[:n]

    from models.base import FEATURE_LABELS
    return [
        {
            "rank":      int(i + 1),
            "feature":   model._feature_cols[j],
            "label":     FEATURE_LABELS.get(model._feature_cols[j],
                                            model._feature_cols[j]),
            "mean_shap": round(float(mean_abs[j]), 5),
        }
        for i, j in enumerate(idxs)
    ]


def _print_summary(results: dict):
    print("\n" + "═"*60)
    print("  MODEL BENCHMARK SUMMARY")
    print("═"*60)

    for task in ["fraud", "distress"]:
        bench = results.get(f"{task}_benchmark", [])
        if not bench:
            continue
        print(f"\n  {task.upper()} DETECTION")
        print(f"  {'Model':<35} {'AUC-ROC':>8} {'AUC-PR':>8} {'Recall@5%':>10}")
        print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*10}")
        for row in bench:
            marker = " ◀" if "Our Model" in str(row.get("model", "")) else ""
            print(f"  {str(row.get('model','')):<35} "
                  f"{row.get('auc_roc', 0):>8.4f} "
                  f"{row.get('auc_pr',  0):>8.4f} "
                  f"{row.get('recall_at_5pct', 0):>10.4f}{marker}")

    cv_f = results.get("fraud_cv", {})
    cv_d = results.get("distress_cv", {})
    print(f"\n  WALK-FORWARD CV (out-of-fold)")
    print(f"  Fraud model    — AUC-ROC: {cv_f.get('oof_auc_roc','N/A')}  "
          f"AUC-PR: {cv_f.get('oof_auc_pr','N/A')}")
    print(f"  Distress model — AUC-ROC: {cv_d.get('oof_auc_roc','N/A')}  "
          f"AUC-PR: {cv_d.get('oof_auc_pr','N/A')}")
    print("═"*60)


def _json_safe(obj):
    """Make numpy types JSON serialisable."""
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    return str(obj)
