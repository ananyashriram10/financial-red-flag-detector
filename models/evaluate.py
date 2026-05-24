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
    # ── Split into train / test (company-level exclusion) ────────────────────
    # Time split first
    train_df = labeled_df[labeled_df["year"] <  test_year_cutoff].copy()
    test_df  = labeled_df[labeled_df["year"] >= test_year_cutoff].copy()

    # Remove from test any company whose fraud/bankrupt label appeared in training.
    # Without this, the model trivially recognises GE as a fraud company in test
    # because it saw GE's earlier years (also labelled fraud) during training —
    # giving inflated AUC (0.99) that means nothing for unseen companies.
    train_labeled_ciks = set(
        train_df.loc[train_df["is_fraud"]   == 1, "cik"].tolist() +
        train_df.loc[train_df["is_bankrupt"]== 1, "cik"].tolist()
    )
    # Keep test rows where either:
    #   a) the company was never labelled in training (genuine unseen company), OR
    #   b) the row itself is clean (we still need clean samples in test)
    test_df = test_df[
        ~test_df["cik"].isin(train_labeled_ciks) |   # unseen fraud/bk company
        (test_df["is_fraud"] == 0) & (test_df["is_bankrupt"] == 0)  # clean rows OK
    ].copy()

    n_excluded = labeled_df["cik"].isin(train_labeled_ciks).sum()
    logger.info(
        f"Train set: {len(train_df):,} rows (pre-{test_year_cutoff}), "
        f"fraud={train_df['is_fraud'].sum()}, bankrupt={train_df['is_bankrupt'].sum()}"
    )
    logger.info(
        f"Test set : {len(test_df):,} rows ({test_year_cutoff}+), "
        f"fraud={test_df['is_fraud'].sum()}, bankrupt={test_df['is_bankrupt'].sum()} "
        f"[{n_excluded:,} rows excluded — companies seen in training]"
    )

    results = {}

    # ── 1. Walk-forward CV results ────────────────────────────────────────────
    logger.info("\n=== Walk-Forward CV: Fraud Model ===")
    fraud_cv = fraud_model.walk_forward_evaluate(labeled_df)
    results["fraud_cv"] = {
        "oof_auc_roc":         fraud_cv.get("oof_auc_roc"),
        "oof_auc_pr":          fraud_cv.get("oof_auc_pr"),
        "oof_recall_at_5pct":  fraud_cv.get("oof_recall_at_5pct"),
        "oof_precision_at_20": fraud_cv.get("oof_precision_at_20"),
        "folds":               fraud_cv.get("fold_metrics", pd.DataFrame()).to_dict("records"),
    }

    logger.info("\n=== Walk-Forward CV: Distress Model ===")
    distress_cv = distress_model.walk_forward_evaluate(labeled_df)
    results["distress_cv"] = {
        "oof_auc_roc":         distress_cv.get("oof_auc_roc"),
        "oof_auc_pr":          distress_cv.get("oof_auc_pr"),
        "oof_recall_at_5pct":  distress_cv.get("oof_recall_at_5pct"),
        "oof_precision_at_20": distress_cv.get("oof_precision_at_20"),
        "folds":               distress_cv.get("fold_metrics", pd.DataFrame()).to_dict("records"),
    }

    # ── 2. Get model predictions on test set ──────────────────────────────────
    fraud_preds   = fraud_model.predict(test_df)
    distress_preds= distress_model.predict(test_df)

    fraud_scores   = fraud_preds["score"]
    distress_scores= distress_preds["score"]

    # ── 3. Benchmark vs Altman + Beneish ──────────────────────────────────────
    # OOF recall/precision passed in so our model's row in the table can show
    # the stable cross-validated numbers alongside the test-set AUC metrics.
    logger.info("\n=== Benchmark: Fraud Task ===")
    fraud_bench = benchmark(
        test_df, fraud_scores, task="fraud",
        oof_recall_5pct  = results["fraud_cv"].get("oof_recall_at_5pct"),
        oof_precision_20 = results["fraud_cv"].get("oof_precision_at_20"),
    )

    logger.info("\n=== Benchmark: Distress Task ===")
    distress_bench = benchmark(
        test_df, distress_scores, task="distress",
        oof_recall_5pct  = results["distress_cv"].get("oof_recall_at_5pct"),
        oof_precision_20 = results["distress_cv"].get("oof_precision_at_20"),
    )

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
    W = 72
    print("\n" + "═"*W)
    print("  MODEL BENCHMARK SUMMARY")
    print("═"*W)

    # ── Section 1: Benchmark table (same test set for all models) ─────────────
    # Precision@20 = "of the 20 companies we flagged most confidently,
    #                 how many are genuine fraud/distress?"
    # Fixed K means this is comparable across models without the n_positive
    # sensitivity that makes Recall@5% noisy on small test sets.
    #
    # For OUR model the table also shows OOF Recall@5% (from walk-forward CV,
    # pooling 35-60 positives across all folds) — far more stable than the
    # test-set number which is based on only ~12 positives.
    # Baselines have no CV so their OOF column is blank.
    for task in ["fraud", "distress"]:
        bench = results.get(f"{task}_benchmark", [])
        if not bench:
            continue
        print(f"\n  ── {task.upper()} DETECTION ──")
        hdr = (f"  {'Model':<35} {'AUC-ROC':>8} {'AUC-PR':>8} "
               f"{'P@20':>6} {'R@5%*':>7} {'OOF R@5%':>9}")
        print(hdr)
        print("  " + "─"*(len(hdr)-2))
        for row in bench:
            is_ours = "Our Model" in str(row.get("model", ""))
            marker  = " ◀" if is_ours else ""

            # OOF recall — available for our model, blank for baselines
            oof_r = row.get("oof_recall_at_5pct")
            oof_r_str = f"{oof_r:>9.4f}" if oof_r is not None else f"{'—':>9}"

            print(
                f"  {str(row.get('model','')):<35} "
                f"{row.get('auc_roc',      0):>8.4f} "
                f"{row.get('auc_pr',       0):>8.4f} "
                f"{row.get('precision_at_20', 0):>6.4f} "
                f"{row.get('recall_at_5pct',  0):>7.4f} "
                f"{oof_r_str}{marker}"
            )
        n_pos = bench[0].get("n_positive", "?") if bench else "?"
        n_tot = bench[0].get("n_test",     "?") if bench else "?"
        print(f"  * test-set Recall@5% (n_positive={n_pos}, n_test={n_tot} — "
              f"noisy; use OOF R@5% for our model)")

    # ── Section 2: Walk-forward CV summary (our models only) ──────────────────
    # These numbers use ALL folds pooled (35-60 positives for fraud model)
    # so they are the trustworthy headline metrics for our models.
    cv_f = results.get("fraud_cv",    {})
    cv_d = results.get("distress_cv", {})

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "N/A"

    print(f"\n  ── WALK-FORWARD CV — OUT-OF-FOLD (pooled across all folds) ──")
    print(f"  {'':30} {'AUC-ROC':>8} {'AUC-PR':>8} {'Recall@5%':>10} {'Prec@20':>8}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    print(f"  {'Fraud model':<30} "
          f"{_fmt(cv_f.get('oof_auc_roc')):>8} "
          f"{_fmt(cv_f.get('oof_auc_pr')):>8} "
          f"{_fmt(cv_f.get('oof_recall_at_5pct')):>10} "
          f"{_fmt(cv_f.get('oof_precision_at_20')):>8}")
    print(f"  {'Distress model':<30} "
          f"{_fmt(cv_d.get('oof_auc_roc')):>8} "
          f"{_fmt(cv_d.get('oof_auc_pr')):>8} "
          f"{_fmt(cv_d.get('oof_recall_at_5pct')):>10} "
          f"{_fmt(cv_d.get('oof_precision_at_20')):>8}")
    print("═"*W)


def _json_safe(obj):
    """Make numpy types JSON serialisable."""
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    return str(obj)
