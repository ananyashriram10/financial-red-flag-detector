"""
base.py
=======
Abstract base class shared by FraudModel and DistressModel.

Both models have identical architecture and training loop — only the
target column and a few hyperparameters differ. This avoids duplication.

Architecture per model
----------------------
  XGBoost classifier          (supervised, 46 features, SMOTE)
+ Isolation Forest             (unsupervised anomaly, same features)
= Weighted ensemble score      (XGB 65% + IsoForest 35%)

Training protocol
-----------------
  Walk-forward cross-validation (temporal, no look-ahead)
  SMOTE applied only inside each training fold
  Hyperparameters tuned on validation AUC-PR (not accuracy)

Output per prediction
---------------------
  score      : float [0, 1]    ensemble fraud/distress probability
  top3       : list[dict]      top 3 SHAP features driving the score
  xgb_prob   : float           raw XGBoost probability
  iso_score  : float           normalised Isolation Forest anomaly score
"""

from __future__ import annotations
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


# ── Feature columns used for training ────────────────────────────────────────
# These must match exactly what pipeline/features.py produces
FEATURE_COLS = [
    # Ratio features
    "f_current_ratio", "f_quick_ratio", "f_cash_ratio",
    "f_debt_equity", "f_debt_assets", "f_interest_coverage",
    "f_gross_margin", "f_ebit_margin", "f_net_margin",
    "f_roa", "f_roe", "f_asset_turnover",
    "f_dso", "f_dio", "f_cfo_ni", "f_cfo_revenue",
    "f_capex_revenue", "f_goodwill_assets",
    # YoY delta features
    "f_yoy_revenue", "f_yoy_cfo", "f_yoy_receivables",
    "f_yoy_assets", "f_yoy_net_income", "f_yoy_debt",
    "f_rev_cfo_divergence",
    # Trend features (3-yr OLS slope)
    "f_trend_current_ratio", "f_trend_gross_margin", "f_trend_net_margin",
    "f_trend_cfo_ni", "f_trend_debt_equity", "f_trend_roa",
    "f_trend_dso", "f_trend_cfo_rev",
    # Interaction features
    "f_interact_rev_up_cfo_dn", "f_interact_dso_rev",
    "f_interact_ni_pos_cfo_neg", "f_interact_lev_cov",
    # Accrual features
    "f_sloan_accrual", "f_total_accruals", "f_cash_roa_gap",
    # Beneish components (raw — model re-weights them)
    "f_b_dsri", "f_b_gmi", "f_b_aqi", "f_b_sgi",
    "f_b_depi", "f_b_sgai", "f_b_lvgi", "f_b_tata",
    # Altman components (raw — model re-weights them)
    "f_z_x1", "f_z_x2", "f_z_x3", "f_z_x5",
]

# Human-readable names for the app's "top 3 features" display
FEATURE_LABELS = {
    "f_current_ratio":           "Current Ratio",
    "f_quick_ratio":             "Quick Ratio",
    "f_cash_ratio":              "Cash Ratio",
    "f_debt_equity":             "Debt / Equity",
    "f_debt_assets":             "Debt / Assets",
    "f_interest_coverage":       "Interest Coverage",
    "f_gross_margin":            "Gross Margin",
    "f_ebit_margin":             "EBIT Margin",
    "f_net_margin":              "Net Margin",
    "f_roa":                     "Return on Assets",
    "f_roe":                     "Return on Equity",
    "f_asset_turnover":          "Asset Turnover",
    "f_dso":                     "Days Sales Outstanding",
    "f_dio":                     "Days Inventory Outstanding",
    "f_cfo_ni":                  "CFO / Net Income Ratio",
    "f_cfo_revenue":             "CFO / Revenue",
    "f_capex_revenue":           "CapEx / Revenue",
    "f_goodwill_assets":         "Goodwill + Intangibles / Assets",
    "f_yoy_revenue":             "Revenue Growth (YoY)",
    "f_yoy_cfo":                 "CFO Growth (YoY)",
    "f_yoy_receivables":         "Receivables Growth (YoY)",
    "f_yoy_assets":              "Asset Growth (YoY)",
    "f_yoy_net_income":          "Net Income Growth (YoY)",
    "f_yoy_debt":                "Debt Growth (YoY)",
    "f_rev_cfo_divergence":      "Revenue–CFO Growth Divergence",
    "f_trend_current_ratio":     "Current Ratio Trend (3yr)",
    "f_trend_gross_margin":      "Gross Margin Trend (3yr)",
    "f_trend_net_margin":        "Net Margin Trend (3yr)",
    "f_trend_cfo_ni":            "CFO/NI Trend (3yr)",
    "f_trend_debt_equity":       "Debt/Equity Trend (3yr)",
    "f_trend_roa":               "ROA Trend (3yr)",
    "f_trend_dso":               "DSO Trend (3yr)",
    "f_trend_cfo_rev":           "CFO/Revenue Trend (3yr)",
    "f_interact_rev_up_cfo_dn":  "Revenue ↑ while CFO ↓",
    "f_interact_dso_rev":        "Rising DSO + Growing Revenue",
    "f_interact_ni_pos_cfo_neg": "Positive NI but Negative CFO",
    "f_interact_lev_cov":        "Rising Leverage + Weak Coverage",
    "f_sloan_accrual":           "Sloan Accrual Ratio",
    "f_total_accruals":          "Total Accruals / Assets",
    "f_cash_roa_gap":            "Cash ROA Gap",
    "f_b_dsri":                  "Beneish: Receivables Index (DSRI)",
    "f_b_gmi":                   "Beneish: Gross Margin Index (GMI)",
    "f_b_aqi":                   "Beneish: Asset Quality Index (AQI)",
    "f_b_sgi":                   "Beneish: Sales Growth Index (SGI)",
    "f_b_depi":                  "Beneish: Depreciation Index (DEPI)",
    "f_b_sgai":                  "Beneish: SGA Index (SGAI)",
    "f_b_lvgi":                  "Beneish: Leverage Index (LVGI)",
    "f_b_tata":                  "Beneish: Total Accruals (TATA)",
    "f_z_x1":                    "Altman: Working Capital / Assets",
    "f_z_x2":                    "Altman: Retained Earnings / Assets",
    "f_z_x3":                    "Altman: EBIT / Assets",
    "f_z_x5":                    "Altman: Revenue / Assets",
}


# ── Walk-forward cross-validation ────────────────────────────────────────────
def walk_forward_splits(
    df: pd.DataFrame,
    min_train_years: int = 8,
    test_window:     int = 3,
    step:            int = 3,
):
    """
    Generator of (train_idx, val_idx) index pairs for temporal CV.

    Example with min_train_years=8, test_window=3, step=3:
      Fold 1: train ≤ 2000, val 2001-2003
      Fold 2: train ≤ 2003, val 2004-2006
      ...

    Never lets future data bleed into training — the only correct way
    to validate time-series financial models.
    """
    years     = sorted(df["year"].unique())
    min_year  = min(years)
    test_start = min_year + min_train_years

    while test_start + test_window - 1 <= max(years):
        train_idx = df.index[df["year"] < test_start].tolist()
        val_idx   = df.index[
            (df["year"] >= test_start) &
            (df["year"] <  test_start + test_window)
        ].tolist()

        if len(train_idx) > 0 and len(val_idx) > 0:
            yield train_idx, val_idx

        test_start += step


# ── GPU detection ────────────────────────────────────────────────────────────
def _detect_device() -> str:
    """Return 'cuda' if a GPU is available (Kaggle T4/P100), else 'cpu'."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=3
        )
        if result.returncode == 0:
            logger.info("GPU detected — XGBoost will use CUDA (T4/P100)")
            return "cuda"
    except Exception:
        pass
    logger.info("No GPU detected — XGBoost will use CPU")
    return "cpu"


_DEVICE = _detect_device()


# ── Base model class ──────────────────────────────────────────────────────────
class BaseRiskModel(ABC):
    """
    Shared training / evaluation / prediction logic.
    Subclasses define: target_col, model_name, xgb_params overrides.
    """

    # --- override in subclass ---
    target_col: str  = ""
    model_name: str  = ""

    # XGBoost defaults (subclass can override specific keys)
    # tree_method='hist' + device='cuda' → GPU acceleration on Kaggle T4
    # Automatically falls back to CPU when no GPU present
    _xgb_base_params = {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",           # optimise for precision-recall AUC
        "tree_method":      "hist",            # works on both CPU and GPU
        "device":           _DEVICE,           # 'cuda' on Kaggle T4, 'cpu' locally
        "max_depth":        5,
        "learning_rate":    0.05,
        "n_estimators":     400,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    }

    def __init__(self, xgb_overrides: dict | None = None):
        params = {**self._xgb_base_params, **(xgb_overrides or {})}
        self.xgb_model    = xgb.XGBClassifier(**params)
        self.iso_model    = IsolationForest(
            n_estimators=200,
            contamination=0.02,   # ~2% positive rate
            random_state=42,
            n_jobs=-1,
        )
        self.iso_scaler   = MinMaxScaler()
        self.explainer    = None   # SHAP TreeExplainer, built after fit
        self._feature_cols: list[str] = []
        self._is_fitted   = False

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _prep(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Return X with available feature columns; NaN-fill with median."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].copy()
        X = X.fillna(X.median(numeric_only=True))
        return X, available

    def _smote_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to balance classes — TRAINING ONLY."""
        pos = y.sum()
        if pos < 6 or pos / len(y) > 0.3:
            return X, y   # too few positives or already balanced
        k = min(5, pos - 1)
        sm = SMOTE(k_neighbors=int(k), random_state=42)
        try:
            Xr, yr = sm.fit_resample(X, y)
            return pd.DataFrame(Xr, columns=X.columns), pd.Series(yr)
        except Exception as e:
            logger.warning(f"SMOTE failed ({e}), using original data")
            return X, y

    def _iso_score(self, X: pd.DataFrame) -> np.ndarray:
        """Isolation Forest → normalised anomaly score [0, 1], higher = riskier."""
        raw = -self.iso_model.decision_function(X)   # more negative = more anomalous
        return self.iso_scaler.transform(raw.reshape(-1, 1)).ravel()

    def _ensemble(self, xgb_prob: np.ndarray,
                  iso_score: np.ndarray) -> np.ndarray:
        return 0.65 * xgb_prob + 0.35 * iso_score

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "BaseRiskModel":
        """
        Full training run on the complete dataset.
        Call after walk_forward_evaluate() to get the production model.
        """
        assert self.target_col in df.columns, \
            f"Column '{self.target_col}' not found in DataFrame"

        X, feat_cols = self._prep(df)
        y = df[self.target_col].astype(int)
        self._feature_cols = feat_cols

        # Class weight for XGBoost
        pos = y.sum()
        neg = len(y) - pos
        spw = neg / max(pos, 1)
        self.xgb_model.set_params(scale_pos_weight=spw)

        # SMOTE on full training set
        Xs, ys = self._smote_resample(X, y)
        logger.info(f"[{self.model_name}] Training on {len(Xs):,} rows "
                    f"(after SMOTE; original: {len(X):,}, "
                    f"positives: {pos}, scale_pos_weight={spw:.1f})")

        self.xgb_model.fit(Xs, ys)

        # Isolation Forest on original (no SMOTE for unsupervised)
        self.iso_model.fit(X)
        iso_raw = -self.iso_model.decision_function(X)
        self.iso_scaler.fit(iso_raw.reshape(-1, 1))

        # Build SHAP explainer
        self.explainer = shap.TreeExplainer(self.xgb_model)

        self._is_fitted = True
        logger.info(f"[{self.model_name}] Training complete.")
        return self

    # ── Walk-forward evaluation ───────────────────────────────────────────────
    def walk_forward_evaluate(
        self,
        df:              pd.DataFrame,
        min_train_years: int = 8,
        test_window:     int = 3,
        step:            int = 3,
    ) -> dict:
        """
        Run walk-forward CV and return per-fold + aggregate metrics.
        Does NOT update self — call fit() separately for the production model.
        """
        assert self.target_col in df.columns

        df = df.reset_index(drop=True)
        X_all, feat_cols = self._prep(df)
        y_all = df[self.target_col].astype(int)

        fold_metrics = []
        oof_scores   = np.full(len(df), np.nan)
        oof_labels   = np.full(len(df), np.nan)

        splits = list(walk_forward_splits(df, min_train_years,
                                          test_window, step))
        if not splits:
            logger.warning("No valid walk-forward splits — dataset too small.")
            return {}

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            X_tr = X_all.iloc[train_idx]
            y_tr = y_all.iloc[train_idx]
            X_vl = X_all.iloc[val_idx]
            y_vl = y_all.iloc[val_idx]

            if y_tr.sum() < 5:
                logger.debug(f"Fold {fold_i}: too few positives in train, skip")
                continue

            # Clone and train on this fold
            spw   = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
            model = xgb.XGBClassifier(**{
                **self._xgb_base_params,
                "scale_pos_weight": spw,
            })
            X_sm, y_sm = self._smote_resample(X_tr, y_tr)
            model.fit(X_sm, y_sm, verbose=False)

            iso   = IsolationForest(n_estimators=200, contamination=0.02,
                                    random_state=42, n_jobs=-1)
            iso.fit(X_tr)
            scaler = MinMaxScaler()
            scaler.fit((-iso.decision_function(X_tr)).reshape(-1, 1))

            xgb_prob  = model.predict_proba(X_vl)[:, 1]
            iso_raw   = -iso.decision_function(X_vl)
            iso_norm  = scaler.transform(iso_raw.reshape(-1, 1)).ravel()
            ensemble  = 0.65 * xgb_prob + 0.35 * iso_norm

            oof_scores[val_idx]  = ensemble
            oof_labels[val_idx]  = y_vl.values

            if y_vl.sum() < 2:
                continue   # can't compute AUC without both classes

            fold_metrics.append({
                "fold":          fold_i + 1,
                "train_years":   f"{df['year'].iloc[train_idx].min()}–"
                                 f"{df['year'].iloc[train_idx].max()}",
                "val_years":     f"{df['year'].iloc[val_idx].min()}–"
                                 f"{df['year'].iloc[val_idx].max()}",
                "n_train":       len(train_idx),
                "n_val":         len(val_idx),
                "n_fraud_val":   int(y_vl.sum()),
                "auc_roc":       round(roc_auc_score(y_vl, ensemble), 4),
                "auc_pr":        round(average_precision_score(y_vl, ensemble), 4),
                "auc_roc_xgb":   round(roc_auc_score(y_vl, xgb_prob), 4),
                "auc_roc_iso":   round(roc_auc_score(y_vl, iso_norm), 4),
            })

        # Out-of-fold aggregate
        mask = ~np.isnan(oof_scores)
        result = {
            "fold_metrics": pd.DataFrame(fold_metrics),
            "oof_auc_roc":  round(roc_auc_score(oof_labels[mask],
                                                  oof_scores[mask]), 4)
                            if mask.sum() > 0 else None,
            "oof_auc_pr":   round(average_precision_score(oof_labels[mask],
                                                           oof_scores[mask]), 4)
                            if mask.sum() > 0 else None,
        }

        # OOF Recall@5% and Precision@20
        # These are computed on the full pooled OOF predictions (all folds
        # combined), not on the small held-out test set.  With ~30k rows and
        # multiple folds the OOF pool typically contains 35-60 positive cases
        # for the fraud model vs only ~12 in the test set, making these
        # numbers far more stable and reliable than test-set recall.
        oof_s = oof_scores[mask]
        oof_l = oof_labels[mask]
        if mask.sum() > 0 and oof_l.sum() > 0:
            # Recall @ top 5%
            thr = np.percentile(oof_s, 95)
            flagged = oof_s >= thr
            oof_recall_5 = float(oof_l[flagged].sum() / oof_l.sum())

            # Precision @ top 20
            top20_idx = np.argsort(-oof_s)[:20]
            oof_prec_20 = float(oof_l[top20_idx].sum()) / 20.0

            result["oof_recall_at_5pct"]  = round(oof_recall_5, 4)
            result["oof_precision_at_20"] = round(oof_prec_20,  4)
        else:
            result["oof_recall_at_5pct"]  = None
            result["oof_precision_at_20"] = None

        logger.info(f"\n[{self.model_name}] Walk-forward results:")
        logger.info(f"  OOF AUC-ROC      : {result['oof_auc_roc']}")
        logger.info(f"  OOF AUC-PR       : {result['oof_auc_pr']}")
        logger.info(f"  OOF Recall@5%    : {result['oof_recall_at_5pct']}")
        logger.info(f"  OOF Precision@20 : {result['oof_precision_at_20']}")
        if not fold_metrics:
            return result
        fdf = result["fold_metrics"]
        logger.info(f"  Per-fold AUC-ROC: "
                    f"mean={fdf['auc_roc'].mean():.4f} "
                    f"std={fdf['auc_roc'].std():.4f}")

        return result

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on a DataFrame. Returns df with added columns:
          score      float   ensemble probability [0, 1]
          xgb_prob   float   raw XGBoost probability
          iso_score  float   normalised Isolation Forest score
          top3       list    top 3 feature dicts driving the score
        """
        assert self._is_fitted, "Call fit() before predict()"

        X, _ = self._prep(df)
        X = X[self._feature_cols].fillna(X[self._feature_cols].median())

        xgb_prob  = self.xgb_model.predict_proba(X)[:, 1]
        iso_score = self._iso_score(X)
        score     = self._ensemble(xgb_prob, iso_score)

        # SHAP top-3
        shap_vals = self.explainer.shap_values(X)   # shape (n, n_features)
        top3_list = self._top3_features(X, shap_vals)

        out = df.copy()
        out["score"]     = score
        out["xgb_prob"]  = xgb_prob
        out["iso_score"] = iso_score
        out["top3"]      = top3_list
        return out

    def predict_single(self, features: dict) -> dict:
        """
        Predict for one company-year given a flat feature dict.
        Returns: {score, xgb_prob, iso_score, top3}
        """
        row = pd.DataFrame([features])
        X, _ = self._prep(row)
        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self._feature_cols].fillna(0)

        xgb_prob  = float(self.xgb_model.predict_proba(X)[:, 1][0])
        iso_score = float(self._iso_score(X)[0])
        score     = float(self._ensemble(
            np.array([xgb_prob]), np.array([iso_score]))[0])

        shap_vals  = self.explainer.shap_values(X)
        top3       = self._top3_features(X, shap_vals)[0]

        return {
            "score":     round(score,     4),
            "xgb_prob":  round(xgb_prob,  4),
            "iso_score": round(iso_score, 4),
            "top3":      top3,
        }

    def _top3_features(
        self, X: pd.DataFrame, shap_vals: np.ndarray
    ) -> list[list[dict]]:
        """
        For each row, return the top 3 features by |SHAP value|.
        Each feature dict:
          name        : human-readable label
          shap_value  : float (positive = increases risk, negative = decreases)
          feature_val : actual value of that feature for this row
          direction   : "increases_risk" | "decreases_risk"
        """
        results = []
        for i in range(len(X)):
            sv   = shap_vals[i]
            idxs = np.argsort(np.abs(sv))[::-1][:3]
            top3 = []
            for j in idxs:
                fname = self._feature_cols[j]
                sv_j  = float(sv[j])
                top3.append({
                    "name":        FEATURE_LABELS.get(fname, fname),
                    "feature_key": fname,
                    "shap_value":  round(sv_j, 4),
                    "feature_val": round(float(X.iloc[i, j]), 4),
                    "direction":   "increases_risk" if sv_j > 0 else "decreases_risk",
                    "impact_pct":  f"{abs(sv_j)*100:.1f}%",
                })
            results.append(top3)
        return results

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"{self.model_name}.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"[{self.model_name}] Saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseRiskModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded model from {path}")
        return obj
