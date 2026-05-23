"""
features.py
===========
Engineers 35+ ML-ready features from raw financial line items.

Design principles
-----------------
• NO look-ahead bias — every feature for year T uses only data ≤ T
• Trend features capture CHANGE (what Altman/Beneish missed)
• Interaction features capture COMBINATIONS (e.g. revenue up + cash down)
• Altman + Beneish component variables included raw (so our model can
  re-weight them instead of using their hand-tuned coefficients)

Feature families
----------------
1. RATIO features       (11) — point-in-time accounting ratios
2. TREND features       (8)  — 3-year OLS slope, normalised
3. YOY DELTA features   (7)  — year-over-year % change
4. INTERACTION features (4)  — joint signals (fraud tells both directions)
5. ACCRUAL features     (3)  — Richardson accruals, Sloan ratio
6. BENEISH components   (8)  — raw, so our model can re-weight
7. ALTMAN components    (5)  — raw, so our model can re-weight

Total: 46 features
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


# ── Pivot helper ──────────────────────────────────────────────────────────────
def _pivot(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long[cik, year, concept, value] → wide[cik, year, concept1, concept2, …]
    """
    return long_df.pivot_table(
        index=["cik", "year"],
        columns="concept",
        values="value",
        aggfunc="first",
    ).reset_index()


# ── Safe math ─────────────────────────────────────────────────────────────────
def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _pct_change(curr: pd.Series, prev: pd.Series) -> pd.Series:
    return _div(curr - prev, prev.abs())


def _slope(series: pd.Series) -> float:
    """OLS slope of a short series, normalised by mean absolute value."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    slope = np.polyfit(x, s.values.astype(float), 1)[0]
    base = s.abs().mean() or 1
    return slope / base


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : long DataFrame[cik, ticker, company_name, year, concept, value]
            (output of edgar_bulk.build_raw_table)
    Output: wide DataFrame[cik, ticker, company_name, year, feat1, feat2, …]
            One row per (company, year).  NaNs where data unavailable.
    """
    # ── Wide format per (cik, year) ───────────────────────────────────────────
    meta = (
        raw_df[["cik", "ticker", "company_name", "year"]]
        .drop_duplicates()
    )
    w = _pivot(raw_df)
    w = w.merge(meta.drop_duplicates(["cik", "year"]), on=["cik", "year"], how="left")
    w = w.sort_values(["cik", "year"]).reset_index(drop=True)

    # Convenience aliases
    def c(col):
        return w[col] if col in w.columns else pd.Series(np.nan, index=w.index)

    ta   = c("total_assets")
    ca   = c("current_assets")
    cl   = c("current_liabilities")
    tl   = c("total_liabilities")
    eq   = c("equity")
    re   = c("retained_earnings")
    cash = c("cash")
    rec  = c("receivables")
    inv  = c("inventory")
    rev  = c("revenue")
    gp   = c("gross_profit")
    cogs = c("cogs")
    oi   = c("operating_income")     # EBIT proxy
    ni   = c("net_income")
    ie   = c("interest_expense")
    da   = c("da")
    sga  = c("sga")
    cfo  = c("cfo")
    cfi  = c("cfi")
    capex= c("capex")
    ltd  = c("long_term_debt")
    std  = c("short_term_debt")
    gw   = c("goodwill")
    intang = c("intangibles")

    total_debt = ltd.fillna(0) + std.fillna(0)
    wc = ca - cl
    ebit = oi   # operating income ≈ EBIT

    out = w[["cik", "ticker", "company_name", "year"]].copy()

    # ════════════════════════════════════════════════════════════════
    # FAMILY 1: RATIO FEATURES (point-in-time)
    # ════════════════════════════════════════════════════════════════
    out["f_current_ratio"]      = _div(ca, cl)
    out["f_quick_ratio"]        = _div(ca - inv.fillna(0), cl)
    out["f_cash_ratio"]         = _div(cash, cl)
    out["f_debt_equity"]        = _div(total_debt, eq)
    out["f_debt_assets"]        = _div(tl, ta)
    out["f_interest_coverage"]  = _div(ebit, ie.abs())
    out["f_gross_margin"]       = _div(gp, rev)
    out["f_ebit_margin"]        = _div(ebit, rev)
    out["f_net_margin"]         = _div(ni, rev)
    out["f_roa"]                = _div(ni, ta)
    out["f_roe"]                = _div(ni, eq)
    out["f_asset_turnover"]     = _div(rev, ta)
    out["f_dso"]                = _div(rec, rev) * 365
    out["f_dio"]                = _div(inv, cogs) * 365
    out["f_cfo_ni"]             = _div(cfo, ni)         # <1 = earnings quality issue
    out["f_cfo_revenue"]        = _div(cfo, rev)
    out["f_capex_revenue"]      = _div(capex.abs(), rev)
    out["f_goodwill_assets"]    = _div(gw.fillna(0) + intang.fillna(0), ta)

    # ════════════════════════════════════════════════════════════════
    # FAMILY 2: YOY DELTA FEATURES (% change vs prior year)
    # ════════════════════════════════════════════════════════════════
    def yoy(series: pd.Series) -> pd.Series:
        return w.groupby("cik").apply(
            lambda g: g.sort_values("year")[series.name].pct_change()
        ).reset_index(level=0, drop=True).reindex(w.index)

    # Assign names so groupby can reference them
    rev.name  = "revenue"
    cfo.name  = "cfo"
    rec.name  = "receivables"
    ta.name   = "total_assets"
    ni.name   = "net_income"
    tl.name   = "total_liabilities"

    out["f_yoy_revenue"]        = w.groupby("cik")["revenue"].pct_change()         if "revenue"         in w else np.nan
    out["f_yoy_cfo"]            = w.groupby("cik")["cfo"].pct_change()             if "cfo"             in w else np.nan
    out["f_yoy_receivables"]    = w.groupby("cik")["receivables"].pct_change()     if "receivables"     in w else np.nan
    out["f_yoy_assets"]         = w.groupby("cik")["total_assets"].pct_change()    if "total_assets"    in w else np.nan
    out["f_yoy_net_income"]     = w.groupby("cik")["net_income"].pct_change()      if "net_income"      in w else np.nan
    out["f_yoy_debt"]           = w.groupby("cik")["total_liabilities"].pct_change()if "total_liabilities" in w else np.nan
    # Revenue - CFO growth divergence (THE classic fraud signal)
    out["f_rev_cfo_divergence"] = out["f_yoy_revenue"].fillna(0) - out["f_yoy_cfo"].fillna(0)

    # ════════════════════════════════════════════════════════════════
    # FAMILY 3: TREND FEATURES (3-yr OLS slope, normalised)
    # ════════════════════════════════════════════════════════════════
    trend_features = {
        "f_trend_current_ratio": out["f_current_ratio"],
        "f_trend_gross_margin":  out["f_gross_margin"],
        "f_trend_net_margin":    out["f_net_margin"],
        "f_trend_cfo_ni":        out["f_cfo_ni"],
        "f_trend_debt_equity":   out["f_debt_equity"],
        "f_trend_roa":           out["f_roa"],
        "f_trend_dso":           out["f_dso"],
        "f_trend_cfo_rev":       out["f_cfo_revenue"],
    }

    for feat_name, feat_series in trend_features.items():
        tmp = feat_series.copy()
        tmp.name = "_tmp"
        slope_col = (
            pd.concat([w["cik"], w["year"], tmp], axis=1)
            .sort_values(["cik", "year"])
            .groupby("cik")["_tmp"]
            .transform(lambda s: s.rolling(3, min_periods=2).apply(_slope, raw=False))
        )
        out[feat_name] = slope_col.values

    # ════════════════════════════════════════════════════════════════
    # FAMILY 4: INTERACTION FEATURES
    # ════════════════════════════════════════════════════════════════
    # Revenue up + CFO down simultaneously (both must be true for flag = 1)
    rev_up  = (out["f_yoy_revenue"] > 0.05).astype(float)
    cfo_dn  = (out["f_yoy_cfo"]     < -0.05).astype(float)
    out["f_interact_rev_up_cfo_dn"] = rev_up * cfo_dn

    # DSO rising + Revenue rising (receivables growing faster than sales)
    dso_up  = (out["f_trend_dso"] > 0).astype(float)
    rev_up2 = (out["f_yoy_revenue"] > 0.03).astype(float)
    out["f_interact_dso_rev"]       = dso_up * rev_up2

    # Earnings quality: NI positive but CFO negative
    ni_pos  = (ni > 0).astype(float)
    cfo_neg = (cfo < 0).astype(float)
    out["f_interact_ni_pos_cfo_neg"]= ni_pos * cfo_neg

    # Leverage rising + Coverage falling
    lev_up  = (out["f_trend_debt_equity"] > 0).astype(float)
    cov_dn  = (out["f_interest_coverage"] < 3).astype(float)
    out["f_interact_lev_cov"]       = lev_up * cov_dn

    # ════════════════════════════════════════════════════════════════
    # FAMILY 5: ACCRUAL QUALITY FEATURES
    # ════════════════════════════════════════════════════════════════
    # Sloan (1996) accrual ratio: (NI - CFO) / avg_assets
    # High accruals → lower future earnings → manipulation signal
    out["f_sloan_accrual"]      = _div(ni - cfo, ta)

    # Richardson et al. (2005) total accruals
    # ΔWC + ΔNCO + ΔFIN scaled by avg assets
    # Simplified: (NI - CFO) / total_assets (same direction)
    out["f_total_accruals"]     = _div(ni - cfo, ta)

    # Cash-based ROA vs reported ROA difference (earnings manipulation proxy)
    out["f_cash_roa_gap"]       = _div(cfo - ni, ta)

    # ════════════════════════════════════════════════════════════════
    # FAMILY 6: BENEISH M-SCORE COMPONENT VARIABLES (raw)
    # Let our ML re-weight these instead of using Beneish's 1999 coefficients
    # ════════════════════════════════════════════════════════════════
    # DSRI: Days Sales Receivables Index = (rec_t/rev_t) / (rec_{t-1}/rev_{t-1})
    dsr_curr = _div(rec, rev)
    dsr_prev = w.groupby("cik")["receivables"].shift(1) / w.groupby("cik")["revenue"].shift(1) if "receivables" in w and "revenue" in w else pd.Series(np.nan, index=w.index)
    out["f_b_dsri"]     = _div(dsr_curr, dsr_prev)

    # GMI: Gross Margin Index = gm_{t-1} / gm_t  (>1 means margin deteriorating)
    gm_curr = _div(gp, rev)
    gm_prev = w.groupby("cik")["gross_profit"].shift(1) / w.groupby("cik")["revenue"].shift(1) if "gross_profit" in w and "revenue" in w else pd.Series(np.nan, index=w.index)
    out["f_b_gmi"]      = _div(gm_prev, gm_curr)

    # AQI: (1 - (CA + PPE) / TA)_t / (1 - (CA + PPE) / TA)_{t-1}
    # Simplified without PPE breakout:
    nca_ratio = 1 - _div(ca, ta)
    nca_prev  = 1 - w.groupby("cik")["current_assets"].shift(1) / w.groupby("cik")["total_assets"].shift(1) if "current_assets" in w else pd.Series(np.nan, index=w.index)
    out["f_b_aqi"]      = _div(nca_ratio, nca_prev)

    # SGI: Sales Growth Index
    rev_prev = w.groupby("cik")["revenue"].shift(1) if "revenue" in w else pd.Series(np.nan, index=w.index)
    out["f_b_sgi"]      = _div(w["revenue"] if "revenue" in w else pd.Series(np.nan, index=w.index), rev_prev)

    # DEPI: Depreciation Index
    if "da" in w and "total_assets" in w:
        dep_rate_curr = _div(da, da + ta)
        dep_rate_prev = w.groupby("cik")["da"].shift(1) / (w.groupby("cik")["da"].shift(1) + w.groupby("cik")["total_assets"].shift(1))
        out["f_b_depi"] = _div(dep_rate_prev, dep_rate_curr)
    else:
        out["f_b_depi"] = np.nan

    # SGAI: SGA Index
    if "sga" in w and "revenue" in w:
        sga_ratio_curr = _div(sga, rev)
        sga_ratio_prev = w.groupby("cik")["sga"].shift(1) / w.groupby("cik")["revenue"].shift(1)
        out["f_b_sgai"] = _div(sga_ratio_curr, sga_ratio_prev)
    else:
        out["f_b_sgai"] = np.nan

    # LVGI: Leverage Index
    if "total_liabilities" in w and "total_assets" in w:
        lev_curr = _div(tl, ta)
        lev_prev = w.groupby("cik")["total_liabilities"].shift(1) / w.groupby("cik")["total_assets"].shift(1)
        out["f_b_lvgi"] = _div(lev_curr, lev_prev)
    else:
        out["f_b_lvgi"] = np.nan

    # TATA: Total Accruals to Total Assets
    out["f_b_tata"] = _div(ni - cfo, ta)

    # ════════════════════════════════════════════════════════════════
    # FAMILY 7: ALTMAN Z-SCORE COMPONENT VARIABLES (raw)
    # ════════════════════════════════════════════════════════════════
    out["f_z_x1"]  = _div(wc,  ta)    # Working Capital / Assets
    out["f_z_x2"]  = _div(re,  ta)    # Retained Earnings / Assets
    out["f_z_x3"]  = _div(ebit,ta)    # EBIT / Assets
    # X4 needs market cap (added later when merging with yfinance data)
    out["f_z_x4"]  = np.nan           # Market Cap / Total Liabilities  (placeholder)
    out["f_z_x5"]  = _div(rev, ta)    # Revenue / Assets

    # ════════════════════════════════════════════════════════════════
    # Clip extreme outliers (winsorise at 1st/99th percentile per column)
    # ════════════════════════════════════════════════════════════════
    feat_cols = [c for c in out.columns if c.startswith("f_")]
    for col in feat_cols:
        lo = out[col].quantile(0.01)
        hi = out[col].quantile(0.99)
        out[col] = out[col].clip(lo, hi)

    logger.info(
        f"Feature engineering done: {len(out):,} rows, "
        f"{len(feat_cols)} features"
    )
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    raw_path = ROOT / "data" / "processed" / "raw_financials.parquet"
    if not raw_path.exists():
        print(f"Raw financials not found at {raw_path}.")
        print("Run pipeline/edgar_bulk.py first.")
    else:
        raw = pd.read_parquet(raw_path)
        feat = engineer_features(raw)
        out  = ROOT / "data" / "processed" / "features.parquet"
        feat.to_parquet(out, index=False)
        print(feat.head(10).to_string())
        print(f"\nShape: {feat.shape}")
        print(f"Saved → {out}")
