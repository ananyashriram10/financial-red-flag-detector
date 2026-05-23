"""
metrics.py — Compute financial ratios from EDGAR data frames.
All calculations return a dict[year → value] or None on missing data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def _align(*frames: pd.DataFrame) -> pd.DataFrame:
    """Inner-join multiple [year, value] frames on year, rename value columns."""
    if not frames:
        return pd.DataFrame()
    result = frames[0].rename(columns={"value": "v0"}).copy()
    for i, df in enumerate(frames[1:], 1):
        result = result.merge(df.rename(columns={"value": f"v{i}"}),
                              on="year", how="inner")
    return result


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.div(b.replace(0, np.nan))


def compute_ratios(d: dict) -> pd.DataFrame:
    """
    Given the dict from edgar.get_financial_data(), compute all ratio
    time-series.  Returns a wide DataFrame indexed by year.
    """
    rows = []

    # Helper: merge two series by year
    def merge2(ka, kb):
        a = d.get(ka, pd.DataFrame())
        b = d.get(kb, pd.DataFrame())
        if a.empty or b.empty:
            return pd.DataFrame()
        return _align(a, b)

    # ── Liquidity ─────────────────────────────────────────────────────────
    m = merge2("current_assets", "current_liabilities")
    if not m.empty:
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "current_ratio",
                         "value": r.v0 / r.v1 if r.v1 else np.nan})

    m = merge2("current_assets", "current_liabilities")
    cash_df = d.get("cash", pd.DataFrame())
    inv_df  = d.get("inventory", pd.DataFrame())
    if not m.empty and not cash_df.empty:
        m2 = _align(d["current_assets"], d["current_liabilities"],
                    cash_df)
        if not inv_df.empty:
            m3 = _align(d["current_assets"], d["current_liabilities"],
                        cash_df, inv_df)
            for _, r in m3.iterrows():
                quick = (r.v0 - r.v3) / r.v1 if r.v1 else np.nan
                rows.append({"year": r.year, "metric": "quick_ratio", "value": quick})
        else:
            for _, r in m2.iterrows():
                quick = (r.v0 - r.v2) / r.v1 if r.v1 else np.nan
                rows.append({"year": r.year, "metric": "quick_ratio", "value": quick})

    # Cash ratio
    m = merge2("cash", "current_liabilities")
    if not m.empty:
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "cash_ratio",
                         "value": r.v0 / r.v1 if r.v1 else np.nan})

    # ── Solvency ──────────────────────────────────────────────────────────
    # Debt / Equity
    ltd = d.get("ltd", pd.DataFrame())
    std = d.get("std", pd.DataFrame())
    eq  = d.get("equity", pd.DataFrame())
    if not ltd.empty and not eq.empty:
        if not std.empty:
            m = _align(ltd, std, eq)
            for _, r in m.iterrows():
                debt = r.v0 + r.v1
                rows.append({"year": r.year, "metric": "debt_equity",
                             "value": debt / r.v2 if r.v2 else np.nan})
                rows.append({"year": r.year, "metric": "total_debt",
                             "value": debt})
        else:
            m = _align(ltd, eq)
            for _, r in m.iterrows():
                rows.append({"year": r.year, "metric": "debt_equity",
                             "value": r.v0 / r.v1 if r.v1 else np.nan})
                rows.append({"year": r.year, "metric": "total_debt",
                             "value": r.v0})

    # Debt / Assets
    liab = d.get("total_liabilities", pd.DataFrame())
    assets = d.get("total_assets", pd.DataFrame())
    if not liab.empty and not assets.empty:
        m = _align(liab, assets)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "debt_assets",
                         "value": r.v0 / r.v1 if r.v1 else np.nan})

    # Interest coverage (EBIT / Interest Expense)
    ebit = d.get("operating_income", pd.DataFrame())
    ie   = d.get("interest_expense", pd.DataFrame())
    if not ebit.empty and not ie.empty:
        m = _align(ebit, ie)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "interest_coverage",
                         "value": r.v0 / abs(r.v1) if r.v1 else np.nan})

    # ── Profitability ─────────────────────────────────────────────────────
    rev = d.get("revenue", pd.DataFrame())
    gp  = d.get("gross_profit", pd.DataFrame())
    ni  = d.get("net_income", pd.DataFrame())
    cfo = d.get("cfo", pd.DataFrame())

    if not rev.empty and not gp.empty:
        m = _align(rev, gp)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "gross_margin",
                         "value": r.v1 / r.v0 if r.v0 else np.nan})

    if not rev.empty and not ebit.empty:
        m = _align(rev, ebit)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "ebit_margin",
                         "value": r.v1 / r.v0 if r.v0 else np.nan})

    if not rev.empty and not ni.empty:
        m = _align(rev, ni)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "net_margin",
                         "value": r.v1 / r.v0 if r.v0 else np.nan})

    if not ni.empty and not assets.empty:
        m = _align(ni, assets)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "roa",
                         "value": r.v0 / r.v1 if r.v1 else np.nan})

    if not ni.empty and not eq.empty:
        m = _align(ni, eq)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "roe",
                         "value": r.v0 / r.v1 if r.v1 else np.nan})

    # ── Cash quality ──────────────────────────────────────────────────────
    # CFO / Net Income — Accruals signal (want ≥ 1.0)
    if not cfo.empty and not ni.empty:
        m = _align(cfo, ni)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "cfo_ni_ratio",
                         "value": r.v0 / r.v1 if r.v1 else np.nan})

    # CFO / Revenue — cash conversion
    if not cfo.empty and not rev.empty:
        m = _align(cfo, rev)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "cfo_revenue",
                         "value": r.v0 / r.v1 if r.v1 else np.nan})

    # Revenue vs CFO growth divergence  (raw values for chart)
    if not rev.empty:
        for _, r in rev.iterrows():
            rows.append({"year": r.year, "metric": "revenue", "value": r.value})
    if not cfo.empty:
        for _, r in cfo.iterrows():
            rows.append({"year": r.year, "metric": "cfo", "value": r.value})
    if not ni.empty:
        for _, r in ni.iterrows():
            rows.append({"year": r.year, "metric": "net_income", "value": r.value})

    # ── Efficiency ────────────────────────────────────────────────────────
    rec = d.get("receivables", pd.DataFrame())
    if not rec.empty and not rev.empty:
        m = _align(rec, rev)
        for _, r in m.iterrows():
            dso = (r.v0 / r.v1) * 365 if r.v1 else np.nan
            rows.append({"year": r.year, "metric": "dso", "value": dso})

    inv_df = d.get("inventory", pd.DataFrame())
    cogs_df = d.get("cogs", pd.DataFrame())
    if not inv_df.empty and not cogs_df.empty:
        m = _align(inv_df, cogs_df)
        for _, r in m.iterrows():
            dio = (r.v0 / r.v1) * 365 if r.v1 else np.nan
            rows.append({"year": r.year, "metric": "dio", "value": dio})

    # Asset turnover
    if not rev.empty and not assets.empty:
        m = _align(rev, assets)
        for _, r in m.iterrows():
            rows.append({"year": r.year, "metric": "asset_turnover",
                         "value": r.v0 / r.v1 if r.v1 else np.nan})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="year", columns="metric", values="value",
                           aggfunc="first")
    pivot = pivot.sort_index()
    # YoY revenue growth
    if "revenue" in pivot.columns:
        pivot["revenue_growth"] = pivot["revenue"].pct_change()
    if "cfo" in pivot.columns:
        pivot["cfo_growth"] = pivot["cfo"].pct_change()
    if "gross_margin" in pivot.columns:
        pivot["gm_delta"] = pivot["gross_margin"].diff()
    return pivot


def altman_z(d: dict, market_cap: Optional[float] = None) -> Optional[float]:
    """
    Altman Z-score for public companies.
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets
    """
    def latest(key):
        df = d.get(key, pd.DataFrame())
        return df["value"].iloc[-1] if not df.empty else None

    ta   = latest("total_assets")
    ca   = latest("current_assets")
    cl   = latest("current_liabilities")
    re   = latest("retained_earnings")
    ebit = latest("operating_income")
    liab = latest("total_liabilities")
    rev  = latest("revenue")

    if any(v is None for v in [ta, ca, cl, re, ebit, liab, rev]):
        return None
    if ta == 0 or liab == 0:
        return None

    wc = ca - cl
    X1 = wc   / ta
    X2 = re   / ta
    X3 = ebit / ta
    X4 = (market_cap or 1) / liab
    X5 = rev  / ta

    return 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5


def beneish_m(d: dict) -> Optional[float]:
    """
    Beneish M-Score — earnings manipulation detector.
    Needs at least 2 years of data.
    Returns score for most recent year. Score > -2.22 → likely manipulator.
    """
    def series(key):
        df = d.get(key, pd.DataFrame())
        if df.empty or len(df) < 2:
            return None
        return df.set_index("year")["value"]

    rev  = series("revenue")
    rec  = series("receivables")
    gp   = series("gross_profit")
    cogs = series("cogs")
    assets = series("total_assets")
    da   = series("da")
    sga  = series("sga")
    liab = series("total_liabilities")
    ni   = series("net_income")
    cfo  = series("cfo")

    # Require key series
    if any(s is None for s in [rev, rec, gp, assets]):
        return None

    # Align to common years (last two)
    years = sorted(set(rev.index) & set(rec.index) &
                   set(gp.index) & set(assets.index))
    if len(years) < 2:
        return None
    y0, y1 = years[-2], years[-1]

    def v(s, y):
        return s[y] if s is not None and y in s.index else None

    def safe(a, b):
        return (a / b) if b and b != 0 else None

    # DSRI — Days Sales in Receivables Index
    dsr0 = safe(v(rec, y0), v(rev, y0))
    dsr1 = safe(v(rec, y1), v(rev, y1))
    DSRI = safe(dsr1, dsr0) if dsr0 else None

    # GMI — Gross Margin Index
    gm0  = safe(v(gp, y0), v(rev, y0))
    gm1  = safe(v(gp, y1), v(rev, y1))
    GMI  = safe(gm0, gm1) if gm1 else None

    # AQI — Asset Quality Index (non-current assets excl PPE / total assets)
    # Simplified: (total_assets - current_assets) / total_assets
    def nca_ratio(y):
        a = v(assets, y)
        ca_s = d.get("current_assets", pd.DataFrame())
        ca_v = ca_s.set_index("year")["value"][y] if not ca_s.empty and y in ca_s.set_index("year").index else None
        return safe(a - ca_v, a) if ca_v is not None else None

    aqr0 = nca_ratio(y0)
    aqr1 = nca_ratio(y1)
    AQI  = safe(aqr1, aqr0) if aqr0 else 1.0

    # SGI — Sales Growth Index
    SGI  = safe(v(rev, y1), v(rev, y0))

    # DEPI — Depreciation Index (skip if no DA)
    DEPI = 1.0
    if da is not None:
        da0 = v(da, y0); da1 = v(da, y1)
        if da0 and da1 and assets is not None:
            a0 = v(assets, y0); a1 = v(assets, y1)
            r0 = safe(da0, (da0 + a0)) if a0 else None
            r1 = safe(da1, (da1 + a1)) if a1 else None
            DEPI = safe(r0, r1) if r0 and r1 else 1.0

    # SGAI — SGA Index
    SGAI = 1.0
    if sga is not None:
        sg0 = safe(v(sga, y0), v(rev, y0))
        sg1 = safe(v(sga, y1), v(rev, y1))
        SGAI = safe(sg1, sg0) if sg0 else 1.0

    # LVGI — Leverage Index
    LVGI = 1.0
    if liab is not None:
        lv0 = safe(v(liab, y0), v(assets, y0))
        lv1 = safe(v(liab, y1), v(assets, y1))
        LVGI = safe(lv1, lv0) if lv0 else 1.0

    # TATA — Total Accruals to Total Assets
    TATA = 0.0
    if ni is not None and cfo is not None:
        ni1  = v(ni,  y1)
        cfo1 = v(cfo, y1)
        a1   = v(assets, y1)
        if ni1 is not None and cfo1 is not None and a1:
            TATA = (ni1 - cfo1) / a1

    # Fill defaults
    if None in [DSRI, GMI, AQI, SGI]:
        return None

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
