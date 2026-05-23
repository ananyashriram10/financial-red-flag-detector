"""
scoring.py — Composite financial health score (0–100) and per-category sub-scores.
Inspired by multi-factor quant models; no external training data required.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def _last(ratios: pd.DataFrame, col: str) -> float | None:
    if col not in ratios.columns:
        return None
    s = ratios[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _trend_ok(ratios: pd.DataFrame, col: str) -> bool | None:
    """True if last 3-yr slope is flat or improving."""
    if col not in ratios.columns:
        return None
    s = ratios[col].dropna().tail(3)
    if len(s) < 2:
        return None
    x = np.arange(len(s), dtype=float)
    slope = np.polyfit(x, s.values.astype(float), 1)[0]
    return slope >= 0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _score_liquidity(ratios: pd.DataFrame) -> float:
    """0–100: higher = better liquidity."""
    score = 50.0  # start neutral
    cr = _last(ratios, "current_ratio")
    if cr is not None:
        # CR 2.0 → 100; CR 1.0 → 50; CR 0.5 → 0
        score = _clamp((cr - 0.5) / 1.5 * 100, 0, 100)

    qr = _last(ratios, "quick_ratio")
    if qr is not None:
        qs = _clamp((qr - 0.3) / 1.0 * 100, 0, 100)
        score = 0.6 * score + 0.4 * qs   # blend

    trend = _trend_ok(ratios, "current_ratio")
    if trend is False:
        score *= 0.85   # penalise deteriorating trend

    return _clamp(score, 0, 100)


def _score_solvency(ratios: pd.DataFrame) -> float:
    score = 50.0
    de = _last(ratios, "debt_equity")
    if de is not None:
        # D/E 0 → 100; D/E 2 → 50; D/E 4+ → 0
        score = _clamp((4.0 - de) / 4.0 * 100, 0, 100)

    ic = _last(ratios, "interest_coverage")
    if ic is not None:
        ics = _clamp((ic - 1.0) / 7.0 * 100, 0, 100)
        score = 0.55 * score + 0.45 * ics

    da = _last(ratios, "debt_assets")
    if da is not None:
        das = _clamp((0.7 - da) / 0.7 * 100, 0, 100)
        score = 0.7 * score + 0.3 * das

    trend = _trend_ok(ratios, "debt_equity")
    if trend is False:
        score *= 0.88

    return _clamp(score, 0, 100)


def _score_profitability(ratios: pd.DataFrame) -> float:
    score = 40.0
    nm = _last(ratios, "net_margin")
    if nm is not None:
        # NM 20% → 100; 0% → 40; negative → scales down
        score = _clamp(nm * 300 + 40, 0, 100)

    gm = _last(ratios, "gross_margin")
    if gm is not None:
        gms = _clamp(gm * 200, 0, 100)
        score = 0.5 * score + 0.5 * gms

    roe = _last(ratios, "roe")
    if roe is not None:
        roes = _clamp(roe * 250 + 40, 0, 100)
        score = 0.6 * score + 0.4 * roes

    trend = _trend_ok(ratios, "gross_margin")
    if trend is False:
        score *= 0.90

    return _clamp(score, 0, 100)


def _score_cash_quality(ratios: pd.DataFrame) -> float:
    score = 50.0
    cfoni = _last(ratios, "cfo_ni_ratio")
    if cfoni is not None:
        # ≥ 1.2 = great; 0.6 = 50; 0 = 0
        score = _clamp(cfoni / 1.2 * 80 + 10, 0, 100)

    cfo_rev = _last(ratios, "cfo_revenue")
    if cfo_rev is not None:
        crs = _clamp(cfo_rev * 300 + 30, 0, 100)
        score = 0.55 * score + 0.45 * crs

    # Revenue–CFO divergence penalty
    rg = _last(ratios, "revenue_growth")
    cg = _last(ratios, "cfo_growth")
    if rg is not None and cg is not None:
        div = rg - cg
        if div > 0.20:
            score *= 0.75
        elif div > 0.10:
            score *= 0.90

    # DSO trend
    dso_trend = _trend_ok(ratios, "dso")
    if dso_trend is False:
        score *= 0.92

    return _clamp(score, 0, 100)


def _score_altman(z: Optional[float]) -> float:
    if z is None:
        return 50.0
    # Z 4+ → 100; Z 3 → 80; Z 1.81 → 50; Z 0 → 20; Z -2 → 0
    if z >= 3.0:
        return _clamp(80 + (z - 3.0) * 10, 0, 100)
    elif z >= 1.81:
        return _clamp(50 + (z - 1.81) / (3.0 - 1.81) * 30, 0, 100)
    else:
        return _clamp(max(0, z / 1.81 * 50), 0, 100)


def _score_beneish(m: Optional[float]) -> float:
    if m is None:
        return 60.0  # neutral / uncertain
    # M < -2.22 = honest; M > -2.22 = manipulator
    # Map: -4 → 100; -2.22 → 55; 0 → 0
    if m <= -2.22:
        return _clamp(55 + (-2.22 - m) / 1.78 * 45, 55, 100)
    else:
        return _clamp(55 - (m + 2.22) / 2.22 * 55, 0, 55)


def composite_score(ratios: pd.DataFrame,
                    altman: Optional[float],
                    beneish: Optional[float]) -> dict[str, float]:
    """
    Returns dict with individual category scores and overall composite (0–100).
    Weights reflect analyst priorities — cash quality and solvency carry most weight.
    """
    liq   = _score_liquidity(ratios)
    sol   = _score_solvency(ratios)
    prof  = _score_profitability(ratios)
    cash  = _score_cash_quality(ratios)
    alt   = _score_altman(altman)
    ben   = _score_beneish(beneish)

    weights = {
        "Liquidity":    0.15,
        "Solvency":     0.20,
        "Profitability":0.15,
        "Cash Quality": 0.25,
        "Altman Z":     0.15,
        "Beneish M":    0.10,
    }
    raw = {
        "Liquidity":    liq,
        "Solvency":     sol,
        "Profitability":prof,
        "Cash Quality": cash,
        "Altman Z":     alt,
        "Beneish M":    ben,
    }
    overall = sum(raw[k] * w for k, w in weights.items())
    raw["Overall"] = overall
    return {k: round(v, 1) for k, v in raw.items()}


def score_label(overall: float) -> tuple[str, str]:
    """Returns (label, color_class)."""
    if overall >= 75:
        return "Healthy", "green"
    elif overall >= 55:
        return "Caution", "amber"
    elif overall >= 35:
        return "At Risk", "orange"
    else:
        return "Critical", "red"
