"""
flags.py — Rule-based red flag detection engine.
Each flag carries a severity (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW) and a short explanation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class Flag:
    severity: str          # "HIGH" | "MEDIUM" | "LOW"
    category: str          # "Liquidity" | "Solvency" | "Profitability" | "Cash Quality" | "Fraud Signal"
    title: str
    detail: str
    emoji: str = field(init=False)

    def __post_init__(self):
        self.emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(self.severity, "⚪")

    def __str__(self):
        return f"{self.emoji} [{self.category}] {self.title}: {self.detail}"


def _last(series: pd.Series) -> float | None:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _trend(series: pd.Series, n: int = 3) -> float | None:
    """Simple linear slope over last n periods (normalised by first value)."""
    s = series.dropna().tail(n)
    if len(s) < 2:
        return None
    x = np.arange(len(s), dtype=float)
    slope = np.polyfit(x, s.values.astype(float), 1)[0]
    base  = abs(s.iloc[0]) or 1
    return slope / base   # normalised slope per year


def detect_flags(ratios: pd.DataFrame, altman: float | None,
                 beneish: float | None) -> List[Flag]:
    flags: List[Flag] = []

    def col(name):
        return ratios[name] if name in ratios.columns else pd.Series(dtype=float)

    # ──────────────────────────────────────────────
    # LIQUIDITY
    # ──────────────────────────────────────────────
    cr = col("current_ratio")
    cr_last = _last(cr)
    cr_trend = _trend(cr)

    if cr_last is not None:
        if cr_last < 1.0:
            flags.append(Flag("HIGH", "Liquidity",
                "Current Ratio below 1.0",
                f"Current ratio = {cr_last:.2f}. Company cannot cover short-term obligations "
                f"with current assets — imminent liquidity risk."))
        elif cr_last < 1.5:
            flags.append(Flag("MEDIUM", "Liquidity",
                "Deteriorating Current Ratio",
                f"Current ratio = {cr_last:.2f} (healthy threshold ≥ 1.5). "
                f"Tight short-term liquidity buffer."))

    if cr_trend is not None and cr_trend < -0.05:
        flags.append(Flag("MEDIUM", "Liquidity",
            "Current Ratio in multi-year decline",
            f"Trend slope: {cr_trend*100:.1f}%/yr. Liquidity is systematically eroding."))

    qr = _last(col("quick_ratio"))
    if qr is not None and qr < 0.7:
        flags.append(Flag("HIGH", "Liquidity",
            "Quick Ratio critically low",
            f"Quick ratio = {qr:.2f}. Without relying on inventory, "
            f"the company struggles to meet near-term debt."))

    # ──────────────────────────────────────────────
    # SOLVENCY
    # ──────────────────────────────────────────────
    de = col("debt_equity")
    de_last  = _last(de)
    de_trend = _trend(de)

    if de_last is not None:
        if de_last > 3.0:
            flags.append(Flag("HIGH", "Solvency",
                "Extreme Debt-to-Equity",
                f"D/E = {de_last:.2f}x. Highly leveraged; sensitive to rate hikes and earnings shocks."))
        elif de_last > 2.0:
            flags.append(Flag("MEDIUM", "Solvency",
                "Elevated Debt-to-Equity",
                f"D/E = {de_last:.2f}x. Above the 2.0x caution threshold."))

    if de_trend is not None and de_trend > 0.08:
        flags.append(Flag("MEDIUM", "Solvency",
            "Debt load accelerating",
            f"D/E growing at ~{de_trend*100:.1f}%/yr — leverage is rising faster than equity."))

    da_ratio = _last(col("debt_assets"))
    if da_ratio is not None and da_ratio > 0.7:
        flags.append(Flag("HIGH", "Solvency",
            "Debt-to-Assets above 70%",
            f"Debt/Assets = {da_ratio:.1%}. Over 70% of assets are funded by creditors."))

    ic = _last(col("interest_coverage"))
    if ic is not None:
        if ic < 1.5:
            flags.append(Flag("HIGH", "Solvency",
                "Interest Coverage below 1.5x",
                f"EBIT covers interest only {ic:.2f}x. A small earnings drop triggers default risk."))
        elif ic < 3.0:
            flags.append(Flag("MEDIUM", "Solvency",
                "Thin Interest Coverage",
                f"Interest coverage = {ic:.2f}x. Healthy minimum is ~3x."))

    # ──────────────────────────────────────────────
    # PROFITABILITY
    # ──────────────────────────────────────────────
    gm = col("gross_margin")
    gm_last  = _last(gm)
    gm_trend = _trend(gm)

    if gm_trend is not None and gm_trend < -0.02:
        flags.append(Flag("MEDIUM", "Profitability",
            "Gross Margin compression",
            f"Gross margin declining at {gm_trend*100:.1f} pp/yr — pricing power or cost pressure."))

    nm = _last(col("net_margin"))
    if nm is not None and nm < 0:
        flags.append(Flag("HIGH", "Profitability",
            "Negative Net Income",
            f"Net margin = {nm:.1%}. Company is burning capital."))

    roe = _last(col("roe"))
    if roe is not None and roe < 0:
        flags.append(Flag("HIGH", "Profitability",
            "Negative Return on Equity",
            f"ROE = {roe:.1%}. Shareholders' equity is being destroyed."))

    roa = _last(col("roa"))
    if roa is not None and roa < 0.02:
        flags.append(Flag("LOW", "Profitability",
            "Low Return on Assets",
            f"ROA = {roa:.1%} — assets are not generating meaningful profits."))

    # ──────────────────────────────────────────────
    # CASH QUALITY (most predictive of fraud)
    # ──────────────────────────────────────────────
    cfoni = col("cfo_ni_ratio")
    cfoni_last  = _last(cfoni)
    cfoni_trend = _trend(cfoni)

    if cfoni_last is not None and cfoni_last < 0.5:
        flags.append(Flag("HIGH", "Cash Quality",
            "CFO far below Net Income (accrual gap)",
            f"CFO/NI = {cfoni_last:.2f}. Revenue is not converting to cash — classic earnings quality warning. "
            f"Companies manipulating earnings via accruals exhibit exactly this pattern."))
    elif cfoni_last is not None and cfoni_last < 0.8:
        flags.append(Flag("MEDIUM", "Cash Quality",
            "CFO-to-Net-Income ratio weakening",
            f"CFO/NI = {cfoni_last:.2f}. Earnings quality is declining."))

    # Revenue growing, cash flow shrinking — THE canonical fraud signal
    rev_g = col("revenue_growth")
    cfo_g = col("cfo_growth")
    rev_last = _last(rev_g)
    cfo_last = _last(cfo_g)

    if rev_last is not None and cfo_last is not None:
        divergence = rev_last - cfo_last
        if rev_last > 0.05 and cfo_last < -0.05:
            flags.append(Flag("HIGH", "Cash Quality",
                "Revenue rising but Operating Cash Flow falling",
                f"Revenue grew {rev_last:.1%} YoY while CFO fell {abs(cfo_last):.1%}. "
                f"This divergence — revenue up, cash down — is the #1 signal in earnings manipulation cases "
                f"(WorldCom, Sunbeam, Lucent all showed this before collapse)."))
        elif divergence > 0.15:
            flags.append(Flag("MEDIUM", "Cash Quality",
                "Revenue-CFO growth divergence widening",
                f"Revenue growth ({rev_last:.1%}) outpacing CFO growth ({cfo_last:.1%}) by "
                f"{divergence:.1%}. Watch for aggressive revenue recognition."))

    cfo_rev = col("cfo_revenue")
    cfo_rev_trend = _trend(cfo_rev)
    if cfo_rev_trend is not None and cfo_rev_trend < -0.03:
        flags.append(Flag("MEDIUM", "Cash Quality",
            "Cash conversion deteriorating",
            f"CFO/Revenue ratio declining at {cfo_rev_trend*100:.1f} pp/yr."))

    # DSO spike
    dso = col("dso")
    dso_last  = _last(dso)
    dso_trend = _trend(dso)
    if dso_trend is not None and dso_trend > 0.05 and dso_last is not None and dso_last > 50:
        flags.append(Flag("MEDIUM", "Cash Quality",
            "Days Sales Outstanding rising",
            f"DSO = {dso_last:.0f} days and growing. Receivables piling up suggests "
            f"channel stuffing or aggressive booking of uncollected revenue."))

    # ──────────────────────────────────────────────
    # MODEL-BASED FRAUD SIGNALS
    # ──────────────────────────────────────────────
    if altman is not None:
        if altman < 1.81:
            flags.append(Flag("HIGH", "Altman Z-Score",
                f"Distress Zone (Z = {altman:.2f})",
                "Z-score below 1.81 places the company in the bankruptcy distress zone. "
                "Altman's original model showed ~72% accuracy in predicting bankruptcies 1–2 years out."))
        elif altman < 2.99:
            flags.append(Flag("MEDIUM", "Altman Z-Score",
                f"Grey Zone (Z = {altman:.2f})",
                "Z-score between 1.81–2.99 is the 'grey zone' — elevated risk but not certain distress."))

    if beneish is not None and beneish > -2.22:
        flags.append(Flag("HIGH", "Beneish M-Score",
            f"Likely Earnings Manipulator (M = {beneish:.2f})",
            f"M-score above -2.22 indicates probable financial statement manipulation. "
            f"Beneish's model caught Enron (M = -1.89) before auditors did. "
            f"This company scores {beneish:.2f}."))

    return sorted(flags, key=lambda f: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[f.severity])
