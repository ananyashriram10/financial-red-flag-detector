"""
charts.py — Plotly chart builders for the Streamlit dashboard.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_PALETTE = {
    "blue":   "#2563EB",
    "red":    "#DC2626",
    "green":  "#16A34A",
    "amber":  "#D97706",
    "purple": "#7C3AED",
    "gray":   "#6B7280",
    "bg":     "#0F172A",
    "card":   "#1E293B",
    "border": "#334155",
    "text":   "#F1F5F9",
}

_LAYOUT = dict(
    paper_bgcolor=_PALETTE["bg"],
    plot_bgcolor=_PALETTE["card"],
    font=dict(color=_PALETTE["text"], family="Inter, sans-serif", size=13),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor=_PALETTE["border"], zerolinecolor=_PALETTE["border"]),
    yaxis=dict(gridcolor=_PALETTE["border"], zerolinecolor=_PALETTE["border"]),
)


def _base(**kwargs):
    fig = go.Figure()
    fig.update_layout(**{**_LAYOUT, **kwargs})
    return fig


# ─── Revenue vs CFO divergence ────────────────────────────────────────────────
def revenue_vs_cfo(ratios: pd.DataFrame) -> go.Figure:
    fig = _base(title="Revenue vs. Operating Cash Flow (normalized)", height=360)
    for col, name, color in [
        ("revenue", "Revenue", _PALETTE["blue"]),
        ("cfo",     "Operating CFO", _PALETTE["green"]),
        ("net_income", "Net Income", _PALETTE["amber"]),
    ]:
        if col not in ratios.columns:
            continue
        s = ratios[col].dropna()
        if s.empty:
            continue
        base = s.iloc[0] or 1
        fig.add_trace(go.Scatter(
            x=s.index.astype(str), y=(s / abs(base)).values,
            name=name, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=_PALETTE["border"])
    fig.update_layout(yaxis_title="Indexed (base = 1.0)", legend=dict(
        bgcolor=_PALETTE["card"], bordercolor=_PALETTE["border"]))
    return fig


# ─── Key ratios over time ─────────────────────────────────────────────────────
def ratios_trend(ratios: pd.DataFrame,
                 metrics: list[tuple[str, str, str]]) -> go.Figure:
    """metrics: list of (column, display_name, color)"""
    rows = (len(metrics) + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=[m[1] for m in metrics],
                        vertical_spacing=0.12)
    for i, (col, name, color) in enumerate(metrics):
        row, c = divmod(i, 2)
        if col not in ratios.columns:
            continue
        s = ratios[col].dropna()
        fig.add_trace(go.Scatter(
            x=s.index.astype(str), y=s.values,
            name=name, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            showlegend=False,
        ), row=row+1, col=c+1)
    fig.update_layout(**{**_LAYOUT, "height": rows * 200,
                         "title": "Key Ratio Trends"})
    fig.update_xaxes(gridcolor=_PALETTE["border"])
    fig.update_yaxes(gridcolor=_PALETTE["border"])
    return fig


# ─── Altman Z-Score gauge ─────────────────────────────────────────────────────
def altman_gauge(z: float | None) -> go.Figure:
    if z is None:
        fig = _base(title="Altman Z-Score", height=260)
        fig.add_annotation(text="Insufficient data", showarrow=False,
                           font=dict(size=18, color=_PALETTE["gray"]),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    z_clipped = max(-2, min(z, 6))
    if z < 1.81:
        color = _PALETTE["red"]; zone = "Distress Zone"
    elif z < 2.99:
        color = _PALETTE["amber"]; zone = "Grey Zone"
    else:
        color = _PALETTE["green"]; zone = "Safe Zone"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=z_clipped,
        number={"suffix": "", "font": {"color": color, "size": 36}},
        title={"text": f"Altman Z-Score — {zone}", "font": {"color": _PALETTE["text"]}},
        gauge={
            "axis": {"range": [-2, 6], "tickcolor": _PALETTE["text"],
                     "tickfont": {"color": _PALETTE["text"]}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": _PALETTE["card"],
            "bordercolor": _PALETTE["border"],
            "steps": [
                {"range": [-2, 1.81], "color": "#7F1D1D"},
                {"range": [1.81, 2.99], "color": "#78350F"},
                {"range": [2.99, 6],   "color": "#14532D"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": z_clipped,
            },
        },
    ))
    fig.update_layout(**{**_LAYOUT, "height": 300})
    return fig


# ─── Beneish M-Score gauge ────────────────────────────────────────────────────
def beneish_gauge(m: float | None) -> go.Figure:
    if m is None:
        fig = _base(title="Beneish M-Score", height=260)
        fig.add_annotation(text="Insufficient data (need ≥2 years)",
                           showarrow=False, font=dict(size=16, color=_PALETTE["gray"]),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    m_clipped = max(-5, min(m, 1))
    if m > -2.22:
        color = _PALETTE["red"]; label = "⚠️ Likely Manipulator"
    else:
        color = _PALETTE["green"]; label = "✅ Non-Manipulator"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=m_clipped,
        number={"font": {"color": color, "size": 36}},
        title={"text": f"Beneish M-Score — {label}",
               "font": {"color": _PALETTE["text"]}},
        gauge={
            "axis": {"range": [-5, 1], "tickcolor": _PALETTE["text"],
                     "tickfont": {"color": _PALETTE["text"]}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": _PALETTE["card"],
            "bordercolor": _PALETTE["border"],
            "steps": [
                {"range": [-5, -2.22], "color": "#14532D"},
                {"range": [-2.22, 1],  "color": "#7F1D1D"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": -2.22,
            },
        },
    ))
    fig.update_layout(**{**_LAYOUT, "height": 300})
    return fig


# ─── Composite health radar ───────────────────────────────────────────────────
def health_radar(scores: dict[str, float]) -> go.Figure:
    categories = list(scores.keys()) + [list(scores.keys())[0]]
    values     = list(scores.values()) + [list(scores.values())[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        line_color=_PALETTE["blue"],
        fillcolor="rgba(37,99,235,0.25)",
        marker=dict(color=_PALETTE["blue"]),
    ))
    fig.update_layout(**{**_LAYOUT,
        "polar": {
            "radialaxis": {"visible": True, "range": [0, 100],
                           "gridcolor": _PALETTE["border"],
                           "tickfont": {"color": _PALETTE["text"]}},
            "angularaxis": {"gridcolor": _PALETTE["border"],
                            "tickfont": {"color": _PALETTE["text"]}},
            "bgcolor": _PALETTE["card"],
        },
        "height": 380,
        "title": "Financial Health Radar (0 = worst, 100 = best)",
    })
    return fig


# ─── Waterfall: FCF decomposition ─────────────────────────────────────────────
def fcf_waterfall(ratios: pd.DataFrame, raw: dict) -> go.Figure:
    fig = _base(title="Free Cash Flow Decomposition (latest year)", height=340)
    labels, values, colors = [], [], []

    def lat(key):
        df = raw.get(key, pd.DataFrame())
        return float(df["value"].iloc[-1]) if not df.empty else 0.0

    cfo   = lat("cfo")
    capex = lat("capex")
    fcf   = cfo - abs(capex)

    for name, val in [("Operating CFO", cfo), ("CapEx", -abs(capex)), ("FCF", fcf)]:
        labels.append(name)
        values.append(val / 1e9)
        colors.append(_PALETTE["green"] if val >= 0 else _PALETTE["red"])

    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"${v:.2f}B" for v in values],
        textposition="outside",
        textfont=dict(color=_PALETTE["text"]),
    ))
    fig.update_layout(yaxis_title="USD Billions", showlegend=False)
    return fig


# ─── Debt maturity snapshot ───────────────────────────────────────────────────
def debt_equity_bar(ratios: pd.DataFrame) -> go.Figure:
    fig = _base(title="Debt / Equity over Time", height=320)
    if "debt_equity" not in ratios.columns:
        return fig
    s = ratios["debt_equity"].dropna()
    colors = [_PALETTE["red"] if v > 2 else
              _PALETTE["amber"] if v > 1 else
              _PALETTE["green"] for v in s.values]
    fig.add_trace(go.Bar(x=s.index.astype(str), y=s.values,
                         marker_color=colors,
                         text=[f"{v:.2f}x" for v in s.values],
                         textposition="outside",
                         textfont=dict(color=_PALETTE["text"])))
    fig.add_hline(y=2.0, line_dash="dot", line_color=_PALETTE["amber"],
                  annotation_text="Caution: 2.0x", annotation_position="top right")
    fig.update_layout(yaxis_title="Debt/Equity (x)", showlegend=False)
    return fig
