"""
app.py — Financial Statement Red Flag Detector
Streamlit dashboard: type a ticker, get a full health report.

Run:  streamlit run app.py
"""

from __future__ import annotations
import time
import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Red Flag Detector",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS (dark theme matching our chart palette) ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0F172A; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* metric cards */
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.6rem;
}
.metric-card .label { color: #94A3B8; font-size: 0.78rem; font-weight: 500;
                       text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { color: #F1F5F9; font-size: 1.6rem; font-weight: 700; margin-top: 2px; }
.metric-card .delta { font-size: 0.82rem; margin-top: 2px; }

/* flag cards */
.flag-high   { background: #450A0A; border-left: 4px solid #DC2626; border-radius: 8px;
               padding: 0.9rem 1.1rem; margin: 0.4rem 0; }
.flag-medium { background: #431407; border-left: 4px solid #D97706; border-radius: 8px;
               padding: 0.9rem 1.1rem; margin: 0.4rem 0; }
.flag-low    { background: #052E16; border-left: 4px solid #16A34A; border-radius: 8px;
               padding: 0.9rem 1.1rem; margin: 0.4rem 0; }
.flag-title  { color: #F1F5F9; font-weight: 600; font-size: 0.95rem; }
.flag-detail { color: #CBD5E1; font-size: 0.85rem; margin-top: 4px; line-height: 1.5; }

/* score badge */
.score-badge {
    display: inline-block;
    padding: 0.4rem 1.1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
}
.score-green  { background: #14532D; color: #86EFAC; }
.score-amber  { background: #78350F; color: #FCD34D; }
.score-orange { background: #7C2D12; color: #FDBA74; }
.score-red    { background: #450A0A; color: #FCA5A5; }

/* section headers */
h2.section { color: #94A3B8; font-size: 0.72rem; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.1em;
             border-bottom: 1px solid #334155; padding-bottom: 6px; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Local imports (after page config) ─────────────────────────────────────────
import edgar
import metrics as mtx
import flags as flg
import scoring as sc
import charts as ch


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str):
    """Fetch EDGAR data (cached 1 hour)."""
    return edgar.get_financial_data(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_cap(ticker: str) -> float | None:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        return float(info.market_cap)
    except Exception:
        return None


def fmt_currency(v: float | None) -> str:
    if v is None or np.isnan(v):
        return "N/A"
    av = abs(v)
    sign = "-" if v < 0 else ""
    if av >= 1e12:
        return f"{sign}${av/1e12:.2f}T"
    elif av >= 1e9:
        return f"{sign}${av/1e9:.2f}B"
    elif av >= 1e6:
        return f"{sign}${av/1e6:.1f}M"
    return f"{sign}${av:,.0f}"


def fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.1%}"


def fmt_x(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}x"


def latest_val(ratios: pd.DataFrame, col: str) -> float | None:
    if col not in ratios.columns:
        return None
    s = ratios[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def delta_str(ratios: pd.DataFrame, col: str) -> str:
    if col not in ratios.columns:
        return ""
    s = ratios[col].dropna()
    if len(s) < 2:
        return ""
    d = s.iloc[-1] - s.iloc[-2]
    arrow = "▲" if d > 0 else "▼"
    pct   = abs(d / s.iloc[-2]) if s.iloc[-2] != 0 else 0
    return f"{arrow} {pct:.1%} YoY"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:1rem;">
  <div style="font-size:2.4rem;">🚨</div>
  <div>
    <div style="color:#F1F5F9; font-size:1.6rem; font-weight:700; line-height:1.1;">
      Financial Red Flag Detector</div>
    <div style="color:#64748B; font-size:0.85rem;">
      SEC EDGAR • Altman Z-Score • Beneish M-Score • Accrual Analysis
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Ticker Input ──────────────────────────────────────────────────────────────
col_in, col_btn, col_ex = st.columns([2, 0.7, 3])
with col_in:
    ticker_raw = st.text_input("", placeholder="Enter ticker (e.g. AAPL, TSLA, GE)…",
                               label_visibility="collapsed")
with col_btn:
    analyze = st.button("Analyze →", type="primary", use_container_width=True)
with col_ex:
    st.markdown(
        "<div style='color:#475569; font-size:0.8rem; padding-top:10px;'>"
        "Pulls live 10-K data from SEC EDGAR · Any US-listed public company · Free</div>",
        unsafe_allow_html=True)

ticker = ticker_raw.strip().upper()

# Example quick-launch chips
st.markdown("<div style='margin-bottom:0.5rem;'>", unsafe_allow_html=True)
example_cols = st.columns(8)
examples = ["AAPL", "TSLA", "GE", "NFLX", "SBUX", "WBD", "SVB", "ENRN"]
chosen = None
for i, ex in enumerate(examples):
    with example_cols[i]:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            chosen = ex
            ticker = ex
st.markdown("</div>", unsafe_allow_html=True)

if chosen:
    analyze = True

# ── Main Analysis ─────────────────────────────────────────────────────────────
if not ticker or not analyze:
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem; color:#475569;'>
      <div style='font-size:3rem; margin-bottom:1rem;'>📊</div>
      <div style='font-size:1.1rem; color:#64748B;'>
        Enter a ticker symbol and click <b>Analyze</b> to detect financial red flags.
      </div>
      <div style='margin-top:0.5rem; font-size:0.82rem;'>
        Data sourced directly from SEC EDGAR filings. No API key required.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Data loading ──────────────────────────────────────────────────────────────
with st.spinner(f"Fetching SEC EDGAR filings for **{ticker}**…"):
    try:
        t0 = time.time()
        data   = load_data(ticker)
        mkt_cap = get_market_cap(ticker)
        ratios = mtx.compute_ratios(data)
        z_score = mtx.altman_z(data, mkt_cap)
        m_score = mtx.beneish_m(data)
        all_flags = flg.detect_flags(ratios, z_score, m_score)
        scores  = sc.composite_score(ratios, z_score, m_score)
        elapsed = time.time() - t0
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        st.stop()

company_name = data.get("company_name", ticker)
overall = scores["Overall"]
label, color_key = sc.score_label(overall)

# ── Company Header ─────────────────────────────────────────────────────────────
high_ct   = sum(1 for f in all_flags if f.severity == "HIGH")
medium_ct = sum(1 for f in all_flags if f.severity == "MEDIUM")

st.markdown(f"""
<div style='background:#1E293B; border:1px solid #334155; border-radius:14px;
            padding:1.3rem 1.8rem; margin-bottom:1.2rem;
            display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;'>
  <div>
    <div style='color:#F1F5F9; font-size:1.45rem; font-weight:700;'>{company_name}</div>
    <div style='color:#64748B; font-size:0.88rem; margin-top:3px;'>
      {ticker} · SEC EDGAR · {elapsed:.1f}s load
      {"· Market Cap " + fmt_currency(mkt_cap) if mkt_cap else ""}
    </div>
  </div>
  <div style='display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;'>
    <div>
      <div style='color:#94A3B8; font-size:0.72rem; text-transform:uppercase; letter-spacing:.05em;'>Health Score</div>
      <span class='score-badge score-{color_key}' style='font-size:1.3rem;'>{overall:.0f}/100 — {label}</span>
    </div>
    <div style='display:flex; gap:0.5rem;'>
      {"<span style='background:#450A0A;color:#FCA5A5;padding:4px 12px;border-radius:999px;font-size:0.8rem;font-weight:600;'>" + str(high_ct) + " HIGH</span>" if high_ct else ""}
      {"<span style='background:#431407;color:#FCD34D;padding:4px 12px;border-radius:999px;font-size:0.8rem;font-weight:600;'>" + str(medium_ct) + " MEDIUM</span>" if medium_ct else ""}
      {"<span style='background:#14532D;color:#86EFAC;padding:4px 12px;border-radius:999px;font-size:0.8rem;font-weight:600;'>✅ CLEAN</span>" if not high_ct and not medium_ct else ""}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Key Metrics Row ────────────────────────────────────────────────────────────
st.markdown("<h2 class='section'>Key Financial Metrics (Most Recent Year)</h2>",
            unsafe_allow_html=True)

m_cols = st.columns(6)
metrics_display = [
    ("Current Ratio",    fmt_x(latest_val(ratios, "current_ratio")),   delta_str(ratios, "current_ratio")),
    ("Debt / Equity",    fmt_x(latest_val(ratios, "debt_equity")),      delta_str(ratios, "debt_equity")),
    ("Net Margin",       fmt_pct(latest_val(ratios, "net_margin")),     delta_str(ratios, "net_margin")),
    ("CFO / Net Income", fmt_x(latest_val(ratios, "cfo_ni_ratio")),    delta_str(ratios, "cfo_ni_ratio")),
    ("Gross Margin",     fmt_pct(latest_val(ratios, "gross_margin")),   delta_str(ratios, "gross_margin")),
    ("Interest Coverage",fmt_x(latest_val(ratios, "interest_coverage")),delta_str(ratios, "interest_coverage")),
]
for i, (label_m, val, dlt) in enumerate(metrics_display):
    with m_cols[i]:
        delta_color = "#86EFAC" if "▲" in dlt else "#FCA5A5" if "▼" in dlt else "#94A3B8"
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>{label_m}</div>
          <div class='value'>{val}</div>
          <div class='delta' style='color:{delta_color};'>{dlt or "—"}</div>
        </div>""", unsafe_allow_html=True)

# ── Charts: Revenue vs CFO + Debt/Equity ──────────────────────────────────────
st.markdown("<h2 class='section'>Trend Analysis</h2>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(ch.revenue_vs_cfo(ratios), use_container_width=True)
with c2:
    st.plotly_chart(ch.debt_equity_bar(ratios), use_container_width=True)

# ── More Charts: Ratio Grid + FCF ─────────────────────────────────────────────
c3, c4 = st.columns([3, 2])
with c3:
    st.plotly_chart(ch.ratios_trend(ratios, [
        ("current_ratio",    "Current Ratio",    "#2563EB"),
        ("net_margin",       "Net Margin",       "#16A34A"),
        ("gross_margin",     "Gross Margin",     "#7C3AED"),
        ("cfo_revenue",      "CFO/Revenue",      "#D97706"),
        ("roe",              "Return on Equity", "#EC4899"),
        ("interest_coverage","Interest Coverage","#06B6D4"),
    ]), use_container_width=True)
with c4:
    st.plotly_chart(ch.fcf_waterfall(ratios, data), use_container_width=True)

# ── Model Scores ──────────────────────────────────────────────────────────────
st.markdown("<h2 class='section'>Fraud & Distress Models</h2>", unsafe_allow_html=True)
mg1, mg2, mg3 = st.columns(3)
with mg1:
    st.plotly_chart(ch.altman_gauge(z_score), use_container_width=True)
    if z_score is not None:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid #334155;border-radius:8px;
                    padding:.8rem 1rem;font-size:.8rem;color:#94A3B8;'>
        <b style='color:#F1F5F9;'>Altman Z-Score guide:</b><br>
        🟢 Z &gt; 2.99 — Safe Zone<br>
        🟡 1.81 &lt; Z &lt; 2.99 — Grey Zone<br>
        🔴 Z &lt; 1.81 — Distress Zone<br>
        <br>Original model achieved <b>72–80% accuracy</b> predicting bankruptcies 1–2 yrs ahead.
        </div>""", unsafe_allow_html=True)

with mg2:
    st.plotly_chart(ch.beneish_gauge(m_score), use_container_width=True)
    if m_score is not None:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid #334155;border-radius:8px;
                    padding:.8rem 1rem;font-size:.8rem;color:#94A3B8;'>
        <b style='color:#F1F5F9;'>Beneish M-Score guide:</b><br>
        🟢 M &lt; -2.22 — Non-manipulator<br>
        🔴 M &gt; -2.22 — Likely manipulator<br>
        <br>Caught <b>Enron</b> (M = -1.89) before auditors.
        Trained on SEC enforcement actions.
        </div>""", unsafe_allow_html=True)

with mg3:
    radar_scores = {k: v for k, v in scores.items() if k != "Overall"}
    st.plotly_chart(ch.health_radar(radar_scores), use_container_width=True)

# ── Red Flags Panel ────────────────────────────────────────────────────────────
st.markdown("<h2 class='section'>🚨 Detected Red Flags</h2>", unsafe_allow_html=True)

if not all_flags:
    st.markdown("""
    <div style='background:#052E16;border:1px solid #166534;border-radius:10px;
                padding:1.2rem 1.5rem;color:#86EFAC;font-size:0.95rem;'>
      ✅ <b>No significant red flags detected.</b> This company's financials look clean across all
      monitored dimensions. Continue monitoring for future changes.
    </div>""", unsafe_allow_html=True)
else:
    tabs = st.tabs(["All Flags",
                    f"🔴 HIGH ({high_ct})",
                    f"🟡 MEDIUM ({medium_ct})",
                    "By Category"])

    def render_flags(flags):
        if not flags:
            st.markdown("<div style='color:#475569;padding:1rem;'>No flags in this group.</div>",
                        unsafe_allow_html=True)
            return
        for f in flags:
            cls = f"flag-{f.severity.lower()}"
            st.markdown(f"""
            <div class='{cls}'>
              <div class='flag-title'>{f.emoji} [{f.category}] {f.title}</div>
              <div class='flag-detail'>{f.detail}</div>
            </div>""", unsafe_allow_html=True)

    with tabs[0]:
        render_flags(all_flags)
    with tabs[1]:
        render_flags([f for f in all_flags if f.severity == "HIGH"])
    with tabs[2]:
        render_flags([f for f in all_flags if f.severity == "MEDIUM"])
    with tabs[3]:
        categories = sorted(set(f.category for f in all_flags))
        for cat in categories:
            cat_flags = [f for f in all_flags if f.category == cat]
            with st.expander(f"{cat} ({len(cat_flags)} flags)", expanded=True):
                render_flags(cat_flags)

# ── Score Breakdown Table ──────────────────────────────────────────────────────
st.markdown("<h2 class='section'>Score Breakdown</h2>", unsafe_allow_html=True)

sc_cols = st.columns(len(scores))
for i, (dim, val) in enumerate(scores.items()):
    if dim == "Overall":
        continue
    _, clr = sc.score_label(val)
    with sc_cols[i]:
        st.markdown(f"""
        <div class='metric-card' style='text-align:center;'>
          <div class='label'>{dim}</div>
          <div class='value'><span class='score-badge score-{clr}'>{val:.0f}</span></div>
        </div>""", unsafe_allow_html=True)

# ── Raw Data Expander ──────────────────────────────────────────────────────────
with st.expander("📋 Raw Ratio Table (all years)"):
    if not ratios.empty:
        display_df = ratios.copy()
        # Format percentage columns
        pct_cols = [c for c in display_df.columns
                    if any(k in c for k in ["margin", "growth", "ratio", "roa", "roe"])]
        for c in pct_cols:
            if c in display_df.columns:
                display_df[c] = display_df[c].map(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        st.dataframe(display_df.style.set_properties(**{
            "background-color": "#1E293B",
            "color": "#F1F5F9",
            "border-color": "#334155",
        }), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid #1E293B;margin:2rem 0 1rem;'>
<div style='color:#334155;font-size:0.75rem;text-align:center;'>
  Data sourced from SEC EDGAR (free, no API key required) · Altman (1968) · Beneish (1999) ·
  Not financial advice · For research and educational use only
</div>
""", unsafe_allow_html=True)
