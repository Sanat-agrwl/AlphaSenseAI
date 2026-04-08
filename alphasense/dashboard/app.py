"""
AlphaSense AI — Streamlit Dashboard
=====================================
5 tabs:
  1. Equity Curve    — portfolio vs NIFTY 500, drawdown chart
  2. Live Signals    — recent BUY/SELL with sentiment/Z-score/PnL
  3. Universe        — quality score rankings, sector distribution
  4. Earnings Calls  — management confidence scores, QoQ deltas
  5. Risk Monitor    — open positions, VIX, sector limits

Loads real data from data/ directory where available;
falls back to demo data for any missing piece.

Run:
    streamlit run alphasense/dashboard/app.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

# ─── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaSense AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body { background: #0a0a0f; }
.stMetric { background: #12121a; border-radius: 10px; border: 1px solid #2a2a3d; padding: 12px; }
h1, h2 { color: #00d97e; }
.stTabs [data-baseweb="tab"] { background: #1e2530; border-radius: 8px; padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)


# ─── Data loaders (real → demo fallback) ─────────────────────────────────────

@st.cache_data(ttl=300)
def _load_universe():
    p = cfg.data_dir / "universe.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame({
        "symbol":        ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                          "PERSISTENT", "CDSL", "POLYCAB", "ASTRAL", "COFORGE"],
        "quality_score": [85, 82, 79, 78, 76, 83, 78, 88, 71, 80],
        "sector":        ["Energy", "IT", "IT", "Banking", "Banking",
                          "IT", "Finance", "Electricals", "Pipes", "IT"],
    })


@st.cache_data(ttl=300)
def _load_backtest():
    p = cfg.results_dir / "test_results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"total_return_pct": 42.5, "annual_return_pct": 18.2,
            "sharpe": 1.45, "max_drawdown_pct": -12.3,
            "n_trades": 156, "win_rate": 58.3,
            "avg_pnl_pct": 3.2, "profit_factor": 1.85}


@st.cache_data(ttl=60)
def _load_signals():
    p = cfg.data_dir / "paper_state.json"
    if p.exists():
        try:
            state  = json.loads(p.read_text())
            trades = state.get("trades", [])
            if trades:
                df = pd.DataFrame(trades)
                df["date"] = pd.to_datetime(df["entry_date"])
                df["direction"] = "SELL"
                df["sentiment"] = np.nan
                df["zscore"]    = np.nan
                return df.tail(20)
        except Exception:
            pass
    # Demo
    return pd.DataFrame({
        "date":       pd.date_range(end=datetime.now(), periods=10, freq="B"),
        "symbol":     ["PERSISTENT","CDSL","DEEPAKNTR","POLYCAB","AARTI",
                       "ASTRAL","MPHASIS","COFORGE","ATUL","KPITTECH"],
        "direction":  ["BUY","BUY","SELL","BUY","SELL","BUY","BUY","SELL","BUY","BUY"],
        "sentiment":  [-0.62,-0.55,0.15,-0.48,0.22,-0.71,-0.45,0.18,-0.58,-0.52],
        "zscore":     [-2.8,-2.3,-0.8,-2.5,-0.5,-3.1,-2.1,-0.6,-2.6,-2.4],
        "quality":    [82,78,75,88,72,85,77,80,79,83],
        "entry_price":[4850,1620,2380,6200,580,1950,2700,5800,6100,1420],
        "pnl_pct":    [None,None,4.2,None,-1.8,None,None,6.1,None,None],
    })


@st.cache_data(ttl=300)
def _load_earnings():
    p = cfg.transcripts_dir / "text"
    records = []
    if p.exists():
        for f in sorted(p.glob("*_scores.json"))[-20:]:
            try:
                d = json.loads(f.read_text())
                records.append(d)
            except Exception:
                pass
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame({
        "symbol":          ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
                            "PERSISTENT","CDSL","POLYCAB","ASTRAL","COFORGE"],
        "quarter":         ["Q3FY25"]*10,
        "confidence_score":[78,72,65,81,74,45,68,82,71,58],
        "prev_confidence": [75,74,71,78,72,72,65,79,73,70],
    })


def _equity_curve():
    p = cfg.results_dir / "test_results.json"
    if p.exists():
        try:
            d      = json.loads(p.read_text())
            eq     = d.get("equity_curve", [])
            if eq:
                df = pd.DataFrame(eq, columns=["date","capital"])
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception:
            pass
    # Demo
    dates   = pd.date_range("2022-01-01", periods=750, freq="B")
    np.random.seed(42)
    rets    = np.random.normal(0.0007, 0.012, len(dates))
    rets[100:130] -= 0.005
    rets[400:420] -= 0.008
    capital = cfg.backtest.capital * np.cumprod(1 + rets)
    bench_r = np.random.normal(0.0004, 0.011, len(dates))
    bench_r[100:130] -= 0.008
    bench   = cfg.backtest.capital * np.cumprod(1 + bench_r)
    return pd.DataFrame({"date": dates, "capital": capital, "benchmark": bench})


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Settings")
    sent_thr = st.slider("Sentiment threshold", -1.0, 0.0,
                         cfg.signal.sentiment_threshold, 0.05)
    z_thr    = st.slider("Z-score threshold",  -4.0,-1.0,
                         cfg.signal.zscore_threshold, 0.1)
    vix_max  = st.slider("Max India VIX",       15.0,40.0,
                         cfg.signal.vix_max, 1.0)
    st.markdown("---")
    universe = _load_universe()
    st.markdown(f"**Universe:** {len(universe)} stocks")
    st.markdown(f"**Updated:** {datetime.now().strftime('%H:%M IST')}")
    st.markdown(f"**Mode:** {cfg.kite.mode.upper()}")


# ─── Header ──────────────────────────────────────────────────────────────────

st.title("AlphaSense AI")
st.caption("Sentiment-driven quantitative intelligence for Indian markets")

results = _load_backtest()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Annual Return",  f"{results['annual_return_pct']:.1f}%",  "vs ~12% NIFTY 500")
c2.metric("Sharpe Ratio",   f"{results['sharpe']:.2f}",              "Target > 1.2")
c3.metric("Max Drawdown",   f"{results['max_drawdown_pct']:.1f}%",   "Limit −18%")
c4.metric("Win Rate",       f"{results['win_rate']:.1f}%",
          f"{results['n_trades']} trades")
c5.metric("Profit Factor",  f"{results['profit_factor']:.2f}", "")

st.markdown("---")


# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Equity Curve", "🚨 Live Signals",
    "🏛 Universe",     "🎙 Earnings Calls", "⚠ Risk Monitor",
])

# ── Tab 1: Equity Curve ───────────────────────────────────────────────────────
with tab1:
    eq = _equity_curve()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["capital"],
                             name="AlphaSense", line=dict(color="#00d97e", width=2),
                             fill="tozeroy", fillcolor="rgba(0,217,126,0.08)"))
    if "benchmark" in eq.columns:
        fig.add_trace(go.Scatter(x=eq["date"], y=eq["benchmark"],
                                 name="NIFTY 500", line=dict(color="#6c757d", width=1.5, dash="dot")))
    fig.update_layout(title="Portfolio Equity Curve (Out-of-Sample: 2022–2024)",
                      template="plotly_dark", height=480,
                      yaxis_title="₹ Portfolio Value",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    eq["peak"] = eq["capital"].cummax()
    eq["dd"]   = (eq["capital"] - eq["peak"]) / eq["peak"] * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=eq["date"], y=eq["dd"],
                                fill="tozeroy", fillcolor="rgba(230,55,87,0.3)",
                                line=dict(color="#e63757", width=1), name="Drawdown"))
    fig_dd.update_layout(title="Drawdown (%)", template="plotly_dark",
                         height=220, showlegend=False)
    st.plotly_chart(fig_dd, use_container_width=True)


# ── Tab 2: Live Signals ───────────────────────────────────────────────────────
with tab2:
    signals = _load_signals()
    st.subheader("Recent Signals")

    def _color_dir(v):
        return "color:#00d97e;font-weight:bold" if v == "BUY" \
               else "color:#e63757;font-weight:bold" if v == "SELL" else ""

    def _color_pnl(v):
        if pd.isna(v): return ""
        return "color:#00d97e" if v > 0 else "color:#e63757"

    styled = signals.style \
        .applymap(_color_dir, subset=["direction"]) \
        .format({
            "sentiment":   lambda x: f"{x:.2f}" if pd.notna(x) else "—",
            "zscore":      lambda x: f"{x:.1f}"  if pd.notna(x) else "—",
            "entry_price": lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—",
            "pnl_pct":     lambda x: f"{x:.1f}%" if pd.notna(x) else "Open",
        })
    if "pnl_pct" in signals.columns:
        styled = styled.applymap(_color_pnl, subset=["pnl_pct"])
    st.dataframe(styled, use_container_width=True, height=380)

    if "sentiment" in signals.columns and signals["sentiment"].notna().any():
        col1, col2 = st.columns(2)
        with col1:
            fig_s = px.histogram(signals.dropna(subset=["sentiment"]),
                                 x="sentiment", nbins=15, title="Sentiment at Entry",
                                 color_discrete_sequence=["#00d97e"])
            fig_s.add_vline(x=sent_thr, line_dash="dash", line_color="red")
            fig_s.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_s, use_container_width=True)
        with col2:
            fig_z = px.histogram(signals.dropna(subset=["zscore"]),
                                 x="zscore", nbins=15, title="Z-Score at Entry",
                                 color_discrete_sequence=["#4dabf7"])
            fig_z.add_vline(x=z_thr, line_dash="dash", line_color="red")
            fig_z.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_z, use_container_width=True)


# ── Tab 3: Universe ───────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"Quality Universe — {len(universe)} stocks")
    col1, col2 = st.columns([2, 1])
    with col1:
        top30 = universe.head(30).sort_values("quality_score")
        fig_u = px.bar(top30, x="quality_score", y="symbol", orientation="h",
                       color="quality_score", color_continuous_scale="Viridis",
                       title="Top 30 by Quality Score")
        fig_u.update_layout(template="plotly_dark", height=580,
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_u, use_container_width=True)
    with col2:
        st.markdown("""
**Tier 1 (Hard Eliminators)**
- ROE > 15% for 3+ years
- Debt/Equity < 1.0
- Promoter holding > 30%
- No qualified audit
- No SEBI action (24 mo)

**Tier 2 (Quality Score)**
- ROCE trend (25%)
- FCF conversion (30%)
- Accruals ratio (25%)
- Revenue concentration (20%)
""")
        if "sector" in universe.columns:
            fig_s = px.pie(universe, names="sector", title="Sector Split",
                           color_discrete_sequence=px.colors.qualitative.Dark24)
            fig_s.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig_s, use_container_width=True)


# ── Tab 4: Earnings Calls ─────────────────────────────────────────────────────
with tab4:
    earnings = _load_earnings()
    if "confidence_score" not in earnings.columns:
        st.info("No earnings scores yet. Run: python scripts/score_transcripts.py")
    else:
        st.subheader("Management Confidence Scores")

        if "prev_confidence" in earnings.columns:
            earnings["delta"] = earnings["confidence_score"] - earnings["prev_confidence"]
            anomalies = earnings[earnings["delta"] < -10]
            if not anomalies.empty:
                st.error(f"⚠ {len(anomalies)} stock(s) showing significant confidence drops!")
                for _, r in anomalies.iterrows():
                    st.markdown(f"**{r['symbol']}**: {r.get('prev_confidence','?')} → "
                                f"{r['confidence_score']} (Δ = {r['delta']:.0f})")

        col1, col2 = st.columns(2)
        with col1:
            fig_c = go.Figure()
            colors = earnings["confidence_score"].apply(
                lambda x: "#00d97e" if x > 70 else "#f6c343" if x > 50 else "#e63757"
            )
            fig_c.add_trace(go.Bar(x=earnings["symbol"], y=earnings["confidence_score"],
                                   name="Current", marker_color=colors))
            if "prev_confidence" in earnings.columns:
                fig_c.add_trace(go.Scatter(x=earnings["symbol"],
                                           y=earnings["prev_confidence"],
                                           name="Prev Quarter", mode="markers",
                                           marker=dict(size=10, color="white",
                                                       symbol="diamond")))
            fig_c.update_layout(title="Management Confidence (0–100)",
                                template="plotly_dark", height=380)
            st.plotly_chart(fig_c, use_container_width=True)

        with col2:
            if "evasion_scores" in earnings.columns:
                earnings["evasion"] = earnings["evasion_scores"].apply(
                    lambda x: x.get("evasion_score", 0) if isinstance(x, dict) else x
                )
            elif "evasion" not in earnings.columns:
                earnings["evasion"] = 30

            fig_e = px.bar(earnings.sort_values("evasion"),
                           x="evasion", y="symbol", orientation="h",
                           color="evasion", color_continuous_scale="RdYlGn_r",
                           title="Evasion Score (lower = better)")
            fig_e.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_e, use_container_width=True)


# ── Tab 5: Risk Monitor ───────────────────────────────────────────────────────
with tab5:
    st.subheader("Risk Monitor")

    # Live paper positions
    paper_pos = 0
    paper_state_path = cfg.data_dir / "paper_state.json"
    if paper_state_path.exists():
        try:
            state     = json.loads(paper_state_path.read_text())
            paper_pos = len(state.get("positions", {}))
        except Exception:
            pass

    col1, col2, col3 = st.columns(3)
    col1.metric("Open Positions", f"{paper_pos} / {cfg.signal.max_positions} max")
    col2.metric("Mode", cfg.kite.mode.upper())
    col3.metric("Max Position Size", f"₹{cfg.backtest.position_size/1e5:.0f}L")

    col1.metric("Sentiment Threshold", cfg.signal.sentiment_threshold)
    col2.metric("Z-Score Threshold",   cfg.signal.zscore_threshold)
    col3.metric("VIX Limit",           cfg.signal.vix_max)

    st.markdown("---")

    # Sector exposure (demo — replace with live position lookup)
    sectors = pd.DataFrame({
        "sector":       ["IT", "Banking", "Chemicals", "Auto", "Pharma", "Cap Goods"],
        "exposure_pct": [22, 18, 15, 12, 10, 8],
    })
    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(x=sectors["sector"], y=sectors["exposure_pct"],
                           name="Current", marker_color="#4dabf7"))
    fig_r.add_hline(y=cfg.signal.max_sector_pct * 100,
                    line_dash="dash", line_color="red",
                    annotation_text="Limit 25%")
    fig_r.update_layout(title="Sector Exposure vs Limit",
                        template="plotly_dark", height=320,
                        yaxis_title="% Portfolio")
    st.plotly_chart(fig_r, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<center><small>AlphaSense AI · Confidential · "
    f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}</small></center>",
    unsafe_allow_html=True,
)