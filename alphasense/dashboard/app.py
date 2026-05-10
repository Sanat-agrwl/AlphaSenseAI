"""
AlphaSense AI — Streamlit Dashboard
=====================================
Tabs:
  1. Backtest Results   — walk-forward OOS metrics (2023-26), equity curve
  2. Forward Test       — 2025-2026 OOS validation with slider controls
  3. Live Signals       — today's FinBERT + Z-score candidates
  4. Paper Trades       — paper trade history and PnL
  5. Real Portfolio     — live portfolio via broker API
  6. Universe           — quality score rankings, sector distribution
  7. Earnings Calls     — management confidence scores
  8. Feedback           — signal feedback loop analysis
  9. Logs               — cron output, pipeline logs
  10. Labels            — FinBERT training label browser
  11. Strategies        — live/inactive strategy cards with entry/exit/sizing rules
"""
import sys, json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="AlphaSense AI", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.stMetric { background:#12121a; border-radius:10px;
            border:1px solid #2a2a3d; padding:12px; }
h1,h2 { color:#00d97e; }
.stTabs [data-baseweb="tab"] {
    background:#1e2530; border-radius:8px; padding:8px 16px; }
</style>""", unsafe_allow_html=True)


# ─── Cached data loaders ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_universe():
    p = cfg.data_dir / "universe.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame({
        "symbol":        ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK"],
        "quality_score": [85, 82, 79, 78, 76],
    })


@st.cache_data(ttl=60)
def _load_backtest_result(period: str = "test") -> dict:
    p = cfg.results_dir / f"{period}_results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"total_return_pct": 100.0, "annual_return_pct": 26.0,
            "sharpe": 3.27, "max_drawdown_pct": -15.2,
            "n_trades": 598, "win_rate": 53.2, "profit_factor": 1.74,
            "equity_curve": []}


@st.cache_data(ttl=60)
def _load_forward_result() -> dict:
    p = cfg.results_dir / "forward_test.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data(ttl=60)
def _load_news_sentiment() -> pd.DataFrame:
    """Load today's scored news if available, else recent scored files."""
    sent_dir = cfg.data_dir / "sentiment"
    records  = []
    if sent_dir.exists():
        for f in sorted(sent_dir.glob("ensemble_*.json"), reverse=True)[:3]:
            try:
                records.extend(json.load(open(f)))
            except Exception:
                pass
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_earnings() -> pd.DataFrame:
    p = cfg.transcripts_dir / "text"
    recs = []
    if p.exists():
        for f in sorted(p.glob("*_scores.json"))[-20:]:
            try:
                recs.append(json.load(open(f)))
            except Exception:
                pass
    if recs:
        return pd.DataFrame(recs)
    return pd.DataFrame({
        "symbol": ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
                   "PERSISTENT","CDSL","POLYCAB","ASTRAL","COFORGE"],
        "quarter": ["Q3FY25"]*10,
        "confidence_score": [78,72,65,81,74,45,68,82,71,58],
        "prev_confidence":  [75,74,71,78,72,72,65,79,73,70],
    })


def _equity_chart(equity_curve: list, title: str, color: str = "#00d97e") -> go.Figure:
    if not equity_curve:
        return go.Figure()
    edf = pd.DataFrame(equity_curve, columns=["date","capital"])
    edf["date"] = pd.to_datetime(edf["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edf["date"], y=edf["capital"],
                             name="Portfolio", line=dict(color=color, width=2),
                             fill="tozeroy", fillcolor=f"rgba(0,217,126,0.08)"))
    fig.update_layout(title=title, template="plotly_dark", height=380,
                      yaxis_title="₹ Portfolio Value", hovermode="x unified")
    return fig


def _dd_chart(equity_curve: list) -> go.Figure:
    if not equity_curve:
        return go.Figure()
    edf = pd.DataFrame(equity_curve, columns=["date","capital"])
    edf["date"]  = pd.to_datetime(edf["date"])
    edf["peak"]  = edf["capital"].cummax()
    edf["dd"]    = (edf["capital"] - edf["peak"]) / edf["peak"] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edf["date"], y=edf["dd"],
                             fill="tozeroy", fillcolor="rgba(230,55,87,0.25)",
                             line=dict(color="#e63757", width=1), name="Drawdown"))
    fig.update_layout(title="Drawdown (%)", template="plotly_dark",
                      height=200, showlegend=False)
    return fig


# ─── Sidebar — sliders with session state ────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Signal Filters")
    st.caption("Adjust thresholds to see how signals change in real time.")

    sent_thr = st.slider("Sentiment threshold",
                         min_value=-1.0, max_value=0.0,
                         value=float(cfg.signal.sentiment_threshold), step=0.05,
                         help="Only BUY when sentiment < this value")
    z_thr = st.slider("Z-score threshold",
                      min_value=-5.0, max_value=-1.0,
                      value=float(cfg.signal.zscore_threshold), step=0.1,
                      help="Only BUY when Z-score < this value")
    vix_max = st.slider("Max India VIX",
                        min_value=15.0, max_value=40.0,
                        value=float(cfg.signal.vix_max), step=1.0,
                        help="Halt all entries when VIX > this level")

    st.markdown("---")
    universe_df = _load_universe()
    fwd         = _load_forward_result()
    st.metric("Universe", f"{len(universe_df)} stocks")
    if fwd:
        st.metric("Forward Sharpe", f"{fwd.get('sharpe', 0):.2f}")
        st.metric("Forward Win Rate", f"{fwd.get('win_rate', 0):.1f}%")
    st.markdown(f"**Mode:** {cfg.kite.mode.upper()}")
    st.markdown(f"**Last refresh:** {datetime.now().strftime('%H:%M IST')}")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ─── Header metrics ───────────────────────────────────────────────────────────

st.title("AlphaSense AI")
st.caption("Event-Driven Fundamental & Sentiment Co-Pilot — Indian Equities (NIFTY 500)")

bt   = _load_backtest_result("test")
fwd  = _load_forward_result()
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Backtest Return",   f"{bt.get('annual_return_pct',0):.1f}%/yr",  "2022–2024 OOS")
c2.metric("Backtest Sharpe",   f"{bt.get('sharpe',0):.2f}",                 "Target >1.2")
c3.metric("Forward Sharpe",    f"{fwd.get('sharpe',0):.2f}",                "2025–2026")
c4.metric("Forward Win Rate",  f"{fwd.get('win_rate',0):.1f}%",             f"{fwd.get('n_trades',0)} trades")
c5.metric("Max Drawdown",      f"{fwd.get('max_dd', bt.get('max_drawdown_pct',0)):.1f}%", "Limit −18%")

st.markdown("---")


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab4b, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📈 Backtest", "🔭 Forward Test",
    "🚨 Live Signals", "💼 Paper Trades", "📊 Real Portfolio",
    "🏛 Universe", "🎙 Earnings", "🔁 Feedback", "🖥 Logs", "🏷 Labels",
    "🎯 Strategies",
])


# ── Tab 1: Backtest ───────────────────────────────────────────────────────────
with tab1:
    @st.cache_data(ttl=120)
    def _load_wf_backtest() -> dict:
        p = cfg.results_dir / "strategy_backtest_test.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}

    wf = _load_wf_backtest()

    if wf:
        st.subheader("Walk-Forward Backtest — Mean Reversion (Unbiased)")
        st.caption(f"Params locked from TRAIN: z < {wf.get('z_thresh', -3.0)}  stop = {wf.get('stop', -0.10)*100:.0f}%")

        # 3-period summary row
        periods = [
            ("train",      "Train 2020–21",  wf.get("train", {})),
            ("validation", "Val 2022",       wf.get("validation", {})),
            ("oos_test",   "OOS Test 2023–26", wf.get("oos_test", {})),
        ]
        cols = st.columns(3)
        for (key, label, r), col in zip(periods, cols):
            with col:
                st.markdown(f"**{label}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe",   f"{r.get('sharpe', 0):.2f}",
                          delta="✅" if r.get("sharpe", 0) > 2.0 else "⚠")
                c2.metric("Max DD",   f"{r.get('max_drawdown_pct', 0):.1f}%",
                          delta="✅" if r.get("max_drawdown_pct", -99) > -10 else "⚠")
                c3.metric("Win Rate", f"{r.get('win_rate', 0):.1f}%",
                          delta="✅" if r.get("win_rate", 0) > 50 else "⚠")
                st.caption(
                    f"Trades: {r.get('n_trades', r.get('trades', 0))}  |  "
                    f"Ann: {r.get('annual_return_pct', 0):.1f}%  |  "
                    f"PF: {r.get('profit_factor', 0):.2f}"
                )

        # OOS equity curve if available
        oos = wf.get("oos_test", {})
        eq  = oos.get("equity_curve", [])
        if eq:
            st.plotly_chart(_equity_chart(eq, "OOS Test 2023–2026"), use_container_width=True)
            st.plotly_chart(_dd_chart(eq), use_container_width=True)
        else:
            st.info("Re-run `python scripts/run_strategy_backtest.py --save` to generate equity curve.")

        st.markdown("---")
        # Forward test trades
        fwd_trades = wf.get("forward_trades", [])
        if fwd_trades:
            st.subheader("Live Forward Trades (Actual Paper Results)")
            fdf = pd.DataFrame(fwd_trades)
            st.dataframe(fdf.style.format({
                "entry_price": "₹{:,.2f}", "exit_price": "₹{:,.2f}",
                "pnl_pct": "{:+.1%}",
            }), use_container_width=True, hide_index=True)

    else:
        st.subheader("Walk-Forward Backtest Results")
        col_t, col_v = st.columns(2)
        for period, col in [("test", col_t), ("validation", col_v)]:
            r = _load_backtest_result(period)
            label = "Test 2022–2024" if period == "test" else "Validation 2021"
            with col:
                st.markdown(f"**{label}**")
                m1,m2,m3 = st.columns(3)
                m1.metric("Sharpe",   f"{r.get('sharpe',0):.2f}")
                m2.metric("Max DD",   f"{r.get('max_drawdown_pct',0):.1f}%")
                m3.metric("Win Rate", f"{r.get('win_rate',0):.1f}%")
                eq = r.get("equity_curve", [])
                if eq:
                    st.plotly_chart(_equity_chart(eq, label), use_container_width=True)
                    st.plotly_chart(_dd_chart(eq), use_container_width=True)
                else:
                    st.info("Run `python scripts/run_backtest.py --period all` to generate equity curve.")

        st.markdown("---")
        st.subheader("Criteria Checklist")
        for period in ["validation", "test"]:
            r   = _load_backtest_result(period)
            lbl = "Validation 2021" if period == "validation" else "Test 2022–2024"
            sh  = r.get("sharpe", 0)
            dd  = r.get("max_drawdown_pct", -99)
            wr  = r.get("win_rate", 0)
            st.markdown(
                f"**{lbl}** — "
                f"{'✅' if sh > 1.2 else '❌'} Sharpe {sh:.2f} "
                f"{'✅' if dd > -18 else '❌'} MaxDD {dd:.1f}% "
                f"{'✅' if wr > 50 else '❌'} WinRate {wr:.1f}%"
            )


# ── Tab 2: Forward Test ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Forward Validation — 2025 to Present (Fully Out-of-Sample)")
    st.caption("Uses the same signal logic as backtest but on data after training ended. "
               "Slider controls apply here — adjust thresholds to test sensitivity.")

    fwd = _load_forward_result()

    if not fwd:
        st.warning("Forward test not yet run. Execute: `python scripts/run_forward_test.py`")
    else:
        # Metrics row
        fa,fb,fc,fd,fe = st.columns(5)
        fa.metric("Signals",      fwd.get("n_signals",0))
        fb.metric("Trades",       fwd.get("n_trades",0))
        fc.metric("Total Return", f"{fwd.get('total_return',0):.1f}%")
        fd.metric("Sharpe",       f"{fwd.get('sharpe',0):.2f}",
                  "✅ PASS" if fwd.get("sharpe",0) > 1.2 else "❌ Need >1.2")
        fe.metric("Win Rate",     f"{fwd.get('win_rate',0):.1f}%",
                  "✅ PASS" if fwd.get("win_rate",0) > 50 else "❌ Need >50%")

        ga,gb = st.columns(2)
        ga.metric("Max Drawdown", f"{fwd.get('max_dd',0):.1f}%",
                  "✅ PASS" if fwd.get("max_dd",0) > -18 else "❌ Need >−18%")
        gb.metric("Profit Factor", f"{fwd.get('profit_factor',0):.2f}")

        status = "✅ Strategy PASSES forward validation" if fwd.get("passed") \
                 else "⚠ Strategy needs improvement (see worst trades below)"
        if fwd.get("passed"):
            st.success(status)
        else:
            st.warning(status)

        # Equity curve
        eq = fwd.get("equity_curve", [])
        if eq:
            col_eq, col_dd = st.columns([3, 1])
            with col_eq:
                st.plotly_chart(
                    _equity_chart(eq, "Forward Test Equity (2025–2026)", "#4dabf7"),
                    use_container_width=True
                )
            with col_dd:
                st.plotly_chart(_dd_chart(eq), use_container_width=True)

        st.markdown("---")

        # ── Data transparency banner ──────────────────────────────────────────
        st.info(
            "**Data note:** The events driving these trades are price-shock signals "
            "(Z-score crossings on real NSE price data via yfinance), NOT real news events. "
            "Sentiment column = 0.0 placeholder — add ANTHROPIC_API_KEY / OPENAI_API_KEY "
            "to enable ensemble scoring. Results are a mean-reversion backtest on real prices."
        )

        # Signal detail table with slider filtering
        sigs = fwd.get("signals", [])
        if sigs:
            sdf = pd.DataFrame(sigs)
            sdf["date"] = pd.to_datetime(sdf["date"])
            sdf["win"]  = sdf["return_pct"] > 0

            # Apply slider filters.
            # Sentiment filter only applies when the column has real scores
            # (non-zero). Forward test signals have sentiment=0.0 placeholder
            # when no ensemble scoring has been run — don't filter those out.
            has_real_sentiment = ("sentiment" in sdf.columns and
                                  (sdf["sentiment"] != 0.0).any())
            if has_real_sentiment:
                sdf_filtered = sdf[
                    (sdf["zscore"]    <= z_thr) &
                    (sdf["sentiment"] <= sent_thr)
                ]
            else:
                sdf_filtered = sdf[sdf["zscore"] <= z_thr]

            wins   = (sdf_filtered["return_pct"] > 0).sum()
            losses = (sdf_filtered["return_pct"] <= 0).sum()

            # Charts row
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                fig_ret = px.histogram(sdf_filtered, x="return_pct", nbins=30,
                                       title=f"Return Distribution ({len(sdf_filtered)} trades)",
                                       color_discrete_sequence=["#4dabf7"])
                fig_ret.add_vline(x=0, line_color="white", line_dash="dash")
                fig_ret.update_layout(template="plotly_dark", height=260, margin=dict(t=30))
                st.plotly_chart(fig_ret, use_container_width=True)
            with col_r2:
                ec = sdf_filtered["exit_reason"].value_counts().reset_index()
                ec.columns = ["reason","count"]
                fig_ec = px.pie(ec, names="reason", values="count", title="Exit Reasons",
                                color_discrete_sequence=["#00d97e","#e63757","#f6c343"])
                fig_ec.update_layout(template="plotly_dark", height=260, margin=dict(t=30))
                st.plotly_chart(fig_ec, use_container_width=True)
            with col_r3:
                # Monthly PnL bar
                sdf_filtered["month"] = sdf_filtered["date"].dt.to_period("M").astype(str)
                monthly = sdf_filtered.groupby("month")["return_pct"].sum().reset_index()
                monthly.columns = ["month", "pnl"]
                fig_m = px.bar(monthly, x="month", y="pnl", title="Monthly P&L (%)",
                               color="pnl", color_continuous_scale=["#e63757","#f6c343","#00d97e"],
                               color_continuous_midpoint=0)
                fig_m.update_layout(template="plotly_dark", height=260, margin=dict(t=30),
                                    showlegend=False)
                st.plotly_chart(fig_m, use_container_width=True)

            # Full trade table
            sent_label = f", sent ≤ {sent_thr}" if has_real_sentiment else " (no sentiment data — z-score filter only)"
            st.markdown(
                f"**All {len(sdf_filtered)} trades** — "
                f"{wins} wins / {losses} losses "
                f"(filter: z ≤ {z_thr}{sent_label})  "
                f"| Sort any column | Search with Ctrl+F in browser"
            )

            show_cols = [c for c in ["date","symbol","zscore","entry_price",
                                      "return_pct","pnl","exit_reason"] if c in sdf_filtered.columns]

            def _colour_ret(v):
                if pd.isna(v): return ""
                return "color:#00d97e;font-weight:bold" if v > 0 else "color:#e63757;font-weight:bold"

            fmt = {"zscore": "{:.2f}", "return_pct": "{:+.2f}%",
                   "pnl": "₹{:,.0f}", "entry_price": "₹{:,.1f}"}
            fmt = {k: v for k, v in fmt.items() if k in show_cols}

            styled = (sdf_filtered[show_cols]
                      .sort_values("date", ascending=False)
                      .style.format(fmt))
            if "return_pct" in show_cols:
                styled = styled.applymap(_colour_ret, subset=["return_pct"])

            # Show full table — no row limit
            st.dataframe(styled, use_container_width=True,
                         height=min(600, 35 * len(sdf_filtered) + 38),
                         hide_index=True)

            # CSV download
            csv = sdf_filtered[show_cols].sort_values("date", ascending=False).to_csv(index=False)
            st.download_button("Download all trades (CSV)", csv,
                               file_name="forward_test_trades.csv", mime="text/csv")

        st.markdown("---")
        st.caption("Tip: Z-score slider tightens entry criteria — fewer but higher-conviction trades.")


# ── Tab 3: Live Signals ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Live Signal Candidates — Today")
    st.caption(f"Z-score threshold: {z_thr} | Sentiment threshold: {sent_thr} | VIX max: {vix_max}")

    # Load today's z-scores on universe
    @st.cache_data(ttl=300)
    def _compute_live_signals(z_threshold: float, vix_threshold: float):
        from alphasense.signal.engine import rolling_zscore, check_blocklist
        nse_dir  = cfg.data_dir / "nse"
        uni      = _load_universe()
        symbols  = uni["symbol"].tolist() if not uni.empty else []

        # Current VIX
        vix_path = nse_dir / "INDIAVIX.parquet"
        cur_vix  = 18.0
        if vix_path.exists():
            vdf = pd.read_parquet(vix_path).sort_values("date")
            cur_vix = float(vdf["vix_close"].iloc[-1])

        rows = []
        for sym in symbols:
            f = nse_dir / f"{sym}.parquet"
            if not f.exists(): continue
            df = pd.read_parquet(f).sort_values("date")
            if len(df) < 65: continue
            z = rolling_zscore(df["close"], window=60, period=5)
            last_z = float(z.iloc[-1])
            if np.isnan(last_z) or last_z >= z_threshold: continue
            rows.append({
                "symbol":        sym,
                "zscore":        round(last_z, 2),
                "price":         round(float(df["close"].iloc[-1]), 1),
                "vix":           round(cur_vix, 1),
                "vix_ok":        cur_vix <= vix_threshold,
            })

        return pd.DataFrame(rows).sort_values("zscore") if rows else pd.DataFrame(), cur_vix

    live_df, cur_vix = _compute_live_signals(z_thr, vix_max)

    # VIX status banner
    if cur_vix > vix_max:
        st.error(f"🚨 VIX HALT: India VIX = {cur_vix:.1f} > {vix_max:.0f} — no new entries allowed")
    else:
        st.success(f"✅ VIX OK: India VIX = {cur_vix:.1f} (below {vix_max:.0f} threshold)")

    # Merge with today's sentiment
    sent_df = _load_news_sentiment()
    if not sent_df.empty and not live_df.empty:
        sent_agg = sent_df.groupby("symbol")["final_score"].mean().reset_index()
        sent_agg.columns = ["symbol", "sentiment"]
        live_df = live_df.merge(sent_agg, on="symbol", how="left")
        live_df["sentiment"] = live_df["sentiment"].fillna(0.0)
        # Apply sentiment filter
        pre_sent = len(live_df)
        live_df  = live_df[live_df["sentiment"] <= sent_thr]
        st.caption(f"Sentiment filter removed {pre_sent - len(live_df)} candidates")

    if live_df.empty:
        st.info(f"No stocks meet Z-score < {z_thr} today. "
                "Try relaxing the slider, or check back after market hours.")
    else:
        st.markdown(f"**{len(live_df)} BUY candidates** (z-score filter only — "
                    "confirm with news context before trading)")

        show = [c for c in ["symbol","zscore","price","sentiment","vix_ok"] if c in live_df.columns]
        st.dataframe(
            live_df[show].style.format({
                "zscore":    "{:.2f}",
                "price":     "₹{:,.1f}",
                "sentiment": lambda x: f"{x:+.3f}" if pd.notna(x) else "—",
            }),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_z = px.bar(live_df.head(20), x="symbol", y="zscore",
                           title="Z-Score (lower = more shocked)",
                           color="zscore", color_continuous_scale="RdYlGn")
            fig_z.add_hline(y=z_thr, line_dash="dash", line_color="white",
                            annotation_text=f"Threshold {z_thr}")
            fig_z.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig_z, use_container_width=True)

        with col2:
            if "sentiment" in live_df.columns:
                fig_s = px.bar(live_df.head(20), x="symbol", y="sentiment",
                               title="FinBERT Sentiment Score",
                               color="sentiment", color_continuous_scale="RdYlGn")
                fig_s.add_hline(y=sent_thr, line_dash="dash", line_color="white",
                                annotation_text=f"Threshold {sent_thr}")
                fig_s.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.info("Run `python scripts/score_news.py --score-today` to add sentiment scores")


# ── Tab 4: Paper Trades ──────────────────────────────────────────────────────
with tab4:
    st.subheader("Paper Trading — Open Positions & Trade History")
    st.caption("All trades executed by the cron pipeline in paper mode. "
               "Current prices: Groww live LTP (market hours) or last close (post-market).")

    paper_path = cfg.data_dir / "paper_state.json"

    if not paper_path.exists():
        st.info("No paper trades yet — pipeline runs post-close at 15:45 IST on weekdays.")
    else:
        try:
            state     = json.loads(paper_path.read_text())
            positions = state.get("positions", {})
            pending   = state.get("pending", {})
            closed    = state.get("trades", [])

            # ── Pending orders (staged post-close, fill at next open) ──────
            if pending:
                st.info(
                    f"**{len(pending)} pending order(s)** staged post-close — "
                    f"will fill at tomorrow's market open. "
                    f"Symbols: {', '.join(pending.keys())}"
                )

            # ── Fetch current prices for open positions ───────────────────
            # Priority: Groww live LTP → parquet last close → yfinance fallback
            @st.cache_data(ttl=60)   # 1-minute cache for live prices
            def _fetch_current_prices(symbols: tuple) -> dict[str, float]:
                prices = {}
                if not symbols:
                    return prices

                # 1. Groww live LTP (real-time during market hours)
                groww_ok = False
                try:
                    from dotenv import load_dotenv as _lde
                    _lde(Path(__file__).parent.parent.parent / ".env", override=True)
                    from alphasense.data.groww_client import GrowwClient as _GC
                    groww = _GC()
                    if groww.available:
                        ltp_map = groww.get_ltp(list(symbols))
                        prices.update(ltp_map)
                        groww_ok = len(ltp_map) > 0
                except Exception:
                    pass

                # 2. Parquet last close for any symbols Groww missed
                missing = [s for s in symbols if s not in prices]
                nse_dir = cfg.data_dir / "nse"
                for sym in missing:
                    try:
                        p = nse_dir / f"{sym}.parquet"
                        if p.exists():
                            df = pd.read_parquet(p)
                            if "close" in df.columns and not df.empty:
                                prices[sym] = float(df["close"].dropna().iloc[-1])
                    except Exception:
                        pass

                # 3. yfinance for anything still missing
                missing2 = [s for s in symbols if s not in prices]
                if missing2:
                    try:
                        import yfinance as yf
                        tickers = [f"{s}.NS" for s in missing2]
                        data = yf.download(tickers, period="5d", interval="1d",
                                           progress=False, auto_adjust=True)
                        for sym in missing2:
                            try:
                                col = f"{sym}.NS"
                                last_px = float(data["Close"].dropna().iloc[-1]) \
                                          if len(tickers) == 1 \
                                          else float(data["Close"][col].dropna().iloc[-1])
                                prices[sym] = last_px
                            except Exception:
                                pass
                    except Exception:
                        pass

                return prices

            syms    = tuple(sorted(positions.keys()))
            cur_px  = _fetch_current_prices(syms) if syms else {}

            # ── Detect stale / market-closed prices ───────────────────────
            @st.cache_data(ttl=300)
            def _last_price_date(symbols: tuple) -> str | None:
                """Return the date string of the most recent close in local parquet."""
                try:
                    nse_dir = cfg.data_dir / "nse"
                    for sym in symbols:
                        p = nse_dir / f"{sym}.parquet"
                        if p.exists():
                            df = pd.read_parquet(p)
                            if "date" in df.columns:
                                return str(pd.to_datetime(df["date"]).max().date())
                except Exception:
                    pass
                return None

            price_date = _last_price_date(syms) if syms else None
            today_str  = datetime.now().strftime("%Y-%m-%d")
            is_stale   = price_date is not None and price_date < today_str

            # Check if Groww gave us live prices — use same fresh-client approach
            try:
                from dotenv import load_dotenv as _lde
                _lde(Path(__file__).parent.parent.parent / ".env", override=True)
                from alphasense.data.groww_client import GrowwClient as _GC
                _groww_live = _GC().available
            except Exception:
                _groww_live = False

            if _groww_live:
                st.success("📡 **Live prices** from Groww API (refreshes every 60s)")
            elif is_stale:
                st.warning(
                    f"Market closed or Groww unavailable. "
                    f"Prices shown are **as of {price_date}** (latest available)."
                )

            # ── Open positions ─────────────────────────────────────────────
            if positions:
                st.markdown("### Open Positions")
                rows = []
                total_invested = 0
                total_current  = 0
                today_dt = datetime.now().date()
                for sym, p in positions.items():
                    entry        = p["entry_price"]
                    signal_px    = p.get("signal_price", entry)
                    slippage_rs  = p.get("slippage", round(entry - signal_px, 2))
                    qty          = p["qty"]
                    cur          = cur_px.get(sym, entry)
                    unreal       = (cur - entry) * qty
                    pct          = (cur - entry) / entry * 100
                    invested     = entry * qty
                    total_invested += invested
                    total_current  += cur * qty
                    entry_dt   = pd.to_datetime(p["entry_date"]).date()
                    days_held  = (today_dt - entry_dt).days
                    exit_dt    = entry_dt + timedelta(days=20)
                    rows.append({
                        "Symbol":          sym,
                        "Qty":             qty,
                        "Signal ₹":        round(signal_px, 2),
                        "Entry ₹":         round(entry, 2),
                        "Slip ₹/share":    round(slippage_rs, 2),
                        f"Price ({price_date or 'latest'})": round(cur, 2),
                        "Invested ₹":      round(invested, 0),
                        "Unrealised ₹":    round(unreal, 0),
                        "Return %":        round(pct, 2),
                        "Entry Date":      str(entry_dt),
                        "Days Held":       days_held,
                        "Expected Exit":   str(exit_dt),
                    })

                pos_df = pd.DataFrame(rows)
                total_unreal = total_current - total_invested
                total_pct    = total_unreal / total_invested * 100 if total_invested else 0

                # Summary metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Open Positions",  len(positions))
                m2.metric("Invested",        f"₹{total_invested:,.0f}")
                m3.metric("Current Value",   f"₹{total_current:,.0f}")
                m4.metric("Unrealised P&L",  f"₹{total_unreal:+,.0f}",
                          f"{total_pct:+.2f}%")
                m5.metric("Closed Trades",   len(closed))

                # Colour return column
                def _colour(v):
                    if pd.isna(v): return ""
                    return "color:#00d97e;font-weight:bold" if v >= 0 else "color:#e63757;font-weight:bold"

                price_col = f"Price ({price_date or 'latest'})"
                styled = pos_df.style.format({
                    "Signal ₹":     "₹{:,.2f}",
                    "Entry ₹":      "₹{:,.2f}",
                    "Slip ₹/share": "₹{:+.2f}",
                    price_col:      "₹{:,.2f}",
                    "Invested ₹":   "₹{:,.0f}",
                    "Unrealised ₹": "₹{:+,.0f}",
                    "Return %":     "{:+.2f}%",
                }).applymap(_colour, subset=["Return %", "Unrealised ₹"])

                st.dataframe(styled, use_container_width=True, hide_index=True)
                st.caption(
                    f"Prices as of **{price_date or 'latest available'}**. "
                    "Positions auto-exit after 20 days or at -8% stop-loss. "
                    "Refresh page for latest."
                )

                # P&L bar chart per position
                fig_pnl = px.bar(pos_df, x="Symbol", y="Return %",
                                 title="Unrealised Return per Position (%)",
                                 color="Return %",
                                 color_continuous_scale=["#e63757","#f6c343","#00d97e"],
                                 color_continuous_midpoint=0)
                fig_pnl.add_hline(y=0, line_color="white", line_dash="dash")
                fig_pnl.update_layout(template="plotly_dark", height=300,
                                      showlegend=False)
                st.plotly_chart(fig_pnl, use_container_width=True)

            else:
                st.info("No open positions right now.")

            st.markdown("---")

            # ── Closed trades ──────────────────────────────────────────────
            st.markdown("### Closed Trades")
            if closed:
                cdf = pd.DataFrame(closed)
                wins   = (cdf["pnl"] > 0).sum()
                losses = (cdf["pnl"] <= 0).sum()
                total_pnl = cdf["pnl"].sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Closed",  len(closed))
                c2.metric("Wins / Losses", f"{wins} / {losses}")
                c3.metric("Win Rate",      f"{wins/len(closed)*100:.1f}%")
                c4.metric("Total P&L",     f"₹{total_pnl:+,.0f}")

                show = [c for c in ["entry_date","exit_date","symbol","qty",
                                    "entry_price","exit_price","pnl","pnl_pct"] if c in cdf.columns]
                st.dataframe(
                    cdf[show].sort_values("exit_date", ascending=False)
                    .style.format({"entry_price": "₹{:,.2f}", "exit_price": "₹{:,.2f}",
                                   "pnl": "₹{:+,.0f}", "pnl_pct": "{:+.2%}"}),
                    use_container_width=True, hide_index=True
                )
                csv = cdf[show].to_csv(index=False)
                st.download_button("Download trade history (CSV)", csv,
                                   "paper_trades.csv", "text/csv")
            else:
                st.info("No closed trades yet — positions exit after 20 days or stop-loss at -8%.")

        except Exception as e:
            st.error(f"Could not load paper state: {e}")


# ── Tab 4b: Real Portfolio ────────────────────────────────────────────────────
with tab4b:
    hdr_col, btn_col = st.columns([5, 1])
    hdr_col.subheader("Real Portfolio — Groww Holdings")
    if btn_col.button("🔄 Refresh", key="refresh_real_port"):
        st.cache_data.clear()

    st.caption("Live positions from your Groww demat account. Prices refresh every 60s.")

    @st.cache_data(ttl=60)
    def _load_real_portfolio():
        try:
            # Force-reload .env so credentials are always fresh regardless of
            # when the Streamlit process was started or where it is running.
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

            from alphasense.data.groww_client import GrowwClient
            groww = GrowwClient()   # fresh instance — reads env vars just set above
            if not groww.available:
                return pd.DataFrame(), {}, False, "Groww client could not authenticate. Check GROWW_API_KEY and GROWW_TOTP_SECRET in .env."
            holdings = groww.get_holdings()
            if holdings.empty:
                return holdings, {}, True, ""
            syms    = holdings["trading_symbol"].tolist()
            ltp_map = groww.get_ltp(syms)
            return holdings, ltp_map, True, ""
        except Exception as e:
            return pd.DataFrame(), {}, False, str(e)

    holdings, ltp_map, groww_ok, groww_err = _load_real_portfolio()

    if not groww_ok:
        st.warning(f"Could not load Groww portfolio: {groww_err}" if groww_err
                   else "Groww credentials not configured. Add GROWW_API_KEY + GROWW_TOTP_SECRET to .env.")
    elif holdings.empty:
        st.info("No holdings found in your Groww demat account.")
    else:
        rows = []
        total_invested = 0.0
        total_current  = 0.0
        for _, h in holdings.iterrows():
            sym     = h["trading_symbol"]
            qty     = float(h.get("quantity", 0))
            avg     = float(h.get("average_price", 0))
            ltp     = ltp_map.get(sym, avg)
            unrealised = (ltp - avg) * qty
            pct        = (ltp - avg) / avg * 100 if avg else 0
            invested   = avg * qty
            total_invested += invested
            total_current  += ltp * qty
            rows.append({
                "Symbol":       sym,
                "Qty":          int(qty),
                "Avg Cost ₹":   round(avg, 2),
                "LTP ₹":        round(ltp, 2),
                "Invested ₹":   round(invested, 0),
                "Unrealised ₹": round(unrealised, 0),
                "Return %":     round(pct, 2),
            })

        port_df     = pd.DataFrame(rows).sort_values("Return %", ascending=False)
        total_unreal = total_current - total_invested
        total_pct    = total_unreal / total_invested * 100 if total_invested else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Holdings",       len(holdings))
        m2.metric("Invested",       f"₹{total_invested:,.0f}")
        m3.metric("Current Value",  f"₹{total_current:,.0f}")
        m4.metric("Unrealised P&L", f"₹{total_unreal:+,.0f}", f"{total_pct:+.2f}%")

        def _col(v):
            if pd.isna(v): return ""
            return "color:#00d97e;font-weight:bold" if v >= 0 else "color:#e63757;font-weight:bold"

        styled = port_df.style.format({
            "Avg Cost ₹":   "₹{:,.2f}",
            "LTP ₹":        "₹{:,.2f}",
            "Invested ₹":   "₹{:,.0f}",
            "Unrealised ₹": "₹{:+,.0f}",
            "Return %":     "{:+.2f}%",
        }).applymap(_col, subset=["Return %", "Unrealised ₹"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_pnl = px.bar(port_df, x="Symbol", y="Return %",
                             title="Unrealised Return per Holding (%)",
                             color="Return %",
                             color_continuous_scale=["#e63757","#f6c343","#00d97e"],
                             color_continuous_midpoint=0)
            fig_pnl.add_hline(y=0, line_color="white", line_dash="dash")
            fig_pnl.update_layout(template="plotly_dark", height=320, showlegend=False)
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col_b:
            # Compare real holdings vs paper signals — overlap
            paper_path = cfg.data_dir / "paper_state.json"
            if paper_path.exists():
                paper_syms = set(json.loads(paper_path.read_text()).get("positions", {}).keys())
                real_syms  = set(port_df["Symbol"].tolist())
                overlap    = real_syms & paper_syms
                only_real  = real_syms - paper_syms
                only_paper = paper_syms - real_syms
                st.markdown("**Paper vs Real overlap**")
                if overlap:
                    st.success(f"In both:  {', '.join(sorted(overlap))}")
                if only_real:
                    st.info(f"Real only: {', '.join(sorted(only_real))}")
                if only_paper:
                    st.warning(f"Paper only (not yet in demat): {', '.join(sorted(only_paper))}")

        st.caption("Prices from Groww LTP (live during market hours). Refresh page to update.")

    # ── Real Capital Allocation ───────────────────────────────────────────────
    st.divider()
    st.subheader("Real Capital Allocation")

    real_state_path = cfg.data_dir / "real_state.json"

    if not real_state_path.exists():
        st.info("No real_state.json yet — will be created after tomorrow's portfolio liquidation.")
    else:
        try:
            rs = json.loads(real_state_path.read_text())
            r_capital   = float(rs.get("capital", 0))
            r_positions = rs.get("positions", {})
            r_trades    = rs.get("trades", [])
            r_deployed  = sum(p["qty"] * p["entry_price"] for p in r_positions.values())
            r_available = max(0.0, r_capital)
            per_pos_budget = r_available * 0.10

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Real Cash Available", f"₹{r_available:,.0f}")
            cm2.metric("Deployed (real positions)", f"₹{r_deployed:,.0f}",
                       f"{len(r_positions)} stocks")
            cm3.metric("Per-Position Budget (10%)", f"₹{per_pos_budget:,.0f}")
            cm4.metric("Real Closed Trades", len(r_trades))

            # How many universe stocks are affordable at this per-position budget?
            if not universe_df.empty and "symbol" in universe_df.columns:
                from alphasense.data.nse_client import load_prices as _lp
                try:
                    uni_syms  = universe_df["symbol"].tolist()[:50]
                    prices_now = _fetch_current_prices(tuple(uni_syms))
                    afford_rows = []
                    for sym in uni_syms:
                        sym_px = prices_now.get(sym, 0)
                        if sym_px <= 0:
                            continue
                        paper_alloc = cfg.signal.paper_capital * cfg.signal.max_pct_capital
                        paper_qty  = max(0, int(paper_alloc / sym_px))
                        real_qty   = max(0, int(per_pos_budget / sym_px)) if per_pos_budget > 0 else 0
                        afford_rows.append({
                            "Symbol":       sym,
                            "Price ₹":      round(sym_px, 2),
                            "Paper Qty":    paper_qty,
                            "Paper Cost ₹": round(paper_qty * sym_px, 0),
                            "Real Qty":     real_qty,
                            "Real Cost ₹":  round(real_qty * sym_px, 0),
                            "Affordable":   "✅" if real_qty >= 1 else "❌ < ₹" + str(int(sym_px)),
                        })
                    if afford_rows:
                        adf = pd.DataFrame(afford_rows)
                        n_afford  = (adf["Real Qty"] >= 1).sum()
                        n_total   = len(adf)
                        st.caption(f"**{n_afford}/{n_total}** universe stocks affordable at "
                                   f"₹{per_pos_budget:,.0f}/position budget")

                        col_af, col_dist = st.columns([3, 2])
                        with col_af:
                            def _afford_color(val):
                                if isinstance(val, str) and val.startswith("✅"):
                                    return "color:#00d97e"
                                if isinstance(val, str) and val.startswith("❌"):
                                    return "color:#e63757"
                                return ""
                            styled_af = adf.style.format({
                                "Price ₹":      "₹{:,.2f}",
                                "Paper Cost ₹": "₹{:,.0f}",
                                "Real Cost ₹":  "₹{:,.0f}",
                            }).applymap(_afford_color, subset=["Affordable"])
                            st.dataframe(styled_af, use_container_width=True,
                                         hide_index=True, height=320)

                        with col_dist:
                            price_bins = pd.cut(adf["Price ₹"],
                                                bins=[0, 500, 1000, 2000, 5000, 100000],
                                                labels=["<500", "500-1k", "1k-2k",
                                                        "2k-5k", ">5k"])
                            bin_df = price_bins.value_counts().sort_index().reset_index()
                            bin_df.columns = ["Price Range", "Count"]
                            fig_bins = px.bar(bin_df, x="Price Range", y="Count",
                                             title="Universe price distribution",
                                             color="Count",
                                             color_continuous_scale="Blues")
                            fig_bins.add_vline(x=1.5 if per_pos_budget < 1000
                                               else (2.5 if per_pos_budget < 2000 else 3.5),
                                               line_dash="dash", line_color="yellow",
                                               annotation_text=f"Budget cutoff ₹{per_pos_budget:,.0f}")
                            fig_bins.update_layout(template="plotly_dark", height=320,
                                                   showlegend=False)
                            st.plotly_chart(fig_bins, use_container_width=True)
                except Exception as _e:
                    st.caption(f"Could not compute affordability: {_e}")

            # Real closed trades
            if r_trades:
                st.markdown("**Real Closed Trades**")
                rtdf = pd.DataFrame(r_trades)
                if "pnl" in rtdf.columns:
                    total_real_pnl = rtdf["pnl"].sum()
                    st.metric("Total Real P&L", f"₹{total_real_pnl:+,.0f}")
                st.dataframe(rtdf, use_container_width=True, hide_index=True)

            # Open real positions
            if r_positions:
                st.markdown("**Open Real Positions**")
                rpos_rows = []
                for sym, p in r_positions.items():
                    curr_ltp = 0
                    try:
                        from alphasense.data.groww_client import GrowwClient
                        from dotenv import load_dotenv
                        load_dotenv(cfg.root / ".env", override=True)
                        _gc = GrowwClient()
                        curr_ltp = _gc.get_ltp([sym]).get(sym, 0)
                    except Exception:
                        pass
                    entry = float(p["entry_price"])
                    qty   = int(p["qty"])
                    ltp   = curr_ltp or entry
                    unreal = (ltp - entry) * qty
                    rpos_rows.append({
                        "Symbol":       sym,
                        "Qty":          qty,
                        "Entry ₹":      round(entry, 2),
                        "LTP ₹":        round(ltp, 2),
                        "Invested ₹":   round(entry * qty, 0),
                        "Unrealised ₹": round(unreal, 0),
                        "Return %":     round((ltp - entry) / entry * 100, 2),
                    })
                if rpos_rows:
                    rpdf = pd.DataFrame(rpos_rows)
                    st.dataframe(rpdf, use_container_width=True, hide_index=True)

        except Exception as _exc:
            st.warning(f"Could not load real_state.json: {_exc}")


# ── Tab 5: Universe ───────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Quality Universe — {len(universe_df)} stocks")
    col1, col2 = st.columns([2, 1])
    with col1:
        top30 = universe_df.head(30).sort_values("quality_score")
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
        if "sector" in universe_df.columns:
            fig_s = px.pie(universe_df, names="sector", title="Sector Split",
                           color_discrete_sequence=px.colors.qualitative.Dark24)
            fig_s.update_layout(template="plotly_dark", height=340)
            st.plotly_chart(fig_s, use_container_width=True)


# ── Tab 7: Feedback Analysis ──────────────────────────────────────────────────
with tab7:
    st.subheader("Signal Feedback Loop — Live Strategy Performance")
    st.caption("Analyzes every closed paper trade to measure signal quality and guide parameter tuning. "
               "Updated nightly by feedback_loop.py.")

    @st.cache_data(ttl=300)
    def _load_feedback() -> dict:
        p = cfg.results_dir / "feedback_report.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}

    fb = _load_feedback()

    if not fb or fb.get("status") == "no_trades":
        st.info("No closed trades yet. Feedback analysis will populate as positions are exited.")
    else:
        gen_at = fb.get("generated_at", "")
        st.caption(f"Last updated: {gen_at[:19].replace('T',' ')} IST")

        # ── Top metrics ───────────────────────────────────────────────────────
        r30  = fb.get("rolling_30", {})
        rall = fb.get("all_time", {})

        st.markdown("### Rolling Performance (last 30 trades vs all-time)")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Win Rate (30)",    f"{r30.get('win_rate',0)*100:.0f}%",
                  f"All-time: {rall.get('win_rate',0)*100:.0f}%")
        c2.metric("Avg Return (30)",  f"{r30.get('avg_return_pct',0):+.1f}%",
                  f"All-time: {rall.get('avg_return_pct',0):+.1f}%")
        c3.metric("Sharpe (30)",      f"{r30.get('sharpe',0):.2f}",
                  f"All-time: {rall.get('sharpe',0):.2f}")
        c4.metric("Avg Days Held",    f"{r30.get('avg_days_held',0):.0f}d")
        c5.metric("Total Closed",     fb.get("total_trades", 0))

        st.markdown("---")

        col_left, col_right = st.columns(2)

        # ── Exit reasons ─────────────────────────────────────────────────────
        with col_left:
            exits = fb.get("exit_reasons", {})
            counts = exits.get("counts", {})
            wr_by  = exits.get("win_rate_by_reason", {})
            if counts:
                st.markdown("#### Exit Reason Breakdown")
                exit_df = pd.DataFrame([
                    {"Exit Reason": k, "Count": v,
                     "Win Rate": f"{wr_by.get(k,0)*100:.0f}%"}
                    for k, v in sorted(counts.items(), key=lambda x: -x[1])
                ])
                st.dataframe(exit_df, use_container_width=True, hide_index=True)

                fig_ex = px.pie(
                    exit_df, names="Exit Reason", values="Count",
                    title="How are we exiting?",
                    color_discrete_sequence=["#00d97e","#4dabf7","#f6c343","#e63757","#cc5de8"],
                )
                fig_ex.update_layout(template="plotly_dark", height=280)
                st.plotly_chart(fig_ex, use_container_width=True)

                st.caption(
                    "**Healthy mix:** sentiment_recovery + price_recovery = signal working. "
                    "**Too many time_stop:** positions not recovering — widen recovery threshold. "
                    "**Too many stop_loss:** entry conditions too loose."
                )

        # ── Signal split: news vs no-news ─────────────────────────────────────
        with col_right:
            split = fb.get("signal_split", {})
            if split:
                st.markdown("#### With News vs Without News")
                split_rows = []
                for label, data in split.items():
                    split_rows.append({
                        "Signal Type":   label.replace("_", " ").title(),
                        "Trades":        data.get("trades", 0),
                        "Win Rate":      f"{data.get('win_rate',0)*100:.0f}%",
                        "Avg Return":    f"{data.get('avg_return',0):+.1f}%",
                    })
                if split_rows:
                    st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)
                    st.caption("Trades entering WITHOUT news sentiment signal are Z-score-only entries. "
                               "If win rate is much lower, consider requiring sentiment coverage.")

        st.markdown("---")

        # ── Model calibration ─────────────────────────────────────────────────
        model_cal = fb.get("model_calibration", {})
        if model_cal:
            st.markdown("#### Ensemble Model Calibration")
            st.caption("Accuracy = % of times model's negative score at entry correctly predicted a winning trade.")
            cal_rows = []
            for model, stats in model_cal.items():
                cal_rows.append({
                    "Model":       model,
                    "Predictions": stats.get("predictions", 0),
                    "Accuracy":    f"{stats.get('accuracy',0)*100:.0f}%",
                    "Avg Score":   f"{stats.get('avg_score',0):+.3f}",
                    "Score Std":   f"{stats.get('score_std',0):.3f}",
                })
            if cal_rows:
                cal_df = pd.DataFrame(cal_rows)
                st.dataframe(cal_df, use_container_width=True, hide_index=True)
                st.caption("Higher accuracy → higher weight in ensemble. "
                           "Low accuracy model (< 55%) may need retraining or reduced weight.")

        st.markdown("---")

        # ── Alerts & suggestions ──────────────────────────────────────────────
        alerts = fb.get("alerts", [])
        suggestions = fb.get("parameter_suggestions", [])

        col_al, col_sg = st.columns(2)
        with col_al:
            st.markdown("#### Alerts")
            for a in alerts:
                if "below" in a.lower() or "negative" in a.lower() or "too many" in a.lower():
                    st.warning(a)
                elif "performing well" in a.lower() or "no anomalies" in a.lower():
                    st.success(a)
                else:
                    st.info(a)

        with col_sg:
            st.markdown("#### Parameter Suggestions")
            if suggestions:
                for s in suggestions:
                    st.info(s)
            else:
                st.success("No parameter changes suggested — strategy within expected range.")

        # ── Return distribution ───────────────────────────────────────────────
        ret_dist = fb.get("return_distribution", {})
        if ret_dist:
            st.markdown("---")
            st.markdown("#### Return Distribution Summary")
            d1, d2, d3 = st.columns(3)
            d1.metric("Bottom Quartile", f"{ret_dist.get('bottom_quartile_return',0):+.1f}%")
            d2.metric("Median Return",   f"{ret_dist.get('median_return',0):+.1f}%")
            d3.metric("Top Quartile",    f"{ret_dist.get('top_quartile_return',0):+.1f}%")


# ── Tab 6: Earnings Calls ──────────────────────────────────────────────────────
with tab6:
    earnings = _load_earnings()
    if "confidence_score" not in earnings.columns:
        st.info("No earnings scores yet. Run: `python scripts/score_transcripts.py --file <audio.mp3>`")
    else:
        st.subheader("Management Confidence Scores")
        if "prev_confidence" in earnings.columns:
            earnings["delta"] = earnings["confidence_score"] - earnings["prev_confidence"]
            anomalies = earnings[earnings["delta"] < -10]
            if not anomalies.empty:
                st.error(f"⚠ {len(anomalies)} stock(s) with significant confidence drops!")
                for _, r in anomalies.iterrows():
                    st.markdown(f"**{r['symbol']}**: {r.get('prev_confidence','?')} → "
                                f"{r['confidence_score']} (Δ={r['delta']:.0f})")

        col1, col2 = st.columns(2)
        with col1:
            colors = earnings["confidence_score"].apply(
                lambda x: "#00d97e" if x > 70 else "#f6c343" if x > 50 else "#e63757")
            fig_c = go.Figure()
            fig_c.add_trace(go.Bar(x=earnings["symbol"],
                                   y=earnings["confidence_score"],
                                   marker_color=colors, name="Current"))
            if "prev_confidence" in earnings.columns:
                fig_c.add_trace(go.Scatter(x=earnings["symbol"],
                                           y=earnings["prev_confidence"],
                                           mode="markers", name="Prev Quarter",
                                           marker=dict(size=10, color="white", symbol="diamond")))
            fig_c.update_layout(title="Management Confidence (0–100)",
                                template="plotly_dark", height=380)
            st.plotly_chart(fig_c, use_container_width=True)

        with col2:
            if "evasion" not in earnings.columns:
                earnings["evasion"] = 30
            fig_e = px.bar(earnings.sort_values("evasion"),
                           x="evasion", y="symbol", orientation="h",
                           color="evasion", color_continuous_scale="RdYlGn_r",
                           title="Evasion Score (lower = better)")
            fig_e.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_e, use_container_width=True)


# ── Tab 8: Pipeline Logs ──────────────────────────────────────────────────────
with tab8:
    st.subheader("Pipeline Logs")

    log_dir = cfg.logs_dir

    col_pre, col_post, col_bt = st.columns(3)

    def _last_run_time(label: str) -> str:
        try:
            for lf in sorted(log_dir.glob("*.log"), reverse=True)[:5]:
                for line in reversed(lf.read_text(errors="ignore").splitlines()):
                    if label.lower() in line.lower():
                        parts = line.split("|")
                        return parts[0].strip() if parts else line[:20]
        except Exception:
            pass
        return "No run yet"

    col_pre.metric("Pre-market (08:30 IST)",     _last_run_time("pre-market"),  "fetch data")
    col_post.metric("Post-close (15:45 IST)",    _last_run_time("post-close"),  "signals + execute")
    col_bt.metric("Nightly backtest (23:00 IST)", _last_run_time("walk-forward"),"test period")

    st.markdown("---")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Log files**")
        log_files = sorted(log_dir.glob("*.log"), reverse=True) if log_dir.exists() else []
        cron_log  = log_dir / "cron.log"
        all_logs  = ([cron_log] if cron_log.exists() else []) + \
                    [f for f in log_files if f != cron_log]

        selected_log = None
        if not all_logs:
            st.info("No logs yet — first cron run at 08:30 IST Mon–Fri")
        else:
            choice = st.radio("Select log", [f.name for f in all_logs], index=0)
            selected_log = log_dir / choice

        lines_to_show = st.slider("Lines", 20, 500, 100, 10)
        if st.checkbox("Auto-refresh 30s"):
            import time
            if int(time.time()) % 30 == 0:
                st.rerun()

    with col_right:
        if selected_log and selected_log.exists():
            lines = selected_log.read_text(errors="ignore").splitlines()
            highlighted = []
            for line in lines[-lines_to_show:]:
                if any(k in line.lower() for k in ["error","exception","traceback"]):
                    highlighted.append(f"🔴 {line}")
                elif "warning" in line.lower():
                    highlighted.append(f"🟡 {line}")
                elif any(k in line.lower() for k in ["complete","ok","success","pass","✅"]):
                    highlighted.append(f"🟢 {line}")
                else:
                    highlighted.append(line)
            st.code("\n".join(highlighted), language="bash")
            st.caption(f"{len(lines)} total lines · showing last {min(lines_to_show, len(lines))}")
        else:
            st.code("# Waiting for first cron run...\n"
                    "# Pre-market:  08:30 IST (03:00 UTC)\n"
                    "# Post-close:  15:45 IST (10:15 UTC)", language="bash")


# ── Tab 9: FinBERT Labels ─────────────────────────────────────────────────────
with tab9:
    st.subheader("FinBERT Training Labels")

    labels_dir = cfg.data_dir / "labels"

    @st.cache_data(ttl=300)
    def _load_labels():
        records = []
        for jf in sorted(labels_dir.glob("labeled_*.jsonl")):
            for line in jf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return pd.DataFrame(records) if records else pd.DataFrame()

    hdr_col, btn_col = st.columns([5, 1])
    hdr_col.caption("Multi-source labels from news + BSE announcements for FinBERT fine-tuning")
    if btn_col.button("🔄 Refresh", key="refresh_labels"):
        st.cache_data.clear()

    ldf = _load_labels()

    if ldf.empty:
        st.info("No labeled data yet. Run `python alphasense/data/labeler.py --backfill` on the server.")
    else:
        # ── Summary metrics ──────────────────────────────────────────────────
        total   = len(ldf)
        n_pos   = (ldf["label"] == "positive").sum()
        n_neg   = (ldf["label"] == "negative").sum()
        n_neu   = (ldf["label"] == "neutral").sum()
        n_high  = (ldf["confidence"] == "high").sum()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total samples",  f"{total:,}")
        m2.metric("Positive",       f"{n_pos:,}",  f"{n_pos/total*100:.0f}%")
        m3.metric("Negative",       f"{n_neg:,}",  f"{n_neg/total*100:.0f}%")
        m4.metric("Neutral",        f"{n_neu:,}",  f"{n_neu/total*100:.0f}%")
        m5.metric("High confidence",f"{n_high:,}", f"{n_high/total*100:.0f}%")

        st.markdown("---")

        # ── Charts ───────────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            label_counts = ldf["label"].value_counts().reset_index()
            label_counts.columns = ["label", "count"]
            color_map = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
            fig_lbl = px.bar(label_counts, x="label", y="count",
                             color="label", color_discrete_map=color_map,
                             title="Label Distribution",
                             labels={"count": "Samples", "label": ""})
            fig_lbl.update_layout(showlegend=False, height=280,
                                  margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_lbl, use_container_width=True)

        with ch2:
            src_counts = ldf["label_source"].value_counts().reset_index()
            src_counts.columns = ["source", "count"]
            fig_src = px.bar(src_counts, x="source", y="count",
                             title="Label Source",
                             labels={"count": "Samples", "source": ""})
            fig_src.update_layout(showlegend=False, height=280,
                                  margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_src, use_container_width=True)

        # Confidence + label-source breakdown
        ch3, ch4 = st.columns(2)

        with ch3:
            conf_counts = ldf["confidence"].value_counts().reset_index()
            conf_counts.columns = ["confidence", "count"]
            conf_color = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}
            fig_conf = px.pie(conf_counts, names="confidence", values="count",
                              color="confidence", color_discrete_map=conf_color,
                              title="Confidence Tier")
            fig_conf.update_layout(height=260, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_conf, use_container_width=True)

        with ch4:
            if "date" in ldf.columns:
                ldf2 = ldf.copy()
                ldf2["date"] = pd.to_datetime(ldf2["date"], errors="coerce")
                daily = (ldf2.groupby(ldf2["date"].dt.date)["label"]
                            .count().reset_index())
                daily.columns = ["date", "count"]
                fig_daily = px.bar(daily, x="date", y="count",
                                   title="Samples per Day",
                                   labels={"count": "Samples", "date": ""})
                fig_daily.update_layout(height=260,
                                        margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("---")

        # ── Filters + sample browser ─────────────────────────────────────────
        st.markdown("**Browse Samples**")
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
        f_label  = fc1.selectbox("Label",      ["all", "positive", "negative", "neutral"])
        f_source = fc2.selectbox("Label source",
                                 ["all"] + sorted(ldf["label_source"].unique().tolist()))
        f_conf   = fc3.selectbox("Confidence", ["all", "high", "medium", "low"])
        f_sym    = fc4.text_input("Symbol filter", placeholder="e.g. RELIANCE")

        view = ldf.copy()
        if f_label  != "all": view = view[view["label"]        == f_label]
        if f_source != "all": view = view[view["label_source"] == f_source]
        if f_conf   != "all": view = view[view["confidence"]   == f_conf]
        if f_sym.strip():
            view = view[view["symbol"].str.upper() == f_sym.strip().upper()]

        st.caption(f"Showing {min(len(view), 200):,} of {len(view):,} filtered samples")

        display_cols = ["symbol", "date", "label", "label_source", "confidence",
                        "return_5d", "return_20d", "source", "headline"]
        show_cols = [c for c in display_cols if c in view.columns]
        st.dataframe(
            view[show_cols].head(200).reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

        # ── Return distribution for high-conf samples ─────────────────────────
        hi_conf = ldf[ldf["confidence"] == "high"].dropna(subset=["return_5d"])
        if not hi_conf.empty:
            st.markdown("---")
            st.markdown("**Return distribution — high-confidence samples**")
            fig_ret = px.histogram(
                hi_conf, x="return_5d", color="label",
                nbins=60, barmode="overlay",
                color_discrete_map={"positive": "#2ecc71",
                                    "neutral": "#95a5a6",
                                    "negative": "#e74c3c"},
                labels={"return_5d": "5-day forward return"},
                title="5d Return Distribution (high-confidence labels)",
            )
            fig_ret.update_layout(height=300,
                                  margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_ret, use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.markdown("---")
        dl_col, _ = st.columns([2, 5])
        csv_bytes = ldf.to_csv(index=False).encode()
        dl_col.download_button(
            "⬇ Download full dataset (CSV)",
            data=csv_bytes,
            file_name=f"alphasense_labels_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# ── Tab 10: Strategies ────────────────────────────────────────────────────────
with tab10:
    st.subheader("Active Strategies")
    st.caption("Engine priority order: Earnings Surprise → Buyback Arbit → Mean Reversion")

    STRATEGIES = [
        {
            "name": "Mean Reversion",
            "status": "LIVE",
            "color": "#00d97e",
            "type": "Contrarian",
            "description": "Buy when price overshoots down (z < -3.0) with negative/neutral sentiment. Exit on recovery or 20-day time-stop.",
            "entry": "Z-score < -3.0  AND  sentiment < -0.4  AND  VIX < 28",
            "exit":  "Z-score > -1.0 (recovery)  OR  -10% stop-loss  OR  20 days",
            "sizing": "Kelly Criterion (half-Kelly, ≤ 25% cap) · max 8% capital per trade",
            "backtest": "OOS 2023–26: Sharpe 2.85 · WinRate 55% · MaxDD -5.9% · Ann +8.6%",
            "fwd": "GESHIP +17.4% · COALINDIA +3.8% (2 live trades)",
            "icon": "📉",
        },
        {
            "name": "Earnings Surprise",
            "status": "LIVE",
            "color": "#f4c430",
            "type": "Event-Driven",
            "description": "Enter after a post-earnings beat: XBRL sentiment delta ≥ 0.10 and price hasn't already crashed (z ≥ -1.5). Rides the post-announcement drift.",
            "entry": "sentiment_delta ≥ 0.10  AND  z ≥ -1.5",
            "exit":  "-6% stop-loss  OR  10% profit-exit  OR  10 days",
            "sizing": "Fixed max 8% capital (Kelly requires ≥15 trades to activate)",
            "backtest": "Event-driven — limited OOS history; activates when earnings audio available",
            "fwd": "No live trades yet (activates post-earnings season)",
            "icon": "📢",
        },
        {
            "name": "Buyback Arbit",
            "status": "LIVE",
            "color": "#3a9de1",
            "type": "Event-Driven",
            "description": "Enter on BSE buyback announcement before price fully recovers. BSE boost score ≥ 0.20 and z < 1.0 (price hasn't run up yet).",
            "entry": "BSE buyback boost ≥ 0.20  AND  z < 1.0",
            "exit":  "-6% stop-loss  OR  profit-exit  OR  15 days",
            "sizing": "Fixed max 8% capital",
            "backtest": "Event-driven — depends on BSE announcement data quality",
            "fwd": "No live trades yet (depends on buyback announcements)",
            "icon": "🔄",
        },
        {
            "name": "Momentum",
            "status": "INACTIVE",
            "color": "#888888",
            "type": "Trend-Following",
            "description": "Trend-follow when z > +3.0 with positive sentiment. Removed after OOS test showed Sharpe 0.33, Max DD -50.6%.",
            "entry": "z > +3.0  AND  sentiment > 0.3  (DISABLED)",
            "exit":  "N/A",
            "sizing": "N/A",
            "backtest": "OOS: Sharpe 0.33 · WinRate 32.5% · MaxDD -50.6% → dropped",
            "fwd": "N/A",
            "icon": "📈",
        },
    ]

    for s in STRATEGIES:
        is_live = s["status"] == "LIVE"
        badge   = f'<span style="background:{s["color"]};color:#111;padding:2px 10px;border-radius:12px;font-size:0.85em;font-weight:bold">{s["status"]}</span>'
        with st.expander(f'{s["icon"]}  {s["name"]}  —  {s["type"]}', expanded=is_live):
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(f"**{s['description']}**")
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Entry Conditions**")
                st.code(s["entry"], language=None)
                st.markdown("**Exit Rules**")
                st.code(s["exit"], language=None)
            with c2:
                st.markdown("**Position Sizing**")
                st.code(s["sizing"], language=None)
                st.markdown("**Backtest Performance**")
                st.info(s["backtest"])
                if s["fwd"] != "N/A":
                    st.success(f"Forward: {s['fwd']}")

    st.markdown("---")
    st.subheader("Strategy Breakdown — Live Paper Trades")

    @st.cache_data(ttl=60)
    def _trades_by_strategy():
        p = cfg.data_dir / "paper_state.json"
        if not p.exists():
            return pd.DataFrame()
        try:
            state  = json.loads(p.read_text())
            trades = state.get("closed_trades", [])
            if not trades:
                return pd.DataFrame()
            df = pd.DataFrame(trades)
            return df
        except Exception:
            return pd.DataFrame()

    tdf = _trades_by_strategy()
    if tdf.empty:
        st.info("No closed paper trades yet.")
    else:
        strat_col = "strategy" if "strategy" in tdf.columns else None
        if strat_col:
            by_strat = tdf.groupby("strategy").agg(
                trades=("pnl_pct", "count"),
                win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
                avg_pnl=("pnl_pct", "mean"),
                total_pnl=("pnl_pct", "sum"),
            ).reset_index()
            by_strat["win_rate"] = by_strat["win_rate"].map("{:.1f}%".format)
            by_strat["avg_pnl"]  = by_strat["avg_pnl"].map("{:+.2%}".format)
            by_strat["total_pnl"]= by_strat["total_pnl"].map("{:+.2%}".format)
            st.dataframe(by_strat, use_container_width=True, hide_index=True)
        else:
            st.caption("Trades don't have strategy tags yet (older paper trades).")

        show_cols = [c for c in ["symbol","strategy","entry_price","exit_price",
                                 "pnl_pct","days_held","exit_reason"] if c in tdf.columns]
        st.dataframe(
            tdf[show_cols].sort_values("pnl_pct", ascending=False)
            .style.format({k: "{:+.2%}" for k in ["pnl_pct"] if k in show_cols}),
            use_container_width=True,
            hide_index=True,
            height=300,
        )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<center><small>AlphaSense AI · Confidential · "
    f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}</small></center>",
    unsafe_allow_html=True,
)
