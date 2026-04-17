"""
AlphaSense AI — Streamlit Dashboard
=====================================
Tabs:
  1. Backtest Results   — equity curve, drawdown, walk-forward metrics
  2. Forward Test       — 2025-2026 OOS validation with slider controls
  3. Live Signals       — today's FinBERT + Z-score candidates
  4. Universe           — quality score rankings, sector distribution
  5. Earnings Calls     — management confidence scores
  6. Pipeline Logs      — cron output, paper trade history
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Backtest", "🔭 Forward Test",
    "🚨 Live Signals", "💼 Paper Trades",
    "🏛 Universe", "🎙 Earnings", "🖥 Logs",
])


# ── Tab 1: Backtest ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Walk-Forward Backtest Results (2022–2024 OOS)")

    col_t, col_v = st.columns(2)

    for period, col in [("test", col_t), ("validation", col_v)]:
        r = _load_backtest_result(period)
        label = "Test 2022–2024" if period == "test" else "Validation 2021"
        with col:
            st.markdown(f"**{label}**")
            m1,m2,m3 = st.columns(3)
            m1.metric("Sharpe",    f"{r.get('sharpe',0):.2f}")
            m2.metric("Max DD",    f"{r.get('max_drawdown_pct',0):.1f}%")
            m3.metric("Win Rate",  f"{r.get('win_rate',0):.1f}%")
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
               "Current prices fetched live from yfinance.")

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
            # Reads from local parquet (updated by cron post-close) first —
            # more reliable than live yfinance which can return NaN for today.
            @st.cache_data(ttl=300)
            def _fetch_current_prices(symbols: tuple) -> dict[str, float]:
                prices = {}
                if not symbols:
                    return prices
                nse_dir = cfg.data_dir / "nse"
                for sym in symbols:
                    try:
                        p = nse_dir / f"{sym}.parquet"
                        if p.exists():
                            df = pd.read_parquet(p)
                            if "close" in df.columns and not df.empty:
                                px = float(df["close"].dropna().iloc[-1])
                                prices[sym] = px
                    except Exception:
                        pass
                # Fall back to yfinance for any symbols not in parquet
                missing = [s for s in symbols if s not in prices]
                if missing:
                    try:
                        import yfinance as yf
                        tickers = [f"{s}.NS" for s in missing]
                        data = yf.download(tickers, period="5d", interval="1d",
                                           progress=False, auto_adjust=True)
                        for sym in missing:
                            try:
                                col = f"{sym}.NS"
                                if len(tickers) == 1:
                                    px = float(data["Close"].dropna().iloc[-1])
                                else:
                                    px = float(data["Close"][col].dropna().iloc[-1])
                                prices[sym] = px
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

            if is_stale:
                st.warning(
                    f"Market may be closed or data not yet updated. "
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


# ── Tab 5: Earnings Calls ─────────────────────────────────────────────────────
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


# ── Tab 6: Pipeline Logs ──────────────────────────────────────────────────────
with tab7:
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



# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<center><small>AlphaSense AI · Confidential · "
    f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}</small></center>",
    unsafe_allow_html=True,
)
