"""
Fundamental Universe Screener
==============================
Two-tier filter on NIFTY 500 to produce an eligible trading universe.

Tier 1 — Hard Eliminators (any failure = excluded):
  • ROE > 15% for 3+ consecutive years
  • Debt-to-equity < 1.0
  • Promoter holding > 30%
  • No qualified audit report
  • No SEBI enforcement in last 24 months

Tier 2 — Quality Score (0–100, weighted composite):
  • ROCE trend        25%
  • FCF conversion    30%
  • Accruals ratio    25%
  • Revenue concentration 20%

If full fundamental data is unavailable (common bootstrap situation),
falls back to price-proxy scoring from downloaded OHLCV data.

Run:
    python scripts/build_universe.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg


# ─── Price-proxy universe (bootstrap / always available) ─────────────────────

def build_price_proxy_universe() -> pd.DataFrame:
    """
    Build a quality universe from price data alone.
    Used when fundamental data is not yet available.
    Filters: price > ₹50, liquidity, volatility, trend, 2yr return.
    """
    from alphasense.data.nse_client import load_prices

    const_path = cfg.nse_dir / "nifty500_constituents.parquet"
    if not const_path.exists():
        logger.error("No NIFTY 500 constituents. Run fetch_prices --constituents first.")
        return pd.DataFrame()

    constituents = pd.read_parquet(const_path)
    symbols      = constituents["symbol"].tolist()
    prices       = load_prices(symbols)
    logger.info(f"Building universe from {len(prices)} stocks...")

    results = []
    for sym, df in prices.items():
        if "close" not in df.columns or len(df) < 500:
            continue
        close  = df["close"]
        volume = df.get("volume", pd.Series([0]*len(df), index=df.index))

        current  = close.iloc[-1]
        if current < 50:               continue  # penny stock

        avg_vol      = volume.tail(60).mean()
        avg_turnover = avg_vol * current
        if avg_turnover < 5_000_000:   continue  # < ₹50L daily liquidity

        daily_ret    = close.pct_change().dropna()
        annual_vol   = daily_ret.tail(252).std() * np.sqrt(252)
        if annual_vol > 0.60:          continue  # too volatile

        ma200        = close.rolling(200).mean().iloc[-1]
        vs_ma200     = (current - ma200) / ma200
        if vs_ma200 < -0.30:           continue  # structural decline

        ret_2y = (close.iloc[-1] - close.iloc[-500]) / close.iloc[-500] if len(close) > 500 else 0
        if ret_2y < -0.20:             continue  # declining business

        # Score
        vol_score       = max(0, min(30, (1 - annual_vol) * 50))
        trend_score     = max(0, min(30, (vs_ma200 + 0.1) * 100))
        return_score    = max(0, min(25, ret_2y * 50))
        liquidity_score = max(0, min(15, (np.log10(max(avg_turnover, 1)) - 5) * 5))
        quality_score   = vol_score + trend_score + return_score + liquidity_score

        results.append({
            "symbol":                sym,
            "current_price":         round(current, 2),
            "avg_daily_turnover_cr": round(avg_turnover / 1e7, 2),
            "annual_volatility":     round(annual_vol, 3),
            "price_vs_200dma":       round(vs_ma200, 3),
            "return_2y":             round(ret_2y, 3),
            "quality_score":         round(quality_score, 1),
        })

    df_out = pd.DataFrame(results).sort_values("quality_score", ascending=False) \
                                  .reset_index(drop=True)
    logger.info(f"Price-proxy universe: {len(df_out)} stocks (from {len(symbols)} NIFTY 500)")
    return df_out


# ─── Full fundamental screener ────────────────────────────────────────────────

def _tier1_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply hard eliminators. Returns stocks passing ALL Tier 1 checks."""
    sc = cfg.screener
    n0 = df["symbol"].nunique()

    if "roe" in df.columns:
        passing = df.groupby("symbol")["roe"].apply(
            lambda x: (x >= sc.min_roe).sum() >= sc.min_roe_years
        )
        df = df[df["symbol"].isin(passing[passing].index)]
        logger.debug(f"  After ROE: {df['symbol'].nunique()}")

    if "debt_to_equity" in df.columns:
        latest  = df.sort_values("filing_date").groupby("symbol").tail(1)
        passing = latest[latest["debt_to_equity"] < sc.max_debt_equity]["symbol"]
        df = df[df["symbol"].isin(passing)]
        logger.debug(f"  After D/E: {df['symbol'].nunique()}")

    if "promoter_holding_pct" in df.columns:
        latest  = df.sort_values("filing_date").groupby("symbol").tail(1)
        valid   = latest.dropna(subset=["promoter_holding_pct"])
        passing = valid[valid["promoter_holding_pct"] >= sc.min_promoter_holding]["symbol"]
        if len(passing) > 0:
            df = df[df["symbol"].isin(passing)]
            logger.debug(f"  After promoter: {df['symbol'].nunique()}")

    if "audit_qualified" in df.columns:
        bad = df[df["audit_qualified"] == True]["symbol"].unique()
        df  = df[~df["symbol"].isin(bad)]

    if "sebi_action" in df.columns:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
        bad    = df[(df["filing_date"] >= cutoff) & (df["sebi_action"] == True)]["symbol"].unique()
        df     = df[~df["symbol"].isin(bad)]

    logger.info(f"Tier 1: {n0} → {df['symbol'].nunique()} stocks")
    return df


def _tier2_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted quality score 0–100."""
    sc     = cfg.screener
    latest = df.sort_values("filing_date").groupby("symbol").tail(1).copy()
    scores = pd.DataFrame({"symbol": latest["symbol"].values})

    # ROCE trend
    if "roce" in df.columns:
        def _roce_trend(g):
            if len(g) < 4:
                return 50
            r0, r4 = g.iloc[-1]["roce"], g.iloc[-4]["roce"]
            return min(100, max(0, 50 + ((r0 - r4) / (r4 + 1e-9)) * 200)) if r4 else 50
        rt = df.groupby("symbol").apply(_roce_trend).reset_index()
        rt.columns = ["symbol", "roce_score"]
        scores = scores.merge(rt, on="symbol", how="left")
        scores["roce_score"] = scores["roce_score"].fillna(50)
    else:
        scores["roce_score"] = 50

    # FCF conversion
    if "operating_cash_flow" in latest.columns and "net_profit" in latest.columns:
        latest["fcf_conv"] = latest["operating_cash_flow"] / latest["net_profit"].replace(0, np.nan)
        m = dict(zip(latest["symbol"], latest["fcf_conv"]))
        scores["fcf_score"] = scores["symbol"].map(m).apply(
            lambda x: min(100, max(0, x * 100)) if pd.notna(x) and x > 0 else 30
        )
    else:
        scores["fcf_score"] = 50

    # Accruals (lower = better)
    if "accruals_ratio" in latest.columns:
        m = dict(zip(latest["symbol"], latest["accruals_ratio"]))
        scores["accruals_score"] = scores["symbol"].map(m).apply(
            lambda x: min(100, max(0, (1 - x * 5) * 100)) if pd.notna(x) else 50
        )
    else:
        scores["accruals_score"] = 50

    # Revenue concentration (lower = better)
    if "revenue_top_customer_pct" in latest.columns:
        m = dict(zip(latest["symbol"], latest["revenue_top_customer_pct"]))
        scores["concentration_score"] = scores["symbol"].map(m).apply(
            lambda x: min(100, max(0, (1 - x) * 100)) if pd.notna(x) else 70
        )
    else:
        scores["concentration_score"] = 70

    scores["quality_score"] = (
        sc.w_roce          * scores["roce_score"]
        + sc.w_fcf         * scores["fcf_score"]
        + sc.w_accruals    * scores["accruals_score"]
        + sc.w_concentration * scores["concentration_score"]
    ).round(1)
    return scores


def run_fundamental_screener(fundamentals_path: Path) -> pd.DataFrame:
    """
    Full pipeline: load fundamentals → Tier 1 → Tier 2 → save universe.
    fundamentals_path: parquet with columns symbol, filing_date, roe,
                       debt_to_equity, promoter_holding_pct, roce, etc.
    """
    if not fundamentals_path.exists():
        logger.warning(f"No fundamentals at {fundamentals_path}. Falling back to price proxy.")
        return build_and_save_universe()

    df = pd.read_parquet(fundamentals_path)
    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"])

    filtered = _tier1_filter(df)
    scores   = _tier2_score(filtered)
    scores["in_universe"] = True
    scores = scores.sort_values("quality_score", ascending=False).reset_index(drop=True)
    logger.info(f"Fundamental universe: {len(scores)} stocks")
    return scores


def build_and_save_universe() -> pd.DataFrame:
    """Entry point: build universe (price proxy), save parquet."""
    universe = build_price_proxy_universe()
    out = cfg.data_dir / "universe.parquet"
    universe.to_parquet(out, index=False)
    logger.success(f"Universe saved: {out} ({len(universe)} stocks)")
    return universe


def load_universe() -> pd.DataFrame:
    path = cfg.data_dir / "universe.parquet"
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("No universe file. Run build_universe.py first.")
    return pd.DataFrame()