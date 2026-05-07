"""
XBRL / Fundamental Data Client
================================
Fetches quarterly financials via yfinance and computes signal-ready
fundamental indicators for every stock in the universe.

BSE direct XBRL API is blocked (returns 301), so we use yfinance which
provides the same data via Yahoo Finance's structured endpoints.

Computed signals per stock:
  • revenue_growth_qoq   — % change last Q vs prior Q
  • ebitda_margin_trend  — last Q margin minus 4Q rolling avg (pp)
  • debt_equity_dir      — "rising" / "falling" / "stable"
  • fcf_quality          — FCF / Net Income  (>1.0 = high quality)
  • overall_score        — [-1, 0, +1]  (-1 = caution, +1 = supportive of BUY)

Stored at: data/xbrl/{symbol}.json
           data/xbrl/_universe_summary.json  (all symbols in one file for fast load)

Usage:
    python alphasense/data/xbrl_client.py --symbol RELIANCE
    python alphasense/data/xbrl_client.py --universe
"""

import json, sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

XBRL_DIR = cfg.data_dir / "xbrl"
XBRL_DIR.mkdir(parents=True, exist_ok=True)

STALE_DAYS = 7   # refresh if file is older than this


# ── yfinance helpers ──────────────────────────────────────────────────────────

def _fetch_yf(symbol: str) -> Optional[dict]:
    """Fetch quarterly financials from Yahoo Finance for `symbol.NS`."""
    try:
        import yfinance as yf
        t   = yf.Ticker(f"{symbol}.NS")
        fin = t.quarterly_financials       # columns = quarters (most recent first)
        bs  = t.quarterly_balance_sheet
        cf  = t.quarterly_cashflow
        return {"fin": fin, "bs": bs, "cf": cf}
    except Exception as e:
        logger.warning(f"yfinance failed for {symbol}: {e}")
        return None


def _safe_row(df: pd.DataFrame, *row_names) -> Optional[pd.Series]:
    """Return first matching row from a DataFrame (case-insensitive)."""
    if df is None or df.empty:
        return None
    for name in row_names:
        for idx in df.index:
            if isinstance(idx, str) and name.lower() in idx.lower():
                return df.loc[idx]
    return None


def _pct_change(series: pd.Series, i: int = 0) -> Optional[float]:
    """% change from position i+1 to i (most-recent = 0)."""
    vals = series.dropna()
    if len(vals) <= i + 1:
        return None
    a, b = float(vals.iloc[i]), float(vals.iloc[i + 1])
    if b == 0:
        return None
    return (a - b) / abs(b)


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(symbol: str, data: dict) -> dict:
    fin, bs, cf = data["fin"], data["bs"], data["cf"]

    result: dict = {
        "symbol":             symbol,
        "refreshed_at":       datetime.now().isoformat(),
        "quarters_available": 0,
        "revenue_growth_qoq": None,
        "ebitda_margin_trend": None,
        "debt_equity_dir":    "unknown",
        "fcf_quality":        None,
        "overall_score":      0,
        "detail":             {},
    }

    # ── Revenue growth ────────────────────────────────────────────────────────
    rev = _safe_row(fin, "Total Revenue", "Revenue", "Net Revenue", "Total Sales")
    if rev is not None:
        result["quarters_available"] = int(rev.dropna().count())
        g = _pct_change(rev)
        if g is not None:
            result["revenue_growth_qoq"] = round(g * 100, 2)
            result["detail"]["revenue_last_q"] = round(float(rev.dropna().iloc[0]) / 1e7, 2)  # ₹Cr
            result["detail"]["revenue_prior_q"] = round(float(rev.dropna().iloc[1]) / 1e7, 2) if len(rev.dropna()) > 1 else None

    # ── EBITDA margin trend ───────────────────────────────────────────────────
    ebitda = _safe_row(fin, "EBITDA", "Operating Income", "EBIT")
    if rev is not None and ebitda is not None:
        rev_v  = rev.dropna()
        eb_v   = ebitda.dropna()
        common = rev_v.index.intersection(eb_v.index)
        if len(common) >= 2:
            margins = (eb_v[common] / rev_v[common]).dropna()
            last_q  = float(margins.iloc[0])
            avg_4q  = float(margins.iloc[:4].mean()) if len(margins) >= 4 else float(margins.mean())
            trend   = (last_q - avg_4q) * 100
            result["ebitda_margin_trend"] = round(trend, 2)
            result["detail"]["ebitda_margin_last_q"] = round(last_q * 100, 1)

    # ── Debt / equity direction ───────────────────────────────────────────────
    debt = _safe_row(bs, "Total Debt", "Long Term Debt", "Total Long Term Debt", "Net Debt")
    eq   = _safe_row(bs, "Stockholders Equity", "Total Equity", "Common Stock Equity")
    if debt is not None and eq is not None:
        d_v = debt.dropna()
        e_v = eq.dropna()
        common = d_v.index.intersection(e_v.index)
        if len(common) >= 2:
            de_ratio = d_v[common] / e_v[common].replace(0, np.nan)
            de_ratio = de_ratio.dropna()
            if len(de_ratio) >= 2:
                chg = float(de_ratio.iloc[0]) - float(de_ratio.iloc[1])
                if chg > 0.05:
                    result["debt_equity_dir"] = "rising"
                elif chg < -0.05:
                    result["debt_equity_dir"] = "falling"
                else:
                    result["debt_equity_dir"] = "stable"
                result["detail"]["de_ratio_last_q"] = round(float(de_ratio.iloc[0]), 2)

    # ── FCF quality ───────────────────────────────────────────────────────────
    ocf   = _safe_row(cf, "Operating Cash Flow", "Cash From Operations", "Total Cash From Operating")
    capex = _safe_row(cf, "Capital Expenditure", "Capex", "Purchase Of Property Plant And Equipment")
    ni    = _safe_row(fin, "Net Income", "Net Income Common Stockholders")
    if ocf is not None and ni is not None:
        ocf_v = ocf.dropna()
        ni_v  = ni.dropna()
        cap_v = capex.dropna() if capex is not None else pd.Series([], dtype=float)
        common = ocf_v.index.intersection(ni_v.index)
        if common.empty:
            pass
        else:
            ocf_q  = float(ocf_v[common[0]])
            ni_q   = float(ni_v[common[0]])
            cap_q  = float(cap_v[cap_v.index.intersection(common[:1])[0]]) \
                     if not cap_v.index.intersection(common[:1]).empty else 0.0
            fcf    = ocf_q - abs(cap_q)
            if ni_q != 0:
                result["fcf_quality"] = round(fcf / abs(ni_q), 2)
            result["detail"]["fcf_cr"] = round(fcf / 1e7, 2)
            result["detail"]["net_income_cr"] = round(ni_q / 1e7, 2)

    # ── Overall score [-1, 0, +1] ─────────────────────────────────────────────
    # +1 signals that fundamentals support a BUY (used to relax thresholds)
    # -1 signals caution (used to tighten thresholds or block)
    score = 0
    g  = result["revenue_growth_qoq"]
    mt = result["ebitda_margin_trend"]
    de = result["debt_equity_dir"]
    fq = result["fcf_quality"]

    positives = 0
    negatives = 0

    if g is not None:
        if g > 5:   positives += 1
        elif g < -5: negatives += 1

    if mt is not None:
        if mt > 1:   positives += 1
        elif mt < -2: negatives += 1

    if de == "rising":    negatives += 1
    elif de == "falling": positives += 1

    if fq is not None:
        if fq > 1.2:  positives += 1
        elif fq < 0.5: negatives += 1

    if positives >= 3 and negatives == 0:
        score = 1
    elif negatives >= 2:
        score = -1

    result["overall_score"] = score
    return result


# ── I/O ───────────────────────────────────────────────────────────────────────

def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).days >= STALE_DAYS


def save_signals(signals: dict):
    path = XBRL_DIR / f"{signals['symbol']}.json"
    path.write_text(json.dumps(signals, indent=2, ensure_ascii=False))


def load_signals(symbol: str) -> Optional[dict]:
    path = XBRL_DIR / f"{symbol}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── Refresh logic ─────────────────────────────────────────────────────────────

def refresh_symbol(symbol: str, force: bool = False) -> Optional[dict]:
    path = XBRL_DIR / f"{symbol}.json"
    if not force and not _is_stale(path):
        logger.debug(f"XBRL {symbol}: cached (< {STALE_DAYS}d old)")
        return load_signals(symbol)

    data = _fetch_yf(symbol)
    if not data:
        return None

    signals = compute_signals(symbol, data)
    save_signals(signals)
    logger.info(f"XBRL {symbol}: rev_growth={signals.get('revenue_growth_qoq')}% "
                f"margin_trend={signals.get('ebitda_margin_trend')} "
                f"D/E={signals.get('debt_equity_dir')} "
                f"FCF={signals.get('fcf_quality')} "
                f"score={signals.get('overall_score')}")
    return signals


def refresh_universe(force: bool = False) -> dict[str, dict]:
    """Refresh XBRL for all symbols in the universe parquet."""
    try:
        from alphasense.screener.fundamental import load_universe
        universe = load_universe()
        if universe.empty:
            logger.warning("Universe empty — run build_universe.py first")
            return {}
        symbols = universe["symbol"].tolist()
    except Exception as e:
        logger.error(f"Could not load universe: {e}")
        return {}

    results = {}
    for i, sym in enumerate(symbols):
        logger.info(f"  [{i+1}/{len(symbols)}] {sym}")
        sig = refresh_symbol(sym, force=force)
        if sig:
            results[sym] = sig
        time_sleep = 0.5   # polite rate-limit for yfinance
        import time; time.sleep(time_sleep)

    # Save universe summary for fast batch loading
    summary_path = XBRL_DIR / "_universe_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"XBRL universe refresh: {len(results)}/{len(symbols)} symbols updated")
    return results


def load_universe_summary() -> dict[str, dict]:
    """Load pre-computed universe summary (fast — single file read)."""
    path = XBRL_DIR / "_universe_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",   help="Single NSE symbol")
    p.add_argument("--universe", action="store_true", help="Refresh all universe stocks")
    p.add_argument("--force",    action="store_true", help="Ignore cache age")
    args = p.parse_args()

    if args.symbol:
        result = refresh_symbol(args.symbol.upper(), force=args.force)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print(f"No data returned for {args.symbol}")
    elif args.universe:
        refresh_universe(force=args.force)
    else:
        p.print_help()
