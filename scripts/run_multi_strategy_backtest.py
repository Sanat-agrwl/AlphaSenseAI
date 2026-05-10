"""
Multi-Strategy Walk-Forward Backtest
======================================
Tests 7 strategies on NSE price data using proper walk-forward:

  TRAIN  2016-01-01 → 2021-12-31   (6 years — param selection only)
  VAL    2022-01-01 → 2022-12-31   (confirm params generalise)
  TEST   2023-01-01 → 2026-04-30   (unbiased OOS — never seen before)

Strategies:
  1. MEAN_REVERSION_ENHANCED  — z < thresh + near 52w low + quality gate
  2. VIX_SPIKE_BUY            — India VIX 1-day jump > threshold → buy oversold
  3. SECTOR_MR                — sector-level z < thresh → buy quality leaders
  4. QUALITY_MOMENTUM         — z > +thresh AND quality ≥ 80 AND 20d high
  5. CAPITULATION_REVERSAL    — 5+ down days + volume spike + z < -2
  6. BOLLINGER_SQUEEZE        — BB squeeze then breakout above upper band
  7. GAP_FADE                 — gap-down > 3% in quality stock → fade the gap

Each strategy is: swept on TRAIN, locked on VAL, reported on TEST.
Survivors (Sharpe > 1.5, WR > 50%, MaxDD > -15%) are flagged for activation.

Run:
    python scripts/run_multi_strategy_backtest.py
    python scripts/run_multi_strategy_backtest.py --strategy vix_spike_buy
    python scripts/run_multi_strategy_backtest.py --save
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg
from alphasense.signal.engine import rolling_zscore

# ── Walk-forward windows ───────────────────────────────────────────────────────
TRAIN_START, TRAIN_END = "2016-01-01", "2021-12-31"
VAL_START,   VAL_END   = "2022-01-01", "2022-12-31"
TEST_START,  TEST_END  = "2023-01-01", "2026-04-30"

INITIAL_CAPITAL = 1_000_000
COST_BPS        = 50
PCT_PER_TRADE   = 0.08
MIN_HISTORY     = 400    # rows needed before a symbol is usable

# Thresholds
PASS_SHARPE  = 1.5
PASS_WR      = 50.0
PASS_MAXDD   = -15.0
PASS_TRADES  = 10       # min OOS trades to be meaningful


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_prices():
    from alphasense.data.nse_client import load_prices
    try:
        from alphasense.screener.fundamental import load_universe
        u    = load_universe()
        syms = u["symbol"].tolist() if not u.empty else None
    except Exception:
        syms = None
    prices = load_prices(syms)
    logger.info(f"Loaded {len(prices)} price series")
    return prices


def _load_vix() -> pd.Series:
    from alphasense.data.nse_client import load_vix
    vdf = load_vix()
    if vdf.empty:
        return pd.Series(dtype=float)
    vdf.index = pd.to_datetime(vdf["date"])
    return vdf["vix_close"]


def _load_quality() -> dict:
    """Return symbol → quality_score mapping."""
    p = cfg.data_dir / "universe.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if "quality_score" in df.columns and "symbol" in df.columns:
            return dict(zip(df["symbol"], df["quality_score"]))
    return {}


def _load_sector() -> dict:
    """Return symbol → sector mapping."""
    p = cfg.data_dir / "nse" / "constituents.csv"
    if p.exists():
        df = pd.read_csv(p)
        col = next((c for c in df.columns if "sector" in c.lower()), None)
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), "symbol")
        if col:
            return dict(zip(df[sym_col], df[col]))
    return {}


def _build_indexed(prices: dict) -> dict:
    out = {}
    for sym, df in prices.items():
        d = df.sort_values("date").copy()
        d.index = pd.to_datetime(d["date"])
        out[sym] = d
    return out


def _precompute_z(price_idx: dict, window: int = 252, period: int = 5) -> dict:
    zs = {}
    for sym, df in price_idx.items():
        if len(df) < MIN_HISTORY:
            continue
        z = rolling_zscore(df["close"], window, period)
        z.index = df.index
        zs[sym] = z
    logger.info(f"Z-scores precomputed for {len(zs)} symbols")
    return zs


def _precompute_52w(price_idx: dict) -> dict:
    """Return {sym: Series of 52w-low proximity (0=at low, 1=at high)}."""
    res = {}
    for sym, df in price_idx.items():
        if len(df) < 260:
            continue
        lo  = df["close"].rolling(252, min_periods=200).min()
        hi  = df["close"].rolling(252, min_periods=200).max()
        rng = (hi - lo).replace(0, np.nan)
        prox = (df["close"] - lo) / rng   # 0 = at 52w low, 1 = at 52w high
        prox.index = df.index
        res[sym] = prox
    return res


def _precompute_bb(price_idx: dict, window: int = 20, n_std: float = 2.0) -> dict:
    """Return {sym: DataFrame with bb_upper, bb_lower, bb_width, squeeze}."""
    res = {}
    for sym, df in price_idx.items():
        if len(df) < window + 10:
            continue
        mid   = df["close"].rolling(window).mean()
        std   = df["close"].rolling(window).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        width = (upper - lower) / mid.replace(0, np.nan)
        # squeeze = width below its own 60-day median
        squeeze = width < width.rolling(60, min_periods=40).median()
        bdf = pd.DataFrame({
            "bb_upper": upper, "bb_lower": lower,
            "bb_width": width, "squeeze": squeeze,
        }, index=df.index)
        res[sym] = bdf
    return res


# ── Trade simulation ───────────────────────────────────────────────────────────

def _sim_trade(sym: str, entry_date: pd.Timestamp, entry_price: float,
               price_idx: dict, stop_loss: float,
               time_stop: int, profit_exit: float,
               z_series: pd.Series = None,
               z_exit: float = None) -> dict | None:
    df     = price_idx.get(sym)
    if df is None:
        return None
    future = df[df.index > entry_date].head(time_stop + 5)
    if future.empty:
        return None

    exit_price  = float(future["close"].iloc[-1])
    exit_reason = "time_stop"
    days_held   = len(future)
    exit_dt     = future.index[-1]

    for i, (dt, row) in enumerate(future.iterrows()):
        close = float(row["close"])
        pnl   = (close - entry_price) / entry_price
        if pnl <= stop_loss:
            exit_price, exit_reason, days_held, exit_dt = close, "stop_loss", i+1, dt
            break
        if pnl >= profit_exit:
            exit_price, exit_reason, days_held, exit_dt = close, "profit_exit", i+1, dt
            break
        # z-score recovery exit (for MR strategies)
        if z_series is not None and z_exit is not None:
            z_now = z_series.asof(dt) if not pd.isna(z_series.asof(dt)) else None
            if z_now is not None and z_now >= z_exit:
                exit_price, exit_reason, days_held, exit_dt = close, "z_recovery", i+1, dt
                break
        if i + 1 >= time_stop:
            exit_price, exit_reason, days_held, exit_dt = close, "time_stop", i+1, dt
            break

    cost    = COST_BPS / 10000
    pnl_net = (exit_price - entry_price) / entry_price - cost
    return {
        "symbol":      sym,
        "entry_date":  str(entry_date.date()),
        "exit_date":   str(exit_dt.date()),
        "entry_price": round(entry_price, 2),
        "exit_price":  round(exit_price, 2),
        "pnl_pct":     round(pnl_net, 4),
        "days_held":   days_held,
        "exit_reason": exit_reason,
    }


def _run_portfolio(signals: pd.DataFrame, price_idx: dict,
                   stop_loss: float, time_stop: int, profit_exit: float,
                   max_pos: int = 15,
                   z_scores: dict = None, z_exit: float = None) -> tuple[list, list]:
    if signals.empty:
        return [], []

    trades   = []
    capital  = INITIAL_CAPITAL
    equity   = [(str(signals["date"].min().date()), capital)]
    open_pos = {}   # sym → exit_date

    for _, sig in signals.sort_values("date").iterrows():
        date = sig["date"]
        open_pos = {s: d for s, d in open_pos.items() if d > date}
        if len(open_pos) >= max_pos or sig["symbol"] in open_pos:
            continue

        z_ser = z_scores.get(sig["symbol"]) if z_scores else None
        t = _sim_trade(sig["symbol"], date, sig["entry_price"],
                       price_idx, stop_loss, time_stop, profit_exit,
                       z_ser, z_exit)
        if t is None:
            continue

        capital += capital * PCT_PER_TRADE * t["pnl_pct"]
        equity.append((t["exit_date"], round(capital, 2)))
        open_pos[sig["symbol"]] = pd.Timestamp(t["exit_date"])
        trades.append(t)

    return trades, equity


def _stats(trades: list, equity: list) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "sharpe": 0,
                "max_drawdown_pct": 0, "annual_return_pct": 0,
                "avg_pnl_pct": 0, "profit_factor": 0}
    pnls = [t["pnl_pct"] for t in trades]
    wins = sum(p > 0 for p in pnls)
    pf   = (sum(p for p in pnls if p > 0) /
            max(1e-9, sum(abs(p) for p in pnls if p < 0)))

    edf = pd.DataFrame(equity, columns=["date","capital"])
    edf["date"] = pd.to_datetime(edf["date"])
    edf = edf.sort_values("date").drop_duplicates("date")
    edf["ret"] = edf["capital"].pct_change().fillna(0)
    sharpe = (edf["ret"].mean() / edf["ret"].std() * np.sqrt(252)
              if edf["ret"].std() > 0 else 0.0)
    edf["peak"] = edf["capital"].cummax()
    edf["dd"]   = (edf["capital"] - edf["peak"]) / edf["peak"] * 100
    max_dd      = float(edf["dd"].min())
    days        = max(1, (edf["date"].iloc[-1] - edf["date"].iloc[0]).days)
    ann         = (edf["capital"].iloc[-1] / INITIAL_CAPITAL) ** (365/days) - 1

    return {
        "n_trades":         len(trades),
        "win_rate":         round(wins / len(pnls) * 100, 1),
        "avg_pnl_pct":      round(np.mean(pnls) * 100, 2),
        "profit_factor":    round(pf, 2),
        "sharpe":           round(float(sharpe), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "annual_return_pct":round(ann * 100, 2),
        "equity_curve":     equity,
    }


# ── Signal generators ──────────────────────────────────────────────────────────

def signals_mr_enhanced(price_idx, z_scores, prox_52w, quality,
                        start, end, z_thresh, q_min=60.0) -> pd.DataFrame:
    """Mean reversion + near 52w low + quality gate."""
    rows = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for sym, z in z_scores.items():
        df   = price_idx[sym]
        prox = prox_52w.get(sym)
        q    = quality.get(sym, 0)
        if prox is None or q < q_min:
            continue
        window = z[(z.index >= s) & (z.index <= e)]
        for date, zval in window.items():
            if pd.isna(zval) or zval >= z_thresh:
                continue
            # Check previous bar wasn't already below threshold (fresh cross)
            prev_idx = z.index.get_loc(date)
            if prev_idx == 0:
                continue
            z_prev = z.iloc[prev_idx - 1]
            if not pd.isna(z_prev) and z_prev < z_thresh:
                continue
            # Proximity: must be in lower 25% of 52w range
            p = prox.asof(date)
            if pd.isna(p) or p > 0.25:
                continue
            price = float(df.loc[date, "close"]) if date in df.index else None
            if price is None:
                continue
            rows.append({"symbol": sym, "date": date, "entry_price": price,
                         "z": round(zval, 3), "quality": q, "prox_52w": round(p, 3)})
    df_sig = pd.DataFrame(rows)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(["symbol", "date"])
    return df_sig


def signals_vix_spike(price_idx, z_scores, vix, start, end,
                      vix_jump_pct, z_thresh) -> pd.DataFrame:
    """VIX jumps > vix_jump_pct in 1 day → buy all oversold stocks that day."""
    if vix.empty:
        return pd.DataFrame()
    rows = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    vix_w = vix[(vix.index >= s) & (vix.index <= e)]
    vix_chg = vix_w.pct_change()
    spike_dates = vix_chg[vix_chg >= vix_jump_pct].index

    for date in spike_dates:
        for sym, z in z_scores.items():
            zval = z.asof(date)
            if pd.isna(zval) or zval >= z_thresh:
                continue
            df = price_idx[sym]
            bar = df[df.index >= date]
            if bar.empty:
                continue
            rows.append({"symbol": sym, "date": date,
                         "entry_price": float(bar["close"].iloc[0]),
                         "z": round(zval, 3), "vix_jump": round(vix_chg[date], 3)})
    df_sig = pd.DataFrame(rows)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(["symbol", "date"])
    return df_sig


def signals_sector_mr(price_idx, z_scores, quality, sector_map,
                      start, end, sector_z_thresh, stock_z_thresh,
                      q_min=60.0, top_n=3) -> pd.DataFrame:
    """Sector z-score below thresh → buy top-quality stocks in that sector."""
    if not sector_map:
        return pd.DataFrame()
    rows = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    sectors = {}
    for sym, sec in sector_map.items():
        if sym in z_scores:
            sectors.setdefault(sec, []).append(sym)

    # Build daily sector z-scores
    all_dates = pd.date_range(s, e, freq="B")
    for date in all_dates:
        for sec, syms in sectors.items():
            vals = [z_scores[sym].asof(date) for sym in syms
                    if sym in z_scores and not pd.isna(z_scores[sym].asof(date))]
            if len(vals) < 3:
                continue
            sec_z = np.mean(vals)
            if sec_z >= sector_z_thresh:
                continue
            # Top-quality oversold stocks in this sector
            candidates = [
                (sym, z_scores[sym].asof(date), quality.get(sym, 0))
                for sym in syms
                if sym in z_scores
                and not pd.isna(z_scores[sym].asof(date))
                and z_scores[sym].asof(date) < stock_z_thresh
                and quality.get(sym, 0) >= q_min
            ]
            candidates.sort(key=lambda x: -x[2])   # highest quality first
            for sym, zval, q in candidates[:top_n]:
                df = price_idx[sym]
                bar = df[df.index >= date]
                if bar.empty:
                    continue
                rows.append({"symbol": sym, "date": date,
                             "entry_price": float(bar["close"].iloc[0]),
                             "z": round(zval, 3), "sector": sec, "quality": q,
                             "sector_z": round(sec_z, 3)})
    df_sig = pd.DataFrame(rows)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(["symbol", "date"])
    return df_sig


def signals_quality_momentum(price_idx, z_scores, quality,
                              start, end, z_thresh, q_min=75.0,
                              lookback=20) -> pd.DataFrame:
    """z > thresh AND quality high AND at N-day high → momentum long."""
    rows = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for sym, z in z_scores.items():
        q = quality.get(sym, 0)
        if q < q_min:
            continue
        df   = price_idx[sym]
        win  = z[(z.index >= s) & (z.index <= e)]
        for date, zval in win.items():
            if pd.isna(zval) or zval <= z_thresh:
                continue
            # Fresh cross upward
            idx = z.index.get_loc(date)
            if idx == 0:
                continue
            if not pd.isna(z.iloc[idx-1]) and z.iloc[idx-1] > z_thresh:
                continue
            # Price at N-day high
            bars_before = df[df.index <= date].tail(lookback + 1)
            if len(bars_before) < lookback:
                continue
            if float(bars_before["close"].iloc[-1]) < float(bars_before["close"].max()):
                continue
            rows.append({"symbol": sym, "date": date,
                         "entry_price": float(df.loc[date, "close"]) if date in df.index else float(bars_before["close"].iloc[-1]),
                         "z": round(zval, 3), "quality": q})
    df_sig = pd.DataFrame(rows)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(["symbol","date"])
    return df_sig


def signals_capitulation(price_idx, z_scores, start, end,
                          z_thresh, n_down_days=5,
                          vol_mult=1.5) -> pd.DataFrame:
    """N consecutive down days + volume spike + z < thresh → capitulation buy."""
    rows = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for sym, df in price_idx.items():
        if "volume" not in df.columns:
            continue
        z    = z_scores.get(sym)
        if z is None:
            continue
        win  = df[(df.index >= s) & (df.index <= e)]
        for i in range(n_down_days, len(win)):
            date = win.index[i]
            # N consecutive down closes
            closes = win["close"].iloc[i - n_down_days:i + 1]
            if not all(closes.iloc[j] < closes.iloc[j-1] for j in range(1, len(closes))):
                continue
            # Volume spike
            vol_now = float(win["volume"].iloc[i])
            vol_avg = float(win["volume"].iloc[max(0, i-20):i].mean())
            if vol_avg <= 0 or vol_now < vol_mult * vol_avg:
                continue
            # Z-score below threshold
            zval = z.asof(date)
            if pd.isna(zval) or zval >= z_thresh:
                continue
            rows.append({"symbol": sym, "date": date,
                         "entry_price": float(win["close"].iloc[i]),
                         "z": round(zval, 3), "vol_ratio": round(vol_now/vol_avg, 2)})
    df_sig = pd.DataFrame(rows)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(["symbol","date"])
    return df_sig


def signals_bb_squeeze(price_idx, bb_data, z_scores, start, end,
                        z_max=1.0) -> pd.DataFrame:
    """BB squeeze then close above upper band → buy breakout."""
    rows = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for sym, bdf in bb_data.items():
        df  = price_idx[sym]
        z   = z_scores.get(sym)
        win = bdf[(bdf.index >= s) & (bdf.index <= e)]
        for i in range(1, len(win)):
            date = win.index[i]
            prev = win.index[i-1]
            # Previous bar was in squeeze, current is not AND close > upper
            if not bool(win.loc[prev, "squeeze"]):
                continue
            if bool(win.loc[date, "squeeze"]):
                continue
            close = float(df.loc[date, "close"]) if date in df.index else None
            upper = float(win.loc[date, "bb_upper"])
            if close is None or close <= upper:
                continue
            # z-score not too extreme (avoid chasing)
            if z is not None:
                zval = z.asof(date)
                if not pd.isna(zval) and zval > z_max:
                    continue
            rows.append({"symbol": sym, "date": date,
                         "entry_price": close,
                         "bb_upper": round(upper, 2)})
    df_sig = pd.DataFrame(rows)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(["symbol","date"])
    return df_sig


def signals_gap_fade(price_idx, quality, start, end,
                     gap_thresh=-0.03, q_min=60.0) -> pd.DataFrame:
    """Gap down > gap_thresh at open in quality stock → fade (buy at open)."""
    rows = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for sym, df in price_idx.items():
        q = quality.get(sym, 0)
        if q < q_min or "open" not in df.columns:
            continue
        win = df[(df.index >= s) & (df.index <= e)]
        if len(win) < 2:
            continue
        prev_close = win["close"].shift(1)
        gap        = (win["open"] - prev_close) / prev_close.replace(0, np.nan)
        fade_dates = gap[gap <= gap_thresh].dropna().index
        for date in fade_dates:
            entry_price = float(win.loc[date, "open"])
            rows.append({"symbol": sym, "date": date,
                         "entry_price": entry_price,
                         "gap_pct": round(float(gap[date]), 4), "quality": q})
    df_sig = pd.DataFrame(rows)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(["symbol","date"])
    return df_sig


# ── Parameter sweep + walk-forward ────────────────────────────────────────────

def _sweep_and_eval(name: str, train_fn, test_fn, val_fn,
                    stop_grid, time_grid, profit_grid,
                    z_scores=None, z_exit=None, max_pos=15,
                    price_idx=None) -> dict:
    """Run sweep on train, lock best, eval on val + test."""

    print(f"\n{'═'*60}")
    print(f"  {name.upper().replace('_',' ')}")
    print(f"{'═'*60}")

    # STEP 1: TRAIN sweep
    train_sigs = train_fn()
    print(f"\n  Train signals: {len(train_sigs)}")
    if len(train_sigs) < 5:
        print("  ⚠  Too few train signals — skipping.")
        return {"status": "no_signals", "name": name}

    best = (stop_grid[0], time_grid[0], profit_grid[0], -999.0)
    print(f"  {'stop':>6} {'days':>5} {'profit':>7} {'n':>5} {'wr%':>6} {'sharpe':>7} {'dd%':>7}")
    print(f"  {'─'*50}")
    for stop in stop_grid:
        for days in time_grid:
            for profit in profit_grid:
                t, eq = _run_portfolio(train_sigs, price_idx, stop, days, profit,
                                       max_pos, z_scores, z_exit)
                if len(t) < 5:
                    continue
                r = _stats(t, eq)
                mk = "✅" if r["sharpe"] > best[3] else "  "
                print(f"  {mk} {stop:>6.2f} {days:>5} {profit:>7.2f} "
                      f"{r['n_trades']:>5} {r['win_rate']:>5.1f}% "
                      f"{r['sharpe']:>7.2f} {r['max_drawdown_pct']:>6.1f}%")
                if r["sharpe"] > best[3]:
                    best = (stop, days, profit, r["sharpe"])

    b_stop, b_days, b_profit, b_sh = best
    print(f"\n  → Best: stop={b_stop}  days={b_days}  profit={b_profit}  sharpe={b_sh:.2f}")

    train_t, train_eq = _run_portfolio(train_sigs, price_idx, b_stop, b_days,
                                       b_profit, max_pos, z_scores, z_exit)
    train_r = _stats(train_t, train_eq)
    _print_period("TRAIN", TRAIN_START, TRAIN_END, train_r)

    # STEP 2: VAL
    val_sigs = val_fn()
    val_t, val_eq = _run_portfolio(val_sigs, price_idx, b_stop, b_days,
                                   b_profit, max_pos, z_scores, z_exit)
    val_r = _stats(val_t, val_eq)
    _print_period("VAL  ", VAL_START, VAL_END, val_r)

    # STEP 3: TEST (unbiased)
    test_sigs = test_fn()
    test_t, test_eq = _run_portfolio(test_sigs, price_idx, b_stop, b_days,
                                     b_profit, max_pos, z_scores, z_exit)
    test_r = _stats(test_t, test_eq)
    _print_period("TEST ", TEST_START, TEST_END, test_r)

    passes = (
        test_r["n_trades"] >= PASS_TRADES and
        test_r["win_rate"]         > PASS_WR    and
        test_r["sharpe"]           > PASS_SHARPE and
        test_r["max_drawdown_pct"] > PASS_MAXDD
    )
    verdict = "✅ PASS — activate" if passes else "❌ FAIL — keep inactive"
    print(f"\n  Verdict: {verdict}")

    return {
        "name":        name,
        "status":      "pass" if passes else "fail",
        "best_stop":   b_stop,
        "best_days":   b_days,
        "best_profit": b_profit,
        "train":       {k: v for k, v in train_r.items() if k != "equity_curve"},
        "validation":  {k: v for k, v in val_r.items()   if k != "equity_curve"},
        "oos_test":    {k: v for k, v in test_r.items()   if k != "equity_curve"},
        "oos_equity":  test_eq,
        "oos_trades":  test_t,
    }


def _print_period(label, start, end, r):
    if r["n_trades"] == 0:
        print(f"\n  {label} [{start} → {end}]: ⚠ 0 trades")
        return
    ok = lambda v: "✅" if v else "❌"
    print(f"\n  {'─'*54}")
    print(f"  {label} [{start} → {end}]")
    print(f"  {'─'*54}")
    print(f"  Trades:    {r['n_trades']:>5}   Win Rate: {r['win_rate']:.1f}%  {ok(r['win_rate']>PASS_WR)}")
    print(f"  Sharpe:    {r['sharpe']:>5.2f}   AvgPnL:  {r['avg_pnl_pct']:+.2f}%")
    print(f"  MaxDD:    {r['max_drawdown_pct']:>6.1f}%  {ok(r['max_drawdown_pct']>PASS_MAXDD)}   Ann: {r['annual_return_pct']:.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(only: str = None, save: bool = False):
    logger.info("Loading data...")
    prices    = _load_prices()
    price_idx = _build_indexed(prices)
    vix       = _load_vix()
    quality   = _load_quality()
    sector    = _load_sector()

    logger.info("Precomputing indicators...")
    z_scores = _precompute_z(price_idx)
    prox_52w = _precompute_52w(price_idx)
    bb_data  = _precompute_bb(price_idx)

    # ── Strategy registry ────────────────────────────────────────────────────
    strategies = {

        "mr_enhanced": lambda: _sweep_and_eval(
            "MR Enhanced (z + 52w-low + quality)",
            train_fn   = lambda: signals_mr_enhanced(price_idx, z_scores, prox_52w, quality, TRAIN_START, TRAIN_END, z_thresh=-3.0, q_min=60),
            val_fn     = lambda: signals_mr_enhanced(price_idx, z_scores, prox_52w, quality, VAL_START, VAL_END, z_thresh=-3.0, q_min=60),
            test_fn    = lambda: signals_mr_enhanced(price_idx, z_scores, prox_52w, quality, TEST_START, TEST_END, z_thresh=-3.0, q_min=60),
            stop_grid  = [-0.06, -0.08, -0.10],
            time_grid  = [15, 20],
            profit_grid= [0.10, 0.15, 0.20],
            z_scores   = z_scores, z_exit=-1.0,
            max_pos=15, price_idx=price_idx,
        ),

        "vix_spike_buy": lambda: _sweep_and_eval(
            "VIX Spike Buy",
            train_fn   = lambda: signals_vix_spike(price_idx, z_scores, vix, TRAIN_START, TRAIN_END, vix_jump_pct=0.15, z_thresh=-2.0),
            val_fn     = lambda: signals_vix_spike(price_idx, z_scores, vix, VAL_START, VAL_END, vix_jump_pct=0.15, z_thresh=-2.0),
            test_fn    = lambda: signals_vix_spike(price_idx, z_scores, vix, TEST_START, TEST_END, vix_jump_pct=0.15, z_thresh=-2.0),
            stop_grid  = [-0.05, -0.07, -0.10],
            time_grid  = [5, 10, 15],
            profit_grid= [0.08, 0.12, 0.15],
            z_scores   = z_scores, z_exit=-0.5,
            max_pos=20, price_idx=price_idx,
        ),

        "sector_mr": lambda: _sweep_and_eval(
            "Sector Mean Reversion",
            train_fn   = lambda: signals_sector_mr(price_idx, z_scores, quality, sector, TRAIN_START, TRAIN_END, sector_z_thresh=-1.5, stock_z_thresh=-2.0, q_min=60),
            val_fn     = lambda: signals_sector_mr(price_idx, z_scores, quality, sector, VAL_START, VAL_END, sector_z_thresh=-1.5, stock_z_thresh=-2.0, q_min=60),
            test_fn    = lambda: signals_sector_mr(price_idx, z_scores, quality, sector, TEST_START, TEST_END, sector_z_thresh=-1.5, stock_z_thresh=-2.0, q_min=60),
            stop_grid  = [-0.06, -0.08],
            time_grid  = [10, 15, 20],
            profit_grid= [0.10, 0.15],
            z_scores   = z_scores, z_exit=-0.5,
            max_pos=15, price_idx=price_idx,
        ),

        "quality_momentum": lambda: _sweep_and_eval(
            "Quality Momentum Long",
            train_fn   = lambda: signals_quality_momentum(price_idx, z_scores, quality, TRAIN_START, TRAIN_END, z_thresh=2.0, q_min=75),
            val_fn     = lambda: signals_quality_momentum(price_idx, z_scores, quality, VAL_START, VAL_END, z_thresh=2.0, q_min=75),
            test_fn    = lambda: signals_quality_momentum(price_idx, z_scores, quality, TEST_START, TEST_END, z_thresh=2.0, q_min=75),
            stop_grid  = [-0.05, -0.07, -0.10],
            time_grid  = [10, 15, 20],
            profit_grid= [0.10, 0.15, 0.20],
            max_pos=10, price_idx=price_idx,
        ),

        "capitulation": lambda: _sweep_and_eval(
            "Capitulation Reversal (5d down + vol spike)",
            train_fn   = lambda: signals_capitulation(price_idx, z_scores, TRAIN_START, TRAIN_END, z_thresh=-2.0, n_down_days=5, vol_mult=1.5),
            val_fn     = lambda: signals_capitulation(price_idx, z_scores, VAL_START, VAL_END, z_thresh=-2.0, n_down_days=5, vol_mult=1.5),
            test_fn    = lambda: signals_capitulation(price_idx, z_scores, TEST_START, TEST_END, z_thresh=-2.0, n_down_days=5, vol_mult=1.5),
            stop_grid  = [-0.06, -0.08, -0.10],
            time_grid  = [10, 15, 20],
            profit_grid= [0.10, 0.15, 0.20],
            z_scores   = z_scores, z_exit=-0.5,
            max_pos=15, price_idx=price_idx,
        ),

        "bb_squeeze": lambda: _sweep_and_eval(
            "Bollinger Band Squeeze Breakout",
            train_fn   = lambda: signals_bb_squeeze(price_idx, bb_data, z_scores, TRAIN_START, TRAIN_END, z_max=1.0),
            val_fn     = lambda: signals_bb_squeeze(price_idx, bb_data, z_scores, VAL_START, VAL_END, z_max=1.0),
            test_fn    = lambda: signals_bb_squeeze(price_idx, bb_data, z_scores, TEST_START, TEST_END, z_max=1.0),
            stop_grid  = [-0.05, -0.07, -0.10],
            time_grid  = [10, 15, 20],
            profit_grid= [0.08, 0.12, 0.15],
            max_pos=10, price_idx=price_idx,
        ),

        "gap_fade": lambda: _sweep_and_eval(
            "Gap-Down Fade (quality stocks)",
            train_fn   = lambda: signals_gap_fade(price_idx, quality, TRAIN_START, TRAIN_END, gap_thresh=-0.03, q_min=65),
            val_fn     = lambda: signals_gap_fade(price_idx, quality, VAL_START, VAL_END, gap_thresh=-0.03, q_min=65),
            test_fn    = lambda: signals_gap_fade(price_idx, quality, TEST_START, TEST_END, gap_thresh=-0.03, q_min=65),
            stop_grid  = [-0.03, -0.05, -0.07],
            time_grid  = [3, 5, 10],
            profit_grid= [0.03, 0.05, 0.08],
            max_pos=10, price_idx=price_idx,
        ),
    }

    if only:
        strategies = {k: v for k, v in strategies.items() if k == only}
        if not strategies:
            print(f"Unknown strategy: {only}. Options: {list(strategies.keys())}")
            return

    print(f"\n{'═'*60}")
    print(f"  MULTI-STRATEGY WALK-FORWARD BACKTEST")
    print(f"  TRAIN {TRAIN_START}→{TRAIN_END} | VAL {VAL_START}→{VAL_END} | TEST {TEST_START}→{TEST_END}")
    print(f"  Testing {len(strategies)} strategies")
    print(f"{'═'*60}")

    results = {}
    for name, run_fn in strategies.items():
        try:
            results[name] = run_fn()
        except Exception as e:
            logger.error(f"{name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"name": name, "status": "error", "error": str(e)}

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'═'*60}")
    print(f"  FINAL SUMMARY — OOS TEST ({TEST_START} → {TEST_END})")
    print(f"{'═'*60}")
    print(f"  {'Strategy':<32} {'N':>5} {'WinR':>6} {'Sharpe':>8} {'MaxDD':>8} {'Ann%':>7}  Verdict")
    print(f"  {'─'*75}")

    passing = []
    for name, r in results.items():
        oos = r.get("oos_test", {})
        if r.get("status") in ("no_signals", "error"):
            print(f"  {name:<32}  — {r.get('status','?')}")
            continue
        v = "✅" if r["status"] == "pass" else "❌"
        print(f"  {name:<32} {oos.get('n_trades',0):>5} "
              f"{oos.get('win_rate',0):>5.1f}% "
              f"{oos.get('sharpe',0):>8.2f} "
              f"{oos.get('max_drawdown_pct',0):>7.1f}% "
              f"{oos.get('annual_return_pct',0):>6.1f}%  {v}")
        if r["status"] == "pass":
            passing.append(name)

    print(f"\n  Strategies passing all criteria: {len(passing)}")
    for s in passing:
        r = results[s]
        print(f"    ✅ {s}  —  stop={r['best_stop']}  days={r['best_days']}  profit={r['best_profit']}")

    if save:
        out = cfg.results_dir / "multi_strategy_backtest.json"
        payload = {
            "generated":   datetime.now().isoformat(),
            "train_start": TRAIN_START, "train_end": TRAIN_END,
            "val_start":   VAL_START,   "val_end":   VAL_END,
            "test_start":  TEST_START,  "test_end":  TEST_END,
            "passing":     passing,
        }
        for name, r in results.items():
            payload[name] = {k: v for k, v in r.items()
                             if k not in ("oos_trades", "oos_equity")}
            payload[f"{name}_equity"]  = r.get("oos_equity", [])
            payload[f"{name}_trades"]  = r.get("oos_trades", [])
        out.write_text(json.dumps(payload, indent=2, default=str))
        logger.success(f"Saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save",     action="store_true")
    p.add_argument("--strategy", default=None, help="Run one strategy only")
    args = p.parse_args()
    main(only=args.strategy, save=args.save)
