"""
Multi-Strategy Backtest
========================
Tests MEAN_REVERSION and MOMENTUM side-by-side and combined.
Uses 2020–2026 NSE price data with strict point-in-time rules.

Features:
  - Fresh signal detection (z-score threshold crossing, not sustained below)
  - Strategy-specific stop-loss, take-profit, and time-stop rules
  - Position capacity: max 15 slots shared across strategies
  - 50 bps round-trip transaction cost
  - Per-strategy and combined metrics + exit-reason breakdown

Run:
    python scripts/run_strategy_backtest.py [--period test|train|all]
    python scripts/run_strategy_backtest.py --save   # saves JSON for dashboard
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg
from alphasense.signal.engine import rolling_zscore, StrategyType

# ── Params ─────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000   # ₹10L paper capital
COST_BPS        = 50          # 0.5% round trip
PCT_PER_TRADE   = 0.08        # 8% of capital per position
MAX_POSITIONS   = 15
MIN_PRICE_ROWS  = 400         # need at least this much history before signalling

PERIODS = {
    "train": ("2020-01-01", "2021-12-31"),
    "val":   ("2022-01-01", "2022-12-31"),
    "test":  ("2023-01-01", "2026-04-30"),
    "all":   ("2020-01-01", "2026-04-30"),
}

# Per-strategy exit rules
EXIT = {
    StrategyType.MEAN_REVERSION: dict(stop=-0.08, time=20, rec_z=-1.0, profit=None),
    StrategyType.MOMENTUM:       dict(stop=-0.05, time=15, rec_z=None,  profit=0.15),
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load(universe_only=True):
    from alphasense.data.nse_client import load_prices, load_vix
    try:
        from alphasense.screener.fundamental import load_universe
        u = load_universe()
        syms = u["symbol"].tolist() if not u.empty and universe_only else None
    except Exception:
        syms = None
    prices = load_prices(syms)
    vix    = load_vix()
    logger.info(f"Loaded {len(prices)} price series, VIX: {not vix.empty}")
    return prices, vix


def _build_indexed(prices: dict) -> dict:
    """date-indexed DataFrames for fast slicing."""
    out = {}
    for sym, df in prices.items():
        d = df.sort_values("date").copy()
        d.index = pd.to_datetime(d["date"])
        out[sym] = d
    return out


def _precompute_signals(price_idx: dict, start: str, end: str, sc) -> pd.DataFrame:
    """
    Vectorised: find fresh z-score threshold crossings for all symbols.
    Returns DataFrame with columns: date, symbol, entry_price, strategy
    """
    s_ts = pd.Timestamp(start)
    e_ts = pd.Timestamp(end)
    rows = []

    for sym, df in price_idx.items():
        if len(df) < MIN_PRICE_ROWS + 10:
            continue
        z = rolling_zscore(df["close"], sc.rolling_window, sc.return_period)
        z.index = df.index

        # Mean reversion: fresh cross below threshold
        mr_cross = (z < sc.zscore_threshold) & (z.shift(1) >= sc.zscore_threshold)
        # Momentum: fresh cross above threshold
        mo_cross = (z > sc.momentum_zscore_min) & (z.shift(1) <= sc.momentum_zscore_min)

        for strat, mask in [(StrategyType.MEAN_REVERSION, mr_cross),
                            (StrategyType.MOMENTUM,       mo_cross)]:
            for sig_date in z[mask].index:
                if not (s_ts <= sig_date <= e_ts):
                    continue
                # Entry at next close (T+1 lag)
                nxt = df[df.index > sig_date]
                if nxt.empty:
                    continue
                entry_price = float(nxt.iloc[0]["close"])
                if entry_price <= 0:
                    continue
                rows.append({
                    "date":        sig_date,
                    "symbol":      sym,
                    "entry_price": entry_price,
                    "strategy":    strat,
                    "z_at_signal": float(z.loc[sig_date]),
                })

    sig_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    logger.info(f"Pre-computed {len(sig_df)} signals "
                f"(MR={( sig_df['strategy']==StrategyType.MEAN_REVERSION).sum()}, "
                f"MO={(sig_df['strategy']==StrategyType.MOMENTUM).sum()})")
    return sig_df


# ── Trade outcome simulator ────────────────────────────────────────────────────

def _trade_outcome(sym: str, entry_date: pd.Timestamp, entry_price: float,
                   strategy: str, df: pd.DataFrame, z: pd.Series) -> dict:
    """Simulate one trade from T+1 open to exit. Returns trade dict."""
    rules = EXIT[strategy]
    future = df[df.index > entry_date]
    if len(future) < 2:
        return None

    stop   = rules["stop"]
    t_stop = rules["time"]
    rec_z  = rules["rec_z"]
    profit = rules["profit"]

    exit_price  = float(future["close"].iloc[-1])
    exit_reason = "time_stop"
    days_held   = min(t_stop, len(future))

    for i, (idx, row) in enumerate(future.iterrows()):
        if i >= t_stop:
            break
        curr_price = float(row["close"])
        pnl        = (curr_price - entry_price) / entry_price

        if pnl <= stop:
            exit_price  = entry_price * (1 + stop)
            exit_reason = "stop_loss"
            days_held   = i + 1
            break

        if profit and pnl >= profit:
            exit_price  = curr_price
            exit_reason = "take_profit"
            days_held   = i + 1
            break

        if rec_z is not None:
            z_now = z[z.index <= idx]
            if len(z_now) > 0 and not pd.isna(z_now.iloc[-1]) and z_now.iloc[-1] > rec_z:
                exit_price  = curr_price
                exit_reason = "recovery"
                days_held   = i + 1
                break

        if strategy == StrategyType.MOMENTUM:
            z_now = z[z.index <= idx]
            if len(z_now) > 0 and not pd.isna(z_now.iloc[-1]) and z_now.iloc[-1] < 0:
                exit_price  = curr_price
                exit_reason = "momentum_reversal"
                days_held   = i + 1
                break

        exit_price = curr_price
        days_held  = i + 1

    gross = (exit_price - entry_price) / entry_price
    net   = gross - COST_BPS / 10_000
    return {
        "symbol":      sym,
        "strategy":    strategy,
        "entry_date":  str(entry_date.date()),
        "entry_price": round(entry_price, 2),
        "exit_price":  round(exit_price, 2),
        "days_held":   days_held,
        "exit_reason": exit_reason,
        "gross_ret":   round(gross, 4),
        "net_ret":     round(net, 4),
    }


# ── Portfolio simulation ───────────────────────────────────────────────────────

def run_sim(sig_df: pd.DataFrame, price_idx: dict, zscores: dict,
            strategies: list, start: str, end: str,
            vix_df: pd.DataFrame = None, sc=None) -> dict:
    """
    Process pre-computed signals in date order, respecting position capacity.
    Returns dict: trades, equity_curve, final_capital.
    """
    if sc is None:
        sc = cfg.signal

    capital       = INITIAL_CAPITAL
    open_pos      = {}   # sym → {entry_date, entry_price, strategy, size, exit_date}
    all_trades    = []
    equity_curve  = [(pd.Timestamp(start), capital)]

    # Pre-compute exit dates/prices for each potential trade
    trade_cache = {}

    def _get_trade(sym, entry_date, entry_price, strategy):
        key = (sym, str(entry_date.date()), strategy)
        if key not in trade_cache:
            if sym not in price_idx or sym not in zscores:
                trade_cache[key] = None
            else:
                trade_cache[key] = _trade_outcome(
                    sym, entry_date, entry_price, strategy,
                    price_idx[sym], zscores[sym]
                )
        return trade_cache[key]

    all_dates = sorted(sig_df["date"].unique().tolist())

    for today in all_dates:
        today_ts = pd.Timestamp(today)

        # ── Close expired positions ───────────────────────────────────────────
        to_close = []
        for sym, pos in open_pos.items():
            exit_dt = pos.get("exit_date")
            if exit_dt and today_ts >= exit_dt:
                t = pos["trade"]
                capital += pos["size"] * t["net_ret"]
                all_trades.append({**t, "pnl_amt": round(pos["size"] * t["net_ret"], 0)})
                to_close.append(sym)
        for sym in to_close:
            del open_pos[sym]

        equity_curve.append((today_ts, capital))

        # ── VIX gate ─────────────────────────────────────────────────────────
        if vix_df is not None and not vix_df.empty:
            vc = vix_df["date"] if "date" in vix_df.columns else pd.Series(vix_df.index)
            avail = vix_df[pd.to_datetime(vix_df["date"] if "date" in vix_df.columns
                                          else vix_df.index) <= today_ts]
            if not avail.empty:
                vix_val = float(avail["vix_close"].iloc[-1])
                if vix_val > sc.vix_max:
                    continue

        if len(open_pos) >= MAX_POSITIONS:
            continue

        # ── Open new positions from today's signals ───────────────────────────
        todays = sig_df[
            (sig_df["date"] == today) &
            (sig_df["strategy"].isin(strategies))
        ].sort_values("z_at_signal")  # worst z-score first (strongest signal)

        for _, row in todays.iterrows():
            if len(open_pos) >= MAX_POSITIONS:
                break
            sym      = row["symbol"]
            strategy = row["strategy"]
            if sym in open_pos:
                continue

            trade = _get_trade(sym, today_ts, row["entry_price"], strategy)
            if trade is None:
                continue

            days  = trade["days_held"]
            size  = capital * PCT_PER_TRADE
            if size < 5_000:
                continue

            open_pos[sym] = {
                "entry_date": today_ts,
                "entry_price": row["entry_price"],
                "strategy":   strategy,
                "size":       size,
                "exit_date":  today_ts + pd.Timedelta(days=days),
                "trade":      trade,
            }

    # Flush remaining open positions
    for sym, pos in open_pos.items():
        t = pos["trade"]
        capital += pos["size"] * t["net_ret"]
        all_trades.append({**t, "pnl_amt": round(pos["size"] * t["net_ret"], 0)})

    equity_curve.append((pd.Timestamp(end), capital))
    return {"trades": all_trades, "equity_curve": equity_curve,
            "final_capital": capital}


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(trades, equity_curve, initial, start, end, label="combined"):
    if not trades:
        return {"label": label, "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0,
                "max_drawdown_pct": 0.0, "total_return_pct": 0.0,
                "annual_return_pct": 0.0, "profit_factor": 0.0,
                "avg_pnl_pct": 0.0, "avg_holding_days": 0.0,
                "final_capital": initial}

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity_curve, columns=["date", "capital"])
    edf["date"] = pd.to_datetime(edf["date"])
    edf = edf.drop_duplicates("date").sort_values("date")

    final  = edf["capital"].iloc[-1]
    years  = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.01)
    total  = (final - initial) / initial
    annual = (1 + total) ** (1 / years) - 1

    ret    = edf["capital"].pct_change().dropna()
    sharpe = float((ret.mean() / ret.std()) * np.sqrt(252)) \
             if len(ret) > 1 and ret.std() > 0 else 0.0

    edf["peak"] = edf["capital"].cummax()
    edf["dd"]   = (edf["capital"] - edf["peak"]) / edf["peak"]
    max_dd      = float(edf["dd"].min()) * 100

    wins  = tdf[tdf["net_ret"] > 0]
    loss  = tdf[tdf["net_ret"] <= 0]
    n     = len(tdf)
    wr    = len(wins) / n * 100
    avg   = float(tdf["net_ret"].mean()) * 100
    hld   = float(tdf["days_held"].mean()) if "days_held" in tdf else 0
    gn    = wins["net_ret"].sum()
    ls    = abs(loss["net_ret"].sum())
    pf    = gn / ls if ls > 0 else float("inf")

    return {
        "label":             label,
        "period":            f"{start} → {end}",
        "n_trades":          n,
        "win_rate":          round(wr, 1),
        "avg_pnl_pct":       round(avg, 2),
        "avg_holding_days":  round(hld, 1),
        "profit_factor":     round(pf, 2) if pf != float("inf") else 99.0,
        "total_return_pct":  round(total * 100, 2),
        "annual_return_pct": round(annual * 100, 2),
        "sharpe":            round(sharpe, 2),
        "max_drawdown_pct":  round(max_dd, 2),
        "final_capital":     round(final, 0),
        "passed":            sharpe > 1.2 and max_dd > -18 and wr > 50,
    }


def _print(m):
    pf = m.get("profit_factor", 0)
    pf_s = "∞" if pf >= 99 else f"{pf:.2f}"
    print(f"\n{'─'*54}")
    print(f"  {m['label'].upper():30s}  {m.get('period','')}")
    print(f"{'─'*54}")
    print(f"  Trades:           {m['n_trades']:>7d}")
    print(f"  Win Rate:         {m['win_rate']:>7.1f}%")
    print(f"  Avg PnL/trade:    {m['avg_pnl_pct']:>7.2f}%")
    print(f"  Avg hold:         {m['avg_holding_days']:>6.1f} days")
    print(f"  Profit Factor:    {pf_s:>8}")
    print(f"  Total Return:     {m['total_return_pct']:>7.2f}%")
    print(f"  Annual Return:    {m['annual_return_pct']:>7.2f}%")
    print(f"  Sharpe:           {m['sharpe']:>7.2f}")
    print(f"  Max Drawdown:     {m['max_drawdown_pct']:>7.2f}%")
    print(f"  Final Capital:    ₹{m['final_capital']:>12,.0f}")
    for name, ok in [("Sharpe > 1.2", m["sharpe"] > 1.2),
                     ("Max DD < 18%", m["max_drawdown_pct"] > -18),
                     ("Win rate > 50%", m["win_rate"] > 50)]:
        print(f"    {'✅' if ok else '❌'} {name}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-strategy backtest")
    p.add_argument("--period",   default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--strategy", default="all",
                   choices=["mean_reversion", "momentum", "all"])
    p.add_argument("--save",     action="store_true",
                   help="Save results JSON for dashboard")
    p.add_argument("--sweep",    action="store_true",
                   help="Run threshold sweep to find optimal params")
    p.add_argument("--mr-z",     type=float, default=None,
                   help="Override mean-reversion z threshold (default: settings)")
    p.add_argument("--mo-z",     type=float, default=None,
                   help="Override momentum z threshold (default: settings)")
    p.add_argument("--stop",     type=float, default=None,
                   help="Override stop-loss pct, e.g. -0.06")
    args = p.parse_args()

    # Apply overrides
    if args.mr_z  is not None: cfg.signal.zscore_threshold      = args.mr_z
    if args.mo_z  is not None: cfg.signal.momentum_zscore_min   = args.mo_z
    if args.stop  is not None: cfg.signal.stop_loss_pct         = args.stop

    from dotenv import load_dotenv
    load_dotenv(cfg.root / ".env", override=True)

    logger.info("Loading price data and VIX...")
    prices, vix = _load()

    logger.info("Building date-indexed price lookup...")
    price_idx = _build_indexed(prices)

    logger.info("Pre-computing z-scores...")
    zscores = {}
    for sym, df in price_idx.items():
        if len(df) < MIN_PRICE_ROWS + 10:
            continue
        z = rolling_zscore(df["close"], cfg.signal.rolling_window,
                           cfg.signal.return_period)
        z.index = df.index
        zscores[sym] = z
    logger.info(f"Z-scores ready for {len(zscores)} symbols")

    start, end = PERIODS[args.period]

    if args.strategy == "all":
        strategies = [StrategyType.MEAN_REVERSION, StrategyType.MOMENTUM]
    else:
        strategies = [args.strategy]

    logger.info(f"Pre-computing signals ({start} → {end})...")
    sig_df = _precompute_signals(price_idx, start, end, cfg.signal)
    sig_df = sig_df[sig_df["strategy"].isin(strategies)]

    logger.info("Running portfolio simulation...")
    result = run_sim(sig_df, price_idx, zscores, strategies,
                     start, end, vix_df=vix)

    all_trades = result["trades"]
    eq         = result["equity_curve"]

    print(f"\n{'═'*54}")
    print(f"  BACKTEST: {args.period.upper()}  ({start} → {end})")
    print(f"  Strategy: {args.strategy.upper()}")
    print(f"{'═'*54}")

    # Combined
    _print(metrics(all_trades, eq, INITIAL_CAPITAL, start, end, "COMBINED"))

    # Per strategy
    for strat in strategies:
        st = [t for t in all_trades if t["strategy"] == strat]
        _print(metrics(st, eq, INITIAL_CAPITAL, start, end, strat))

    # Exit reason breakdown
    if all_trades:
        tdf = pd.DataFrame(all_trades)
        print(f"\n  Exit Reason Breakdown:")
        for strat, grp in tdf.groupby("strategy"):
            print(f"\n    [{strat}]")
            for reason, rgrp in grp.groupby("exit_reason"):
                wr = (rgrp["net_ret"] > 0).mean() * 100
                av = rgrp["net_ret"].mean() * 100
                print(f"      {reason:22s}  n={len(rgrp):4d}  wr={wr:.0f}%  avg={av:+.2f}%")

    # Parameter sweep mode
    if args.sweep:
        print(f"\n{'═'*70}")
        print(f"  THRESHOLD SWEEP  ({start} → {end})")
        print(f"{'═'*70}")
        print(f"  {'z_mr':>6}  {'z_mo':>6}  {'stop':>6}  "
              f"{'trades':>7}  {'wr%':>5}  {'sharpe':>7}  {'dd%':>7}  {'ann%':>7}")
        print(f"  {'─'*68}")

        # Reload pre-computed data once
        _sweep_results = []
        for z_mr in [-2.0, -2.5, -3.0]:
            for z_mo in [2.0, 2.5, 3.0]:
                for stop in [-0.06, -0.08]:
                    cfg.signal.zscore_threshold    = z_mr
                    cfg.signal.momentum_zscore_min = z_mo
                    cfg.signal.stop_loss_pct       = stop
                    cfg.signal.momentum_stop_loss_pct = stop + 0.02   # slightly tighter for momentum

                    _sig = _precompute_signals(price_idx, start, end, cfg.signal)
                    _sig = _sig[_sig["strategy"].isin(strategies)]
                    _res = run_sim(_sig, price_idx, zscores, strategies,
                                   start, end, vix_df=vix)
                    _t   = _res["trades"]
                    _m   = metrics(_t, _res["equity_curve"], INITIAL_CAPITAL, start, end)
                    mark = "✅" if _m["passed"] else "  "
                    print(f"  {mark} {z_mr:>6.1f}  {z_mo:>6.1f}  {stop:>6.2f}  "
                          f"{_m['n_trades']:>7d}  {_m['win_rate']:>5.1f}  "
                          f"{_m['sharpe']:>7.2f}  {_m['max_drawdown_pct']:>7.2f}  "
                          f"{_m['annual_return_pct']:>7.2f}%")
                    _sweep_results.append({**_m, "z_mr": z_mr, "z_mo": z_mo, "stop": stop})

        # Best by Sharpe
        best = max(_sweep_results, key=lambda x: x["sharpe"])
        print(f"\n  Best Sharpe → z_mr={best['z_mr']}  z_mo={best['z_mo']}  "
              f"stop={best['stop']}  sharpe={best['sharpe']}")
        sys.exit(0)

    # Save
    if args.save:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.results_dir / f"strategy_backtest_{args.period}.json"
        out.write_text(json.dumps({
            "metrics_combined": metrics(all_trades, eq, INITIAL_CAPITAL, start, end),
            "metrics_per_strategy": {
                strat: metrics([t for t in all_trades if t["strategy"] == strat],
                                eq, INITIAL_CAPITAL, start, end, strat)
                for strat in strategies
            },
            "trades":      all_trades,
            "equity_curve": [(str(d.date() if hasattr(d,"date") else d), c)
                              for d, c in eq],
        }, indent=2, default=str))
        print(f"\nSaved → {out}")
