"""
Walk-Forward Backtest + Forward Test
======================================
Proper unbiased evaluation of the MEAN_REVERSION strategy.

Pipeline:
  1. SWEEP  — optimise z-threshold and stop-loss on TRAIN data only (2020-2022)
  2. LOCK   — pick best params from train; apply to VALIDATION (2022) to confirm
  3. OOS    — run locked params on TEST (2023-2026), ZERO re-optimisation here
  4. FWD    — load actual paper trade results from paper_state.json

This separates parameter selection from out-of-sample evaluation, so the
test-period numbers are unbiased.  The forward section adds real live evidence.

Run:
    python scripts/run_strategy_backtest.py          # full walk-forward report
    python scripts/run_strategy_backtest.py --save   # also writes JSON for dashboard
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg
from alphasense.signal.engine import rolling_zscore, StrategyType

# ── Fixed periods (never change the test window after seeing it) ───────────────
TRAIN_START, TRAIN_END = "2020-01-01", "2021-12-31"
VAL_START,   VAL_END   = "2022-01-01", "2022-12-31"
TEST_START,  TEST_END  = "2023-01-01", "2026-04-30"

INITIAL_CAPITAL = 1_000_000
COST_BPS        = 50          # 0.5% round-trip
PCT_PER_TRADE   = 0.08
MAX_POSITIONS   = 15
MIN_HISTORY     = 400

# Mean-reversion exit rules (stop updated from settings at runtime)
MR_TIME_STOP = 20
MR_REC_Z     = -1.0


# ── Data loading ───────────────────────────────────────────────────────────────

def _load():
    from alphasense.data.nse_client import load_prices, load_vix
    try:
        from alphasense.screener.fundamental import load_universe
        u    = load_universe()
        syms = u["symbol"].tolist() if not u.empty else None
    except Exception:
        syms = None
    prices = load_prices(syms)
    vix    = load_vix()
    logger.info(f"Loaded {len(prices)} price series")
    return prices, vix


def _build_indexed(prices):
    out = {}
    for sym, df in prices.items():
        d = df.sort_values("date").copy()
        d.index = pd.to_datetime(d["date"])
        out[sym] = d
    return out


def _precompute_z(price_idx, sc):
    zs = {}
    for sym, df in price_idx.items():
        if len(df) < MIN_HISTORY + 10:
            continue
        z = rolling_zscore(df["close"], sc.rolling_window, sc.return_period)
        z.index = df.index
        zs[sym] = z
    logger.info(f"Z-scores for {len(zs)} symbols")
    return zs


# ── Signal detection (mean reversion only) ────────────────────────────────────

def _signals(price_idx, zs, start, end, z_thresh, sc):
    """Fresh crossings where z drops below z_thresh (e.g. -2.5)."""
    s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for sym, df in price_idx.items():
        z = zs.get(sym)
        if z is None:
            continue
        cross = (z < z_thresh) & (z.shift(1) >= z_thresh)
        for sig_date in z[cross].index:
            if not (s_ts <= sig_date <= e_ts):
                continue
            nxt = df[df.index > sig_date]
            if nxt.empty:
                continue
            ep = float(nxt.iloc[0]["close"])
            if ep <= 0:
                continue
            rows.append({"date": sig_date, "symbol": sym,
                         "entry_price": ep, "z_at_signal": float(z.loc[sig_date])})
    sig_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True) \
             if rows else pd.DataFrame()
    return sig_df


# ── Trade outcome simulation ───────────────────────────────────────────────────

def _simulate(sym, entry_date, entry_price, df, z, stop):
    future = df[df.index > entry_date]
    if len(future) < 2:
        return None
    exit_price, exit_reason, days = float(future["close"].iloc[-1]), "time_stop", MR_TIME_STOP
    for i, (idx, row) in enumerate(future.iterrows()):
        if i >= MR_TIME_STOP:
            break
        cp  = float(row["close"])
        pnl = (cp - entry_price) / entry_price
        if pnl <= stop:
            exit_price, exit_reason, days = entry_price * (1 + stop), "stop_loss", i + 1
            break
        z_now = z[z.index <= idx]
        if len(z_now) > 0 and not pd.isna(z_now.iloc[-1]) and z_now.iloc[-1] > MR_REC_Z:
            exit_price, exit_reason, days = cp, "recovery", i + 1
            break
        exit_price, days = cp, i + 1
    gross = (exit_price - entry_price) / entry_price
    return {"symbol": sym, "entry_date": str(entry_date.date()),
            "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
            "days_held": days, "exit_reason": exit_reason,
            "gross_ret": round(gross, 4), "net_ret": round(gross - COST_BPS / 10_000, 4)}


# ── Portfolio simulation ───────────────────────────────────────────────────────

_trade_cache: dict = {}

def _sim_portfolio(sig_df, price_idx, zs, start, end, stop, vix_df=None):
    global _trade_cache
    sc       = cfg.signal
    capital  = INITIAL_CAPITAL
    open_pos = {}
    trades   = []
    equity   = [(pd.Timestamp(start), capital)]

    def _get(sym, dt, ep):
        key = (sym, str(dt.date()), stop)
        if key not in _trade_cache:
            if sym not in price_idx or sym not in zs:
                _trade_cache[key] = None
            else:
                _trade_cache[key] = _simulate(sym, dt, ep, price_idx[sym], zs[sym], stop)
        return _trade_cache[key]

    for today in sorted(sig_df["date"].unique()):
        today_ts = pd.Timestamp(today)

        # Close expired positions
        to_close = []
        for sym, pos in open_pos.items():
            if today_ts >= pos["exit_date"]:
                t = pos["trade"]
                capital += pos["size"] * t["net_ret"]
                trades.append({**t, "pnl_amt": round(pos["size"] * t["net_ret"], 0)})
                to_close.append(sym)
        for sym in to_close:
            del open_pos[sym]

        equity.append((today_ts, capital))

        # VIX gate
        if vix_df is not None and not vix_df.empty:
            av = vix_df[pd.to_datetime(vix_df["date"] if "date" in vix_df.columns
                                        else vix_df.index) <= today_ts]
            if not av.empty and float(av["vix_close"].iloc[-1]) > sc.vix_max:
                continue

        if len(open_pos) >= MAX_POSITIONS:
            continue

        # Open strongest signals today (sorted by z, most extreme first)
        for _, row in sig_df[sig_df["date"] == today].sort_values("z_at_signal").iterrows():
            if len(open_pos) >= MAX_POSITIONS:
                break
            sym = row["symbol"]
            if sym in open_pos:
                continue
            t = _get(sym, today_ts, row["entry_price"])
            if t is None:
                continue
            size = capital * PCT_PER_TRADE
            if size < 5_000:
                continue
            open_pos[sym] = {"size": size, "trade": t,
                             "exit_date": today_ts + pd.Timedelta(days=t["days_held"])}

    # Flush open positions
    for sym, pos in open_pos.items():
        t = pos["trade"]
        capital += pos["size"] * t["net_ret"]
        trades.append({**t, "pnl_amt": round(pos["size"] * t["net_ret"], 0)})

    equity.append((pd.Timestamp(end), capital))
    return {"trades": trades, "equity": equity, "final_capital": capital}


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(trades, equity, initial, start, end, label=""):
    if not trades:
        return {"label": label, "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0,
                "max_drawdown_pct": 0.0, "total_return_pct": 0.0,
                "annual_return_pct": 0.0, "profit_factor": 0.0,
                "avg_pnl_pct": 0.0, "avg_holding_days": 0.0,
                "final_capital": initial, "passed": False}
    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity, columns=["date", "capital"])
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

    n    = len(tdf)
    wins = tdf[tdf["net_ret"] > 0]
    loss = tdf[tdf["net_ret"] <= 0]
    wr   = len(wins) / n * 100
    avg  = float(tdf["net_ret"].mean()) * 100
    hld  = float(tdf["days_held"].mean())
    pf   = wins["net_ret"].sum() / abs(loss["net_ret"].sum()) \
           if loss["net_ret"].sum() != 0 else 99.0

    return {"label": label, "period": f"{start} → {end}",
            "n_trades": n, "win_rate": round(wr, 1),
            "avg_pnl_pct": round(avg, 2), "avg_holding_days": round(hld, 1),
            "profit_factor": round(min(pf, 99.0), 2),
            "total_return_pct": round(total * 100, 2),
            "annual_return_pct": round(annual * 100, 2),
            "sharpe": round(sharpe, 2), "max_drawdown_pct": round(max_dd, 2),
            "final_capital": round(final, 0),
            "passed": sharpe > 1.2 and max_dd > -18 and wr > 50}


def _print(m, header=None):
    if header:
        print(f"\n{'═'*54}")
        print(f"  {header}")
    print(f"{'─'*54}")
    lbl = m.get("label", "")
    per = m.get("period", "")
    print(f"  {lbl:30s}  {per}")
    print(f"{'─'*54}")
    print(f"  Trades:          {m['n_trades']:>7d}")
    print(f"  Win Rate:        {m['win_rate']:>7.1f}%  {'✅' if m['win_rate']>50 else '❌'}")
    print(f"  Avg PnL/trade:   {m['avg_pnl_pct']:>7.2f}%")
    print(f"  Avg hold:        {m['avg_holding_days']:>6.1f} days")
    print(f"  Profit Factor:   {m['profit_factor']:>7.2f}")
    print(f"  Total Return:    {m['total_return_pct']:>7.2f}%")
    print(f"  Annual Return:   {m['annual_return_pct']:>7.2f}%")
    print(f"  Sharpe:          {m['sharpe']:>7.2f}  {'✅' if m['sharpe']>1.2 else '❌'}")
    print(f"  Max Drawdown:    {m['max_drawdown_pct']:>7.2f}%  {'✅' if m['max_drawdown_pct']>-18 else '❌'}")
    print(f"  Final Capital:   ₹{m['final_capital']:>11,.0f}")
    status = "✅ PASS" if m.get("passed") else "❌ FAIL"
    print(f"  Overall:         {status}")


# ── Forward test loader ────────────────────────────────────────────────────────

def _load_forward():
    """Load actual paper trades from paper_state.json."""
    fp = cfg.data_dir / "paper_state.json"
    if not fp.exists():
        return []
    try:
        state  = json.loads(fp.read_text())
        trades = state.get("trades", [])
        closed = [t for t in trades if t.get("pnl_pct") is not None
                  and t.get("direction") == "SELL"]
        return closed
    except Exception as e:
        logger.warning(f"Could not load paper trades: {e}")
        return []


def _print_forward(trades):
    if not trades:
        print("\n  No closed paper trades yet.")
        return
    print(f"\n  {'Symbol':12s}  {'Entry':>10}  {'Exit':>10}  {'PnL%':>7}  {'Days':>5}  Result")
    print(f"  {'─'*58}")
    for t in trades:
        sym   = t.get("symbol", "?")
        ep    = t.get("entry_price", 0)
        xp    = t.get("exit_price", 0)
        pnl   = t.get("pnl_pct", 0) * 100
        ed    = str(t.get("entry_date", ""))[:10]
        xd    = str(t.get("exit_date",  ""))[:10]
        try:
            days = (pd.Timestamp(xd) - pd.Timestamp(ed)).days
        except Exception:
            days = 0
        mark  = "✅" if pnl > 0 else "❌"
        print(f"  {sym:12s}  ₹{ep:>9,.2f}  ₹{xp:>9,.2f}  {pnl:>+7.1f}%  {days:>5d}  {mark}")

    pnls = [t.get("pnl_pct", 0) * 100 for t in trades]
    wr   = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    avg  = np.mean(pnls)
    print(f"\n  Trades: {len(trades)}  Win rate: {wr:.0f}%  Avg: {avg:+.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Walk-forward backtest")
    p.add_argument("--save", action="store_true",
                   help="Write results JSON for dashboard")
    args = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv(cfg.root / ".env", override=True)

    logger.info("Loading data...")
    prices, vix = _load()
    price_idx   = _build_indexed(prices)
    zs          = _precompute_z(price_idx, cfg.signal)

    # ── STEP 1: Sweep on TRAIN data only ─────────────────────────────────────
    print(f"\n{'═'*54}")
    print(f"  STEP 1 — PARAMETER SWEEP (TRAIN: {TRAIN_START} → {TRAIN_END})")
    print(f"  Only training data. Test period never seen here.")
    print(f"{'═'*54}")
    print(f"  {'z_thresh':>9}  {'stop':>6}  "
          f"{'trades':>7}  {'wr%':>5}  {'sharpe':>7}  {'dd%':>7}  {'ann%':>7}")
    print(f"  {'─'*54}")

    sweep_results = []
    for z_thresh in [-2.0, -2.5, -3.0]:
        for stop in [-0.06, -0.08, -0.10]:
            _trade_cache.clear()
            sdf  = _signals(price_idx, zs, TRAIN_START, TRAIN_END, z_thresh, cfg.signal)
            if sdf.empty:
                continue
            res  = _sim_portfolio(sdf, price_idx, zs, TRAIN_START, TRAIN_END, stop, vix)
            m    = _metrics(res["trades"], res["equity"],
                            INITIAL_CAPITAL, TRAIN_START, TRAIN_END)
            mark = "✅" if m["passed"] else "  "
            print(f"  {mark} {z_thresh:>9.1f}  {stop:>6.2f}  "
                  f"{m['n_trades']:>7d}  {m['win_rate']:>5.1f}  "
                  f"{m['sharpe']:>7.2f}  {m['max_drawdown_pct']:>7.2f}  "
                  f"{m['annual_return_pct']:>7.2f}%")
            sweep_results.append({**m, "z_thresh": z_thresh, "stop": stop})

    # Pick best by Sharpe on train
    best_train = max(sweep_results, key=lambda x: x["sharpe"])
    best_z     = best_train["z_thresh"]
    best_stop  = best_train["stop"]
    print(f"\n  → Best on train: z={best_z}  stop={best_stop}  "
          f"sharpe={best_train['sharpe']:.2f}")

    # ── STEP 2: Validate (confirm params on unseen val year) ─────────────────
    _trade_cache.clear()
    print(f"\n{'═'*54}")
    print(f"  STEP 2 — VALIDATION (VAL: {VAL_START} → {VAL_END})")
    print(f"  Locked params from train. No re-optimisation.")
    print(f"{'═'*54}")
    val_sdf = _signals(price_idx, zs, VAL_START, VAL_END, best_z, cfg.signal)
    val_res = _sim_portfolio(val_sdf, price_idx, zs, VAL_START, VAL_END, best_stop, vix)
    val_m   = _metrics(val_res["trades"], val_res["equity"],
                       INITIAL_CAPITAL, VAL_START, VAL_END, "VALIDATION")
    _print(val_m)

    # ── STEP 3: OOS Test (true out-of-sample — never touched) ────────────────
    _trade_cache.clear()
    print(f"\n{'═'*54}")
    print(f"  STEP 3 — OOS TEST (TEST: {TEST_START} → {TEST_END})")
    print(f"  LOCKED params: z={best_z}  stop={best_stop}")
    print(f"  No optimisation. These numbers are unbiased.")
    print(f"{'═'*54}")
    test_sdf = _signals(price_idx, zs, TEST_START, TEST_END, best_z, cfg.signal)
    test_res = _sim_portfolio(test_sdf, price_idx, zs, TEST_START, TEST_END, best_stop, vix)
    test_m   = _metrics(test_res["trades"], test_res["equity"],
                        INITIAL_CAPITAL, TEST_START, TEST_END, "OOS TEST")
    _print(test_m)

    # Exit reason breakdown
    if test_res["trades"]:
        tdf = pd.DataFrame(test_res["trades"])
        print(f"\n  Exit Reasons (OOS Test):")
        for reason, grp in tdf.groupby("exit_reason"):
            wr  = (grp["net_ret"] > 0).mean() * 100
            avg = grp["net_ret"].mean() * 100
            print(f"    {reason:20s}  n={len(grp):4d}  wr={wr:.0f}%  avg={avg:+.2f}%")

    # ── STEP 4: Forward test (actual live paper trades) ───────────────────────
    print(f"\n{'═'*54}")
    print(f"  STEP 4 — FORWARD TEST (Actual paper trades, live system)")
    print(f"  Real signals from daily pipeline, no simulation.")
    print(f"{'═'*54}")
    fwd_trades = _load_forward()
    _print_forward(fwd_trades)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*54}")
    print(f"  SUMMARY  —  Mean Reversion Strategy")
    print(f"  Params: z < {best_z}  stop={best_stop*100:.0f}%  hold≤{MR_TIME_STOP}d")
    print(f"{'═'*54}")
    print(f"  {'Period':20s}  {'Sharpe':>7}  {'MaxDD':>7}  {'WinR':>6}  {'Ann%':>7}")
    print(f"  {'─'*54}")
    for lbl, m in [("Train (2020-21)",   best_train),
                   ("Val   (2022)",      val_m),
                   ("OOS   (2023-26)",   test_m)]:
        chk = "✅" if m.get("passed") else "❌"
        print(f"  {chk} {lbl:20s}  {m['sharpe']:>7.2f}  "
              f"{m['max_drawdown_pct']:>7.2f}%  {m['win_rate']:>6.1f}%  "
              f"{m['annual_return_pct']:>7.2f}%")

    if fwd_trades:
        pnls = [t.get("pnl_pct", 0) * 100 for t in fwd_trades]
        wr   = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg  = np.mean(pnls)
        print(f"  {'  Live forward':22s}  n={len(fwd_trades)}  wr={wr:.0f}%  avg={avg:+.1f}%  (actual)")

    # Save
    if args.save:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.results_dir / "strategy_backtest_test.json"
        out.write_text(json.dumps({
            "best_params":    {"z_thresh": best_z, "stop": best_stop},
            "train":          best_train,
            "validation":     val_m,
            "oos_test":       test_m,
            "forward_trades": fwd_trades,
            "oos_trades":     test_res["trades"],
            "oos_equity":     [(str(d.date() if hasattr(d,"date") else d), c)
                               for d, c in test_res["equity"]],
        }, indent=2, default=str))
        print(f"\nSaved → {out}")
