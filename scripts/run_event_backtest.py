"""
Event-Study Backtest — Earnings Surprise & Buyback Arbit
=========================================================
Uses NSE corporate events (Financial Results / Buyback) + NSE price data
to evaluate both event-driven strategies in a proper walk-forward setup.

Signal logic mirrors the live engine:
  EARNINGS_SURPRISE:
    - NSE event purpose contains "Financial Results"
    - Announcement-day 1d return ≥ +1% (positive surprise proxy)
    - Z-score at entry ≥ -1.5 (price not already depressed)
    Entry: next close after announcement date

  BUYBACK_ARBIT:
    - NSE event purpose contains "Buyback"
    - Z-score at entry < +1.0 (price hasn't fully run)
    Entry: next close after announcement date

Walk-forward:
  TRAIN  2024-01-01 → 2024-06-30  — param sweep only
  TEST   2024-07-01 → 2026-05-08  — locked params, unbiased OOS

Prerequisites:
    python scripts/fetch_nse_events.py --from-date 2024-01-01

Run:
    python scripts/run_event_backtest.py          # full report
    python scripts/run_event_backtest.py --save   # write JSON for dashboard
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

TRAIN_START, TRAIN_END = "2024-01-01", "2024-06-30"
TEST_START,  TEST_END  = "2024-07-01", "2026-05-08"

INITIAL_CAPITAL = 1_000_000
COST_BPS        = 50
PCT_PER_TRADE   = 0.08
MAX_POSITIONS   = 5


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


def _build_indexed(prices: dict) -> dict:
    out = {}
    for sym, df in prices.items():
        d = df.sort_values("date").copy()
        d.index = pd.to_datetime(d["date"])
        out[sym] = d
    return out


def _load_nse_events() -> pd.DataFrame:
    from scripts.fetch_nse_events import load_events
    evs = load_events()
    if not evs:
        raise RuntimeError(
            "No NSE events — run: python scripts/fetch_nse_events.py --from-date 2024-01-01"
        )
    df = pd.DataFrame(evs)
    # NSE date format: "01-Jan-2024"
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    logger.info(
        f"Loaded {len(df)} NSE events "
        f"({df['date'].min().date()} → {df['date'].max().date()})"
    )
    return df


# ── Signal extraction ──────────────────────────────────────────────────────────

def _precompute_z(price_idx: dict) -> dict:
    zs = {}
    sc = cfg.signal
    for sym, df in price_idx.items():
        if len(df) < 260:
            continue
        z = rolling_zscore(df["close"], sc.rolling_window, sc.return_period)
        z.index = df.index
        zs[sym] = z
    return zs


def _extract_earnings_signals(events: pd.DataFrame, price_idx: dict,
                               z_scores: dict,
                               start: str, end: str,
                               min_surprise_pct: float = 0.01) -> pd.DataFrame:
    mask = (
        events["purpose"].str.contains("Financial Results", na=False) &
        (events["date"] >= start) &
        (events["date"] <= end)
    )
    rows = events[mask].copy()
    signals = []

    for _, ev in rows.iterrows():
        sym = str(ev.get("symbol", "")).strip().upper()
        if sym not in price_idx:
            continue
        df = price_idx[sym]
        ann_date = pd.Timestamp(ev["date"].date())

        before      = df[df.index <  ann_date]
        on_or_after = df[df.index >= ann_date]
        if len(before) < 60 or len(on_or_after) == 0:
            continue

        prev_close  = float(before["close"].iloc[-1])
        ann_close   = float(on_or_after["close"].iloc[0])
        surprise_1d = (ann_close - prev_close) / prev_close

        if surprise_1d < min_surprise_pct:
            continue

        entry_date = on_or_after.index[0]
        z = z_scores.get(sym)
        if z is None:
            continue
        z_at = z.asof(entry_date) if entry_date in z.index or True else float("nan")
        if pd.isna(z_at) or float(z_at) < -1.5:
            continue

        signals.append({
            "symbol":      sym,
            "date":        entry_date,
            "entry_price": ann_close,
            "surprise_1d": round(surprise_1d, 4),
            "z_entry":     round(float(z_at), 3),
            "strategy":    "earnings_surprise",
        })

    df_sig = pd.DataFrame(signals)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(subset=["symbol", "date"])
    logger.info(f"Earnings signals [{start}→{end}]: {len(df_sig)}")
    return df_sig


def _extract_buyback_signals(events: pd.DataFrame, price_idx: dict,
                              z_scores: dict,
                              start: str, end: str) -> pd.DataFrame:
    mask = (
        events["purpose"].str.contains("Buyback", na=False) &
        (events["date"] >= start) &
        (events["date"] <= end)
    )
    rows = events[mask].copy()
    signals = []

    for _, ev in rows.iterrows():
        sym = str(ev.get("symbol", "")).strip().upper()
        if sym not in price_idx:
            continue
        df    = price_idx[sym]
        ann_d = pd.Timestamp(ev["date"].date())

        entry_bars = df[df.index >= ann_d]
        if len(entry_bars) == 0:
            continue

        entry_date  = entry_bars.index[0]
        entry_price = float(entry_bars["close"].iloc[0])

        z = z_scores.get(sym)
        if z is None:
            continue
        z_at = z.asof(entry_date)
        if pd.isna(z_at) or float(z_at) >= 1.0:
            continue

        signals.append({
            "symbol":      sym,
            "date":        entry_date,
            "entry_price": entry_price,
            "z_entry":     round(float(z_at), 3),
            "strategy":    "buyback_arbit",
        })

    df_sig = pd.DataFrame(signals)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(subset=["symbol", "date"])
    logger.info(f"Buyback signals [{start}→{end}]: {len(df_sig)}")
    return df_sig


# ── Trade simulation ───────────────────────────────────────────────────────────

def _simulate_trade(sym: str, entry_date: pd.Timestamp, entry_price: float,
                    price_idx: dict, stop_loss: float,
                    time_stop: int, profit_exit: float) -> dict | None:
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
        pnl = (float(row["close"]) - entry_price) / entry_price
        if pnl <= stop_loss:
            exit_price, exit_reason, days_held, exit_dt = (
                float(row["close"]), "stop_loss", i + 1, dt)
            break
        if pnl >= profit_exit:
            exit_price, exit_reason, days_held, exit_dt = (
                float(row["close"]), "profit_exit", i + 1, dt)
            break
        if i + 1 >= time_stop:
            exit_price, exit_reason, days_held, exit_dt = (
                float(row["close"]), "time_stop", i + 1, dt)
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
                   stop_loss: float, time_stop: int,
                   profit_exit: float) -> tuple[list[dict], list]:
    if signals.empty:
        return [], []

    trades   = []
    capital  = INITIAL_CAPITAL
    equity   = [(str(signals["date"].min().date()), capital)]
    open_pos = {}

    for _, sig in signals.sort_values("date").iterrows():
        date = sig["date"]
        open_pos = {s: d for s, d in open_pos.items() if d > date}
        if len(open_pos) >= MAX_POSITIONS or sig["symbol"] in open_pos:
            continue

        t = _simulate_trade(sig["symbol"], date, sig["entry_price"],
                            price_idx, stop_loss, time_stop, profit_exit)
        if t is None:
            continue

        capital += capital * PCT_PER_TRADE * t["pnl_pct"]
        equity.append((t["exit_date"], round(capital, 2)))
        open_pos[sig["symbol"]] = pd.Timestamp(t["exit_date"])
        trades.append(t)

    return trades, equity


# ── Stats ──────────────────────────────────────────────────────────────────────

def _stats(trades: list[dict], equity: list, label: str) -> dict:
    if not trades:
        return {"n_trades": 0, "label": label, "win_rate": 0,
                "sharpe": 0, "max_drawdown_pct": 0, "annual_return_pct": 0}

    pnls = [t["pnl_pct"] for t in trades]
    wins = sum(p > 0 for p in pnls)
    pf   = (sum(p for p in pnls if p > 0) /
            sum(abs(p) for p in pnls if p < 0)) if any(p < 0 for p in pnls) else float("inf")

    edf = pd.DataFrame(equity, columns=["date", "capital"])
    edf["date"] = pd.to_datetime(edf["date"])
    edf = edf.sort_values("date").drop_duplicates("date")
    edf["ret"] = edf["capital"].pct_change().fillna(0)
    sharpe = (edf["ret"].mean() / edf["ret"].std() * np.sqrt(252)
              if edf["ret"].std() > 0 else 0.0)
    edf["peak"] = edf["capital"].cummax()
    edf["dd"]   = (edf["capital"] - edf["peak"]) / edf["peak"] * 100
    max_dd      = float(edf["dd"].min())

    days = max(1, (edf["date"].iloc[-1] - edf["date"].iloc[0]).days)
    ann  = (edf["capital"].iloc[-1] / INITIAL_CAPITAL) ** (365 / days) - 1

    return {
        "label":            label,
        "n_trades":         len(trades),
        "win_rate":         round(wins / len(pnls) * 100, 1),
        "avg_pnl_pct":      round(np.mean(pnls) * 100, 2),
        "profit_factor":    round(pf, 2),
        "sharpe":           round(float(sharpe), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "annual_return_pct":round(ann * 100, 2),
        "equity_curve":     equity,
    }


def _print_block(label: str, r: dict):
    if r.get("n_trades", 0) == 0:
        print(f"\n  {label}: ⚠  No trades")
        return
    ok = lambda v: "✅" if v else "❌"
    print(f"\n{'─'*54}")
    print(f"  {label}")
    print(f"{'─'*54}")
    print(f"  Trades:         {r['n_trades']}")
    print(f"  Win Rate:      {r['win_rate']:.1f}%  {ok(r['win_rate'] > 50)}")
    print(f"  Avg PnL/trade: {r['avg_pnl_pct']:+.2f}%")
    print(f"  Profit Factor: {r['profit_factor']:.2f}")
    print(f"  Sharpe:        {r['sharpe']:.2f}  {ok(r['sharpe'] > 1.5)}")
    print(f"  Max Drawdown: {r['max_drawdown_pct']:.1f}%  {ok(r['max_drawdown_pct'] > -15)}")
    print(f"  Annual Return: {r['annual_return_pct']:.1f}%")


# ── Param sweep ────────────────────────────────────────────────────────────────

def _sweep(signals: pd.DataFrame, price_idx: dict,
           stop_g: list, time_g: list, profit_g: list) -> tuple:
    best = (cfg.signal.earnings_stop_loss_pct,
            cfg.signal.earnings_time_stop_days,
            cfg.signal.earnings_profit_exit_pct, -999.0)
    print(f"\n  {'stop':>6} {'days':>5} {'profit':>7} "
          f"{'n':>5} {'wr%':>6} {'sharpe':>7} {'dd%':>7}")
    print(f"  {'─'*46}")
    for stop in stop_g:
        for days in time_g:
            for profit in profit_g:
                trades, equity = _run_portfolio(signals, price_idx, stop, days, profit)
                if len(trades) < 3:
                    continue
                r = _stats(trades, equity, "sweep")
                mark = "✅" if r["sharpe"] > best[3] else "  "
                print(f"  {mark} {stop:>6.2f} {days:>5} {profit:>7.2f} "
                      f"{r['n_trades']:>5} {r['win_rate']:>5.1f}% "
                      f"{r['sharpe']:>7.2f} {r['max_drawdown_pct']:>6.1f}%")
                if r["sharpe"] > best[3]:
                    best = (stop, days, profit, r["sharpe"])
    print(f"\n  → Best: stop={best[0]}  days={best[1]}  profit={best[2]}  sharpe={best[3]:.2f}")
    return best


# ── Main ───────────────────────────────────────────────────────────────────────

def main(save: bool = False):
    prices    = _load_prices()
    price_idx = _build_indexed(prices)
    events    = _load_nse_events()
    z_scores  = _precompute_z(price_idx)

    print("\n" + "═"*56)
    print("  EVENT-STUDY BACKTEST — Earnings Surprise + Buyback Arbit")
    print(f"  Source: NSE corporate events calendar")
    print(f"  TRAIN: {TRAIN_START} → {TRAIN_END}")
    print(f"  TEST:  {TEST_START} → {TEST_END}")
    print("═"*56)

    CONFIGS = [
        {
            "key":     "earnings_surprise",
            "label":   "Earnings Surprise",
            "extract": lambda s, e: _extract_earnings_signals(
                events, price_idx, z_scores, s, e),
            "stop_g":   [-0.04, -0.06, -0.08],
            "time_g":   [7, 10, 15],
            "profit_g": [0.08, 0.10, 0.15],
        },
        {
            "key":     "buyback_arbit",
            "label":   "Buyback Arbit",
            "extract": lambda s, e: _extract_buyback_signals(
                events, price_idx, z_scores, s, e),
            "stop_g":   [-0.04, -0.06, -0.08],
            "time_g":   [10, 15, 20],
            "profit_g": [0.08, 0.10, 0.15],
        },
    ]

    results = {}

    for cfg_ in CONFIGS:
        key   = cfg_["key"]
        label = cfg_["label"]

        print(f"\n\n{'═'*56}")
        print(f"  {label.upper()}")
        print(f"{'═'*56}")

        # STEP 1: Sweep on TRAIN
        print(f"\n  STEP 1 — PARAM SWEEP  (TRAIN: {TRAIN_START} → {TRAIN_END})")
        train_sigs = cfg_["extract"](TRAIN_START, TRAIN_END)
        if train_sigs.empty:
            print("  ⚠  No signals in training period — cannot evaluate strategy.")
            results[key] = {"status": "no_data", "label": label}
            continue

        best_stop, best_days, best_profit, _ = _sweep(
            train_sigs, price_idx, cfg_["stop_g"], cfg_["time_g"], cfg_["profit_g"]
        )
        train_t, train_eq = _run_portfolio(train_sigs, price_idx,
                                           best_stop, best_days, best_profit)
        train_r = _stats(train_t, train_eq, f"TRAIN {label}")
        _print_block(f"TRAIN  {TRAIN_START} → {TRAIN_END}", train_r)

        # STEP 2: OOS TEST — locked params
        print(f"\n  STEP 2 — OOS TEST  "
              f"stop={best_stop}  days={best_days}  profit={best_profit}")
        test_sigs = cfg_["extract"](TEST_START, TEST_END)
        test_t, test_eq = _run_portfolio(test_sigs, price_idx,
                                         best_stop, best_days, best_profit)
        test_r = _stats(test_t, test_eq, f"OOS TEST {label}")
        _print_block(f"OOS TEST  {TEST_START} → {TEST_END}", test_r)

        passes = (
            test_r.get("n_trades", 0) >= 5 and
            test_r.get("win_rate", 0) > 50 and
            test_r.get("sharpe", 0) > 1.5 and
            test_r.get("max_drawdown_pct", -99) > -15
        )
        verdict = "✅ PASS — activate strategy" if passes else "❌ FAIL — keep inactive"
        print(f"\n  Verdict: {verdict}")

        results[key] = {
            "status":      "pass" if passes else "fail",
            "label":       label,
            "best_stop":   best_stop,
            "best_days":   best_days,
            "best_profit": best_profit,
            "train":       {k: v for k, v in train_r.items() if k != "equity_curve"},
            "oos_test":    {k: v for k, v in test_r.items() if k != "equity_curve"},
            "oos_test_equity": test_eq,
            "trades":      test_t,
        }

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*56}")
    print("  FINAL SUMMARY")
    print(f"{'═'*56}")
    print(f"  {'Strategy':<22} {'N':>5} {'WinR':>6} {'Sharpe':>8} {'MaxDD':>8}  Verdict")
    print(f"  {'─'*56}")
    for key, r in results.items():
        if r.get("status") == "no_data":
            print(f"  {r['label']:<22}  — no events data")
            continue
        oos = r.get("oos_test", {})
        v   = "✅ PASS" if r.get("status") == "pass" else "❌ FAIL"
        print(f"  {r['label']:<22} {oos.get('n_trades',0):>5} "
              f"{oos.get('win_rate',0):>5.1f}% "
              f"{oos.get('sharpe',0):>8.2f} "
              f"{oos.get('max_drawdown_pct',0):>7.1f}%  {v}")

    if save:
        out = cfg.results_dir / "event_backtest.json"
        payload = {
            "generated":   datetime.now().isoformat(),
            "train_start": TRAIN_START, "train_end": TRAIN_END,
            "test_start":  TEST_START,  "test_end":  TEST_END,
        }
        for key, r in results.items():
            payload[key] = {k: v for k, v in r.items()
                            if k not in ("trades", "oos_test_equity")}
            payload[f"{key}_trades"]        = r.get("trades", [])
            payload[f"{key}_equity_curve"]  = r.get("oos_test_equity", [])
        out.write_text(json.dumps(payload, indent=2))
        logger.success(f"Saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    main(save=args.save)
