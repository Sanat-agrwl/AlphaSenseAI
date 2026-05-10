"""
Event-Study Backtest — Earnings Surprise & Buyback Arbit
=========================================================
Uses 2-year historical BSE announcements + NSE price data to evaluate
the two event-driven strategies in a proper walk-forward setup.

Strategy logic mirrors the live engine exactly:
  EARNINGS_SURPRISE: "Financial Results" / "Board Meeting" category
                     with a large positive price surprise (proxy for
                     sentiment_delta until audio is available):
                       - 1-day return on announcement > +earnings_sent_delta_min*10 (1%)
                       - z-score at entry >= -1.5 (price not already crashed)
  BUYBACK_ARBIT:     "Buyback" category → buy within 3 days of announcement
                       - z < 1.0 (price hasn't run yet)
                       - Announcement boost proxy = 0.25 (fixed; BSE Buyback ≥ threshold)

Walk-forward:
  TRAIN  2024-01-01 → 2024-06-30  — param sweep (hold period, stop)
  TEST   2024-07-01 → 2026-05-08  — locked params, unbiased

Run:
    python scripts/run_event_backtest.py          # print report
    python scripts/run_event_backtest.py --save   # also write JSON for dashboard
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg
from alphasense.signal.engine import rolling_zscore

# ── Periods ────────────────────────────────────────────────────────────────────
TRAIN_START, TRAIN_END = "2024-01-01", "2024-06-30"
TEST_START,  TEST_END  = "2024-07-01", "2026-05-08"

INITIAL_CAPITAL = 1_000_000
COST_BPS        = 50
PCT_PER_TRADE   = 0.08
MAX_POSITIONS   = 5     # event strategies are low-frequency; smaller cap


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


def _load_bse() -> pd.DataFrame:
    from alphasense.data.bse_client import load_announcements
    df = load_announcements(relevant_only=False)
    if df.empty:
        raise RuntimeError("No BSE data — run: python scripts/fetch_bse.py --backfill --from-date 2024-01-01")
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded {len(df)} BSE announcements ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


def _build_indexed(prices):
    out = {}
    for sym, df in prices.items():
        d = df.sort_values("date").copy()
        d.index = pd.to_datetime(d["date"])
        out[sym] = d
    return out


def _price_on(price_idx, sym, date):
    """Return close price on or just after `date`."""
    df = price_idx.get(sym)
    if df is None or df.empty:
        return None
    future = df[df.index >= date]
    return float(future["close"].iloc[0]) if len(future) > 0 else None


def _scrip_to_symbol(scrip_code: str, prices: dict) -> str | None:
    """Best-effort map BSE scrip_code (numeric) → NSE symbol."""
    # Try direct match first (some scrapers store NSE symbols)
    if scrip_code in prices:
        return scrip_code
    # Try stored mapping file
    mapping_file = cfg.data_dir / "bse" / "scrip_symbol_map.json"
    if mapping_file.exists():
        mapping = json.loads(mapping_file.read_text())
        return mapping.get(scrip_code)
    return None


def _build_scrip_map(bse_df: pd.DataFrame, prices: dict) -> dict:
    """
    Build scrip_code → NSE symbol map by fuzzy-matching company names
    against price file stems. Saves result so subsequent runs are faster.
    """
    mapping_file = cfg.data_dir / "bse" / "scrip_symbol_map.json"
    if mapping_file.exists():
        return json.loads(mapping_file.read_text())

    # Build name→symbol index from price files
    nse_dir = cfg.data_dir / "nse"
    name_index = {}  # lower_name → symbol
    try:
        uni_csv = cfg.data_dir / "nse" / "constituents.csv"
        if uni_csv.exists():
            uni = pd.read_csv(uni_csv)
            for _, row in uni.iterrows():
                name = str(row.get("company_name", row.get("name", ""))).lower().strip()
                sym  = str(row.get("symbol", "")).strip()
                if name and sym:
                    name_index[name] = sym
                    # Also index first word (e.g. "reliance" → "RELIANCE")
                    first = name.split()[0] if name.split() else ""
                    if first and first not in name_index:
                        name_index[first] = sym
    except Exception as e:
        logger.warning(f"Could not build name index: {e}")

    mapping = {}
    for _, row in bse_df[["scrip_code","company_name"]].drop_duplicates().iterrows():
        code = str(row["scrip_code"])
        name = str(row["company_name"]).lower().strip()
        if code in mapping:
            continue
        # Exact name match
        if name in name_index:
            mapping[code] = name_index[name]
            continue
        # First-word match
        first = name.split()[0] if name.split() else ""
        if first and first in name_index:
            mapping[code] = name_index[first]

    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    mapping_file.write_text(json.dumps(mapping, indent=2))
    logger.info(f"Scrip→symbol map: {len(mapping)} entries")
    return mapping


# ── Signal extraction ──────────────────────────────────────────────────────────

def _extract_earnings_signals(bse_df: pd.DataFrame, price_idx: dict,
                               scrip_map: dict, start: str, end: str,
                               min_surprise_pct: float = 0.01) -> pd.DataFrame:
    """
    Proxy for sentiment_delta: announcement day 1-day return > min_surprise_pct
    and z ≥ -1.5 (price isn't already depressed).
    """
    mask = (
        bse_df["category"].isin({"Financial Results", "Board Meeting"}) &
        (bse_df["date"] >= start) &
        (bse_df["date"] <= end)
    )
    events = bse_df[mask].copy()

    signals = []
    for _, row in events.iterrows():
        sym = scrip_map.get(str(row["scrip_code"]))
        if not sym or sym not in price_idx:
            continue
        df = price_idx[sym]
        ann_date = pd.Timestamp(row["date"].date())

        # Price on and before announcement
        before = df[df.index < ann_date]
        on_or_after = df[df.index >= ann_date]
        if len(before) < 60 or len(on_or_after) == 0:
            continue

        prev_close  = float(before["close"].iloc[-1])
        ann_close   = float(on_or_after["close"].iloc[0])
        surprise_1d = (ann_close - prev_close) / prev_close

        if surprise_1d < min_surprise_pct:
            continue

        # Z-score at entry (must not be depressed — positive momentum)
        z = rolling_zscore(df["close"], window=252, period=5)
        z.index = df.index
        z_at_entry = z.reindex([on_or_after.index[0]], method="nearest")
        if len(z_at_entry) == 0 or pd.isna(z_at_entry.iloc[0]):
            continue
        z_val = float(z_at_entry.iloc[0])
        if z_val < -1.5:
            continue

        signals.append({
            "symbol":      sym,
            "date":        on_or_after.index[0],
            "entry_price": ann_close,
            "surprise_1d": surprise_1d,
            "z_entry":     z_val,
            "strategy":    "earnings_surprise",
        })

    df_sig = pd.DataFrame(signals)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(subset=["symbol","date"])
    logger.info(f"Earnings signals [{start}→{end}]: {len(df_sig)}")
    return df_sig


def _extract_buyback_signals(bse_df: pd.DataFrame, price_idx: dict,
                              scrip_map: dict, start: str, end: str) -> pd.DataFrame:
    """
    Enter within 1 trading day of a BSE Buyback announcement,
    when z < 1.0 (price hasn't fully run up yet).
    """
    mask = (
        bse_df["category"].str.contains("Buyback", case=False, na=False) &
        (bse_df["date"] >= start) &
        (bse_df["date"] <= end)
    )
    events = bse_df[mask].copy()

    signals = []
    for _, row in events.iterrows():
        sym = scrip_map.get(str(row["scrip_code"]))
        if not sym or sym not in price_idx:
            continue
        df = price_idx[sym]
        ann_date = pd.Timestamp(row["date"].date())

        # Entry = next trading day close
        entry_bars = df[df.index >= ann_date]
        if len(entry_bars) == 0:
            continue
        entry_price = float(entry_bars["close"].iloc[0])
        entry_date  = entry_bars.index[0]

        # Z-score: must be < 1.0 (price hasn't run)
        z = rolling_zscore(df["close"], window=252, period=5)
        z.index = df.index
        z_at_entry = z.reindex([entry_date], method="nearest")
        if len(z_at_entry) == 0 or pd.isna(z_at_entry.iloc[0]):
            continue
        if float(z_at_entry.iloc[0]) >= 1.0:
            continue

        signals.append({
            "symbol":      sym,
            "date":        entry_date,
            "entry_price": entry_price,
            "z_entry":     float(z_at_entry.iloc[0]),
            "strategy":    "buyback_arbit",
        })

    df_sig = pd.DataFrame(signals)
    if not df_sig.empty:
        df_sig = df_sig.drop_duplicates(subset=["symbol","date"])
    logger.info(f"Buyback signals [{start}→{end}]: {len(df_sig)}")
    return df_sig


# ── Trade simulation ───────────────────────────────────────────────────────────

def _simulate_trade(sym: str, entry_date: pd.Timestamp, entry_price: float,
                    price_idx: dict, stop_loss: float, time_stop: int,
                    profit_exit: float) -> dict | None:
    df = price_idx.get(sym)
    if df is None:
        return None
    future = df[df.index > entry_date].head(time_stop + 5)
    if future.empty:
        return None

    exit_price  = None
    exit_reason = "time_stop"
    days_held   = 0

    for i, (dt, row) in enumerate(future.iterrows()):
        days_held = i + 1
        pnl = (float(row["close"]) - entry_price) / entry_price
        if pnl <= stop_loss:
            exit_price  = float(row["close"])
            exit_reason = "stop_loss"
            break
        if pnl >= profit_exit:
            exit_price  = float(row["close"])
            exit_reason = "profit_exit"
            break
        if days_held >= time_stop:
            exit_price  = float(row["close"])
            exit_reason = "time_stop"
            break

    if exit_price is None:
        exit_price = float(future["close"].iloc[-1])

    cost   = COST_BPS / 10000
    pnl_net = (exit_price - entry_price) / entry_price - cost

    return {
        "symbol":       sym,
        "entry_date":   str(entry_date.date()),
        "exit_date":    str(dt.date()),
        "entry_price":  round(entry_price, 2),
        "exit_price":   round(exit_price, 2),
        "pnl_pct":      round(pnl_net, 4),
        "days_held":    days_held,
        "exit_reason":  exit_reason,
    }


def _run_portfolio(signals: pd.DataFrame, price_idx: dict,
                   stop_loss: float, time_stop: int,
                   profit_exit: float) -> tuple[list[dict], list]:
    """Run a simple portfolio simulation with MAX_POSITIONS cap."""
    if signals.empty:
        return [], []

    trades   = []
    capital  = INITIAL_CAPITAL
    equity   = [(str(signals["date"].min().date()), capital)]
    open_pos = {}  # sym → exit_date

    for _, sig in signals.sort_values("date").iterrows():
        date = sig["date"]
        # Clear expired positions
        open_pos = {s: d for s, d in open_pos.items() if d > date}
        if len(open_pos) >= MAX_POSITIONS:
            continue
        if sig["symbol"] in open_pos:
            continue

        t = _simulate_trade(
            sig["symbol"], date, sig["entry_price"], price_idx,
            stop_loss, time_stop, profit_exit,
        )
        if t is None:
            continue

        trade_capital = capital * PCT_PER_TRADE
        pnl_abs = trade_capital * t["pnl_pct"]
        capital += pnl_abs
        equity.append((t["exit_date"], round(capital, 2)))

        exit_dt = pd.Timestamp(t["exit_date"])
        open_pos[sig["symbol"]] = exit_dt
        trades.append(t)

    return trades, equity


def _stats(trades: list[dict], equity: list, label: str) -> dict:
    if not trades:
        logger.warning(f"{label}: 0 trades")
        return {"n_trades": 0, "label": label}

    pnls     = [t["pnl_pct"] for t in trades]
    wins     = sum(p > 0 for p in pnls)
    win_rate = wins / len(pnls) * 100
    avg_pnl  = np.mean(pnls) * 100
    profits  = [p for p in pnls if p > 0]
    losses   = [abs(p) for p in pnls if p < 0]
    pf       = (sum(profits) / sum(losses)) if losses else float("inf")

    # Sharpe from equity curve
    edf = pd.DataFrame(equity, columns=["date", "capital"])
    edf["date"] = pd.to_datetime(edf["date"])
    edf = edf.sort_values("date").drop_duplicates("date")
    edf["ret"] = edf["capital"].pct_change().fillna(0)
    sharpe = (edf["ret"].mean() / edf["ret"].std() * np.sqrt(252)
              if edf["ret"].std() > 0 else 0.0)

    # Max drawdown
    edf["peak"] = edf["capital"].cummax()
    edf["dd"]   = (edf["capital"] - edf["peak"]) / edf["peak"] * 100
    max_dd = float(edf["dd"].min())

    # Annual return
    days = (edf["date"].iloc[-1] - edf["date"].iloc[0]).days
    if days > 0:
        ann = (edf["capital"].iloc[-1] / edf["capital"].iloc[0]) ** (365 / days) - 1
    else:
        ann = 0.0

    return {
        "label":            label,
        "n_trades":         len(trades),
        "win_rate":         round(win_rate, 1),
        "avg_pnl_pct":      round(avg_pnl, 2),
        "profit_factor":    round(pf, 2),
        "sharpe":           round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "annual_return_pct":round(ann * 100, 2),
        "equity_curve":     equity,
    }


def _print_block(label: str, r: dict):
    ok = lambda v, t: "✅" if v else "❌"
    print(f"\n{'─'*54}")
    print(f"  {label}")
    print(f"{'─'*54}")
    if r.get("n_trades", 0) == 0:
        print("  ⚠  No trades in this period.")
        return
    print(f"  Trades:          {r['n_trades']}")
    print(f"  Win Rate:       {r['win_rate']:.1f}%  "
          f"{ok(r['win_rate'] > 50, 50)}")
    print(f"  Avg PnL/trade:  {r['avg_pnl_pct']:+.2f}%")
    print(f"  Profit Factor:  {r['profit_factor']:.2f}")
    print(f"  Sharpe:         {r['sharpe']:.2f}  "
          f"{ok(r['sharpe'] > 1.5, 1.5)}")
    print(f"  Max Drawdown:  {r['max_drawdown_pct']:.1f}%  "
          f"{ok(r['max_drawdown_pct'] > -15, -15)}")
    print(f"  Annual Return:  {r['annual_return_pct']:.1f}%")


# ── Parameter sweep ────────────────────────────────────────────────────────────

def _sweep(signals: pd.DataFrame, price_idx: dict,
           stop_grid: list, time_grid: list, profit_grid: list,
           label: str) -> tuple[float, int, float, float]:
    """Return best (stop, time_stop, profit_exit, sharpe) from grid."""
    best = (cfg.signal.earnings_stop_loss_pct,
            cfg.signal.earnings_time_stop_days,
            cfg.signal.earnings_profit_exit_pct, -999.0)
    print(f"\n  Param sweep — {label}")
    print(f"  {'stop':>6} {'days':>5} {'profit':>7} "
          f"{'n':>5} {'wr%':>6} {'sharpe':>7} {'dd%':>7}")
    print(f"  {'─'*52}")
    for stop in stop_grid:
        for days in time_grid:
            for profit in profit_grid:
                trades, equity = _run_portfolio(signals, price_idx,
                                                stop, days, profit)
                if len(trades) < 3:
                    continue
                r = _stats(trades, equity, label)
                mark = "✅" if r["sharpe"] > best[3] else "  "
                print(f"  {mark} {stop:>6.2f} {days:>5} {profit:>7.2f} "
                      f"{r['n_trades']:>5} {r['win_rate']:>5.1f}% "
                      f"{r['sharpe']:>7.2f} {r['max_drawdown_pct']:>6.1f}%")
                if r["sharpe"] > best[3]:
                    best = (stop, days, profit, r["sharpe"])
    print(f"\n  → Best: stop={best[0]}  days={best[1]}  profit={best[2]}  "
          f"sharpe={best[3]:.2f}")
    return best


# ── Main ───────────────────────────────────────────────────────────────────────

def main(save: bool = False):
    prices    = _load_prices()
    price_idx = _build_indexed(prices)
    bse_df    = _load_bse()
    scrip_map = _build_scrip_map(bse_df, prices)

    print("\n" + "═"*56)
    print("  EVENT-STUDY BACKTEST — Earnings Surprise + Buyback Arbit")
    print("  Walk-forward: TRAIN 2024-H1 → TEST 2024-H2 to 2026")
    print("═"*56)

    results = {}

    for strategy, extract_fn, stop_g, time_g, profit_g in [
        (
            "earnings_surprise",
            lambda s, e: _extract_earnings_signals(bse_df, price_idx, scrip_map, s, e),
            [-0.04, -0.06, -0.08],
            [7, 10, 15],
            [0.08, 0.10, 0.15],
        ),
        (
            "buyback_arbit",
            lambda s, e: _extract_buyback_signals(bse_df, price_idx, scrip_map, s, e),
            [-0.04, -0.06, -0.08],
            [10, 15, 20],
            [0.08, 0.10, 0.15],
        ),
    ]:
        strat_label = strategy.replace("_", " ").title()
        print(f"\n\n{'═'*56}")
        print(f"  {strat_label.upper()}")
        print(f"{'═'*56}")

        # ── TRAIN ────────────────────────────────────────────────────────────
        print(f"\n{'─'*56}")
        print(f"  STEP 1 — PARAM SWEEP  (TRAIN: {TRAIN_START} → {TRAIN_END})")
        print(f"{'─'*56}")
        train_sigs = extract_fn(TRAIN_START, TRAIN_END)
        if train_sigs.empty:
            print("  ⚠  No signals in training period — skipping strategy.")
            results[strategy] = {"status": "no_train_signals"}
            continue

        best_stop, best_days, best_profit, best_sh = _sweep(
            train_sigs, price_idx, stop_g, time_g, profit_g,
            f"TRAIN {strat_label}"
        )

        train_trades, train_eq = _run_portfolio(
            train_sigs, price_idx, best_stop, best_days, best_profit
        )
        train_r = _stats(train_trades, train_eq, f"TRAIN {strat_label}")
        _print_block(f"TRAIN  {TRAIN_START} → {TRAIN_END}", train_r)

        # ── TEST ─────────────────────────────────────────────────────────────
        print(f"\n{'─'*56}")
        print(f"  STEP 2 — OOS TEST  (TEST: {TEST_START} → {TEST_END})")
        print(f"  LOCKED params: stop={best_stop}  days={best_days}  profit={best_profit}")
        print(f"{'─'*56}")
        test_sigs = extract_fn(TEST_START, TEST_END)
        test_trades, test_eq = _run_portfolio(
            test_sigs, price_idx, best_stop, best_days, best_profit
        )
        test_r = _stats(test_trades, test_eq, f"OOS TEST {strat_label}")
        _print_block(f"OOS TEST  {TEST_START} → {TEST_END}", test_r)

        # ── Verdict ───────────────────────────────────────────────────────────
        passes = (
            test_r.get("win_rate", 0) > 50 and
            test_r.get("sharpe", 0) > 1.5 and
            test_r.get("max_drawdown_pct", -99) > -15
        )
        print(f"\n  Verdict: {'✅ PASS — strategy can go live' if passes else '❌ FAIL — do not activate'}")

        results[strategy] = {
            "status":       "pass" if passes else "fail",
            "best_stop":    best_stop,
            "best_days":    best_days,
            "best_profit":  best_profit,
            "train":        train_r,
            "oos_test":     test_r,
            "trades":       test_trades,
        }

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*56}")
    print("  SUMMARY")
    print(f"{'═'*56}")
    print(f"  {'Strategy':<22} {'Trades':>7} {'WinR':>6} {'Sharpe':>8} {'MaxDD':>8}  Verdict")
    print(f"  {'─'*56}")
    for strat, r in results.items():
        if r.get("status") == "no_train_signals":
            print(f"  {strat:<22}  — no signals in training period")
            continue
        oos = r.get("oos_test", {})
        v   = "✅ PASS" if r.get("status") == "pass" else "❌ FAIL"
        print(f"  {strat:<22} {oos.get('n_trades',0):>7} "
              f"{oos.get('win_rate',0):>5.1f}% "
              f"{oos.get('sharpe',0):>8.2f} "
              f"{oos.get('max_drawdown_pct',0):>7.1f}%  {v}")

    if save:
        out = cfg.results_dir / "event_backtest.json"
        save_data = {
            "generated":  datetime.now().isoformat(),
            "train_start": TRAIN_START, "train_end": TRAIN_END,
            "test_start":  TEST_START,  "test_end":  TEST_END,
        }
        for strat, r in results.items():
            save_data[strat] = {k: v for k, v in r.items() if k != "trades"}
            save_data[f"{strat}_trades"] = r.get("trades", [])
        out.write_text(json.dumps(save_data, indent=2))
        logger.success(f"Saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save", action="store_true", help="Write JSON results for dashboard")
    args = p.parse_args()
    main(save=args.save)
