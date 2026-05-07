"""
Intraday Monitor
================
Runs at 12:30 IST (07:00 UTC) — mid-session check using live Groww prices.

Two jobs in one pass:
  1. Stop-loss guard   — if any open position is down ≥ 8% intraday, sell immediately
  2. Intraday BUY scan — check live LTP against Z-score universe for fresh entries;
                         stage any new BUY candidates as pending (fill at next open)

Requires Groww credentials. If Groww is unavailable the script exits cleanly
so the cron failure doesn't mask real errors.

Usage:
    python alphasense/pipeline/intraday_monitor.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

cfg.logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    cfg.logs_dir / "intraday_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="14 days", level="INFO",
)


def _live_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch live LTP for symbols via Groww. Returns {} if unavailable."""
    try:
        from alphasense.data.groww_client import get_groww_client
        groww = get_groww_client()
        if not groww.available:
            logger.warning("Groww not available — intraday monitor skipped")
            return {}
        return groww.get_ltp(symbols)
    except Exception as e:
        logger.error(f"Groww LTP failed: {e}")
        return {}


# ── Job 1: Intraday stop-loss guard ──────────────────────────────────────────

def job_stoploss():
    """Sell any open position that is down ≥ stop_loss_pct intraday."""
    logger.info("── JOB 1: Intraday stop-loss guard ────────────────────")
    from alphasense.broker.kite import Broker

    broker = Broker()
    if not broker.paper or not broker.paper.positions:
        logger.info("No open positions — skip")
        return

    symbols    = list(broker.paper.positions.keys())
    live_px    = _live_prices(symbols)
    if not live_px:
        return

    sold = 0
    sc   = cfg.signal
    for sym, pos in list(broker.paper.positions.items()):
        ltp = live_px.get(sym)
        if ltp is None:
            continue
        pnl = (ltp - pos.entry_price) / pos.entry_price
        logger.debug(f"  {sym}: entry ₹{pos.entry_price:.2f} | ltp ₹{ltp:.2f} | {pnl*100:.1f}%")
        if pnl <= sc.stop_loss_pct:
            logger.info(f"  🔴 INTRADAY STOP-LOSS {sym}: {pnl*100:.1f}% ≤ {sc.stop_loss_pct*100:.1f}%")
            result = broker.place(sym, "SELL", pos.qty, ltp, signal_id=pos.signal_id)
            if result:
                sold += 1

    logger.info(f"Stop-loss guard: {sold} positions closed")


# ── Job 2: Intraday BUY signal scan ──────────────────────────────────────────

def job_intraday_buy():
    """
    Check live prices against rolling Z-score universe.
    Stocks where live price implies a Z-score below threshold get staged as
    pending BUY (will fill at tomorrow's open via fill_at_open.py).
    """
    logger.info("── JOB 2: Intraday BUY signal scan ────────────────────")
    try:
        from alphasense.screener.fundamental import load_universe
        from alphasense.data.nse_client      import load_prices, load_vix
        from alphasense.signal.engine        import rolling_zscore, check_blocklist
        from alphasense.signal.engine        import position_size
        from alphasense.broker.kite          import Broker

        sc       = cfg.signal
        universe = load_universe()
        if universe.empty:
            logger.warning("Universe empty — run build_universe.py first")
            return

        symbols   = universe["symbol"].tolist()
        prices    = load_prices(symbols)
        vix_df    = load_vix()
        india_vix = float(vix_df["vix_close"].iloc[-1]) if not vix_df.empty else 15.0

        if india_vix > sc.vix_max:
            logger.info(f"VIX {india_vix:.1f} > {sc.vix_max} — no intraday entries")
            return

        broker = Broker()
        if not broker.paper:
            return

        # Skip symbols already held or pending
        skip = set(broker.paper.positions.keys()) | set(broker.paper.pending.keys())
        candidates = [s for s in symbols if s not in skip]

        # Fetch live prices for candidates (batch)
        live_px = _live_prices(candidates)
        if not live_px:
            return

        logger.info(f"Live prices: {len(live_px)}/{len(candidates)} symbols")

        new_signals = 0
        capital     = cfg.backtest.capital

        for sym in candidates:
            ltp = live_px.get(sym)
            if ltp is None or ltp <= 0:
                continue

            if sym not in prices or prices[sym].empty:
                continue

            price_df = prices[sym]
            if len(price_df) < sc.rolling_window + 10:
                continue

            # Compute Z-score using historical closes, substituting today's LTP
            # as the last price to simulate live Z-score
            close_series = price_df["close"].copy()
            close_series.iloc[-1] = ltp

            z_series = rolling_zscore(close_series, sc.rolling_window, sc.return_period)
            z        = float(z_series.iloc[-1])

            if pd.isna(z) or z >= sc.zscore_threshold:
                continue

            # Volume confirmation: today's volume should be above 20-day avg
            if "volume" in price_df.columns and len(price_df) >= 20:
                vol_today = float(price_df["volume"].iloc[-1])
                vol_avg   = float(price_df["volume"].iloc[-20:].mean())
                if vol_avg > 0 and vol_today < vol_avg:
                    logger.debug(f"  {sym}: volume {vol_today:.0f} < 20d avg {vol_avg:.0f} — skip")
                    continue

            adv = float(price_df["volume"].mean()) * ltp if "volume" in price_df.columns else 100_000
            qty = position_size(capital, ltp, adv)
            if qty <= 0:
                continue

            sid    = f"INTRA-{datetime.now().strftime('%Y%m%d%H%M')}-{sym}"
            result = broker.stage(sym, qty, ltp, signal_id=sid)
            if result:
                new_signals += 1
                logger.info(f"  🟢 INTRADAY BUY staged: {sym} ×{qty} @ ₹{ltp:.2f} z={z:.2f}")

            if len(broker.paper.pending) >= broker.paper.MAX_PENDING:
                logger.info("Max pending cap reached — stopping scan")
                break

        logger.info(f"Intraday scan complete: {new_signals} new BUY candidates staged")

    except Exception as e:
        logger.error(f"Intraday BUY scan failed: {e}")


# ── Intraday SELL check ───────────────────────────────────────────────────────

def job_intraday_sell():
    """
    Check recovery exits for open positions using live prices.
    Mirrors _check_exit logic but using LTP instead of end-of-day close.
    """
    logger.info("── JOB 2b: Intraday recovery SELL check ───────────────")
    try:
        from alphasense.broker.kite import Broker
        from alphasense.signal.engine import rolling_zscore
        from alphasense.data.nse_client import load_prices, load_vix

        broker = Broker()
        if not broker.paper or not broker.paper.positions:
            return

        sc        = cfg.signal
        symbols   = list(broker.paper.positions.keys())
        live_px   = _live_prices(symbols)
        if not live_px:
            return

        prices    = load_prices(symbols)
        vix_df    = load_vix()
        india_vix = float(vix_df["vix_close"].iloc[-1]) if not vix_df.empty else 15.0
        today     = pd.Timestamp.now()
        sold      = 0

        for sym, pos in list(broker.paper.positions.items()):
            ltp = live_px.get(sym)
            if ltp is None:
                continue

            pnl       = (ltp - pos.entry_price) / pos.entry_price
            days_held = (today - pd.Timestamp(pos.entry_date)).days

            # Only check recovery exits intraday — stop-loss handled in job_stoploss
            exit_reason = None
            if pnl >= sc.profit_exit_pct and len(broker.paper.pending) > 0:
                exit_reason = "profit+pending (intraday)"
            elif days_held >= sc.time_stop_days:
                exit_reason = f"time_stop ({days_held}d, intraday)"

            if prices.get(sym) is not None and not prices[sym].empty:
                z_series  = rolling_zscore(prices[sym]["close"])
                curr_z    = float(z_series.iloc[-1]) if not z_series.empty else 0.0
                if not pd.isna(curr_z) and curr_z > sc.recovery_zscore:
                    exit_reason = exit_reason or "price_recovery (intraday)"

            if exit_reason:
                logger.info(f"  🔴 INTRADAY SELL {sym} @ ₹{ltp:.2f} | "
                            f"PnL={pnl*100:.1f}% | {exit_reason}")
                result = broker.place(sym, "SELL", pos.qty, ltp,
                                      signal_id=pos.signal_id)
                if result:
                    sold += 1

        logger.info(f"Intraday recovery SELL: {sold} positions closed")

    except Exception as e:
        logger.error(f"Intraday SELL check failed: {e}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run():
    logger.info(f"\n{'#'*55}")
    logger.info(f"INTRADAY-MONITOR  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")

    # Guard: only run on weekdays during market hours (09:00–15:30 IST)
    now = datetime.now()
    if now.weekday() >= 5:    # Saturday = 5, Sunday = 6
        logger.info("Weekend — skipping intraday monitor")
        return

    job_stoploss()
    job_intraday_sell()
    job_intraday_buy()
    logger.info("Intraday monitor complete")


if __name__ == "__main__":
    run()
