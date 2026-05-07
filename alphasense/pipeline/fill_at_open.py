"""
Fill Pending Orders at Market Open
=====================================
Runs at 09:30 IST (04:00 UTC) — 15 minutes after NSE opens at 09:15.

Fetches live OHLC from Groww API to get today's actual open price, then
fills any pending BUY orders. This is more accurate than the post-close
parquet approach because it uses the real observed open, not yesterday's close.

If Groww is unavailable, falls back to parquet open prices (same as daily.py).

Usage:
    python alphasense/pipeline/fill_at_open.py
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

cfg.logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    cfg.logs_dir / "fill_open_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="14 days", level="INFO",
)


def run():
    logger.info(f"\n{'#'*55}")
    logger.info(f"FILL-AT-OPEN  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")

    from alphasense.broker.kite import Broker

    broker = Broker()
    if not broker.paper or not broker.paper.pending:
        logger.info("No pending orders — nothing to do")
        return

    pending_syms = list(broker.paper.pending.keys())
    logger.info(f"Pending orders: {pending_syms}")

    open_prices: dict[str, float] = {}

    # ── 1. Try Groww live OHLC (today's actual open) ─────────────────────────
    try:
        from alphasense.data.groww_client import get_groww_client
        groww = get_groww_client()
        if groww.available:
            ohlc = groww.get_ohlc(pending_syms)
            for sym, d in ohlc.items():
                op = d.get("open", 0)
                if op and op > 0:
                    open_prices[sym] = float(op)
                    logger.info(f"  Groww open {sym}: ₹{op:.2f}")
            logger.info(f"Groww OHLC: {len(open_prices)}/{len(pending_syms)} symbols fetched")
        else:
            logger.info("Groww not available — will use parquet fallback")
    except Exception as e:
        logger.warning(f"Groww OHLC failed: {e}")

    # ── 2. Parquet fallback for any symbols Groww missed ─────────────────────
    missing = [s for s in pending_syms if s not in open_prices]
    if missing:
        nse_dir   = cfg.data_dir / "nse"
        today_str = pd.Timestamp.now().normalize()
        for sym in missing:
            p = nse_dir / f"{sym}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    if "open" in df.columns:
                        today_row = df[pd.to_datetime(df["date"]) >= today_str]
                        if not today_row.empty:
                            op = float(today_row["open"].iloc[0])
                            open_prices[sym] = op
                            logger.info(f"  Parquet open {sym}: ₹{op:.2f}")
                        else:
                            # Market just opened — parquet may not have today's row yet.
                            # Use latest close as proxy (conservative).
                            close = float(df["close"].dropna().iloc[-1])
                            open_prices[sym] = close
                            logger.info(f"  Parquet close (proxy) {sym}: ₹{close:.2f}")
                except Exception as e:
                    logger.warning(f"  Parquet fallback {sym}: {e}")

    if not open_prices:
        logger.warning("No open prices obtained — pending orders unchanged")
        return

    filled = broker.fill_pending(open_prices)
    logger.info(f"Fill-at-open complete: {filled} orders filled")


if __name__ == "__main__":
    run()
