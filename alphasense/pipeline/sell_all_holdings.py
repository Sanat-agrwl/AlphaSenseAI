"""
One-Time Portfolio Liquidation
================================
Sells every Groww holding at market open.
Run once on 2026-05-08 at 09:35 IST (04:05 UTC) after sell_all_holdings cron.

After all sells are placed it writes data/real_state.json with
  capital = estimated_proceeds
This becomes the starting cash for all future real Groww trades.

Usage:
    python alphasense/pipeline/sell_all_holdings.py
"""

import json, sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

cfg.logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    cfg.logs_dir / "liquidation_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="7 days", level="INFO",
)

REAL_STATE_FILE = cfg.data_dir / "real_state.json"
DONE_FLAG       = cfg.data_dir / ".liquidation_done"   # prevent double-run


def run():
    logger.info(f"\n{'#'*55}")
    logger.info(f"LIQUIDATION  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")

    if DONE_FLAG.exists():
        logger.info("Liquidation already completed (flag file present) — skipping.")
        return

    from dotenv import load_dotenv
    load_dotenv(cfg.root / ".env", override=True)

    from alphasense.data.groww_client import GrowwClient
    groww = GrowwClient()

    if not groww.available:
        logger.error("Groww client not available — check GROWW_API_KEY + GROWW_TOTP_SECRET")
        return

    holdings_df = groww.get_holdings()
    if holdings_df.empty:
        logger.info("No holdings found — nothing to liquidate.")
        _write_real_state(0.0)
        DONE_FLAG.touch()
        return

    logger.info(f"Holdings to sell: {len(holdings_df)}")
    logger.debug(f"Columns: {holdings_df.columns.tolist()}")

    # ── Normalise column names across Groww SDK versions ──────────────────────
    col_sym = _find_col(holdings_df, ["tradingSymbol", "trading_symbol", "symbol", "Symbol"])
    col_qty = _find_col(holdings_df, ["quantity", "Quantity", "qty"])
    col_avg = _find_col(holdings_df, ["averagePrice", "average_price", "avg_price",
                                       "avgPrice", "costPrice"])
    if not col_sym or not col_qty:
        logger.error(f"Cannot find symbol/qty columns. Got: {holdings_df.columns.tolist()}")
        logger.error("MANUAL ACTION REQUIRED: sell holdings manually in Groww app "
                     "and then set real_state.json capital to the actual proceeds.")
        return

    syms = [str(holdings_df.loc[i, col_sym]) for i in holdings_df.index
            if holdings_df.loc[i, col_sym]]
    ltp_map = groww.get_ltp(syms) if syms else {}

    total_proceeds = 0.0
    sold, failed = 0, 0

    for _, row in holdings_df.iterrows():
        sym = str(row[col_sym]) if col_sym else ""
        qty = int(row[col_qty]) if col_qty and not pd.isna(row.get(col_qty, 0)) else 0
        avg = float(row[col_avg]) if col_avg and not pd.isna(row.get(col_avg, 0)) else 0.0

        if not sym or qty <= 0:
            continue

        ltp      = ltp_map.get(sym, avg) or avg
        est_val  = qty * ltp

        logger.info(f"  Selling {sym} ×{qty} @ ₹{ltp:.2f}  est. ₹{est_val:,.0f}")
        order_id = groww.place_market_order(sym, qty, "SELL")
        if order_id:
            total_proceeds += est_val
            sold += 1
            logger.info(f"    ✅ {sym} order {order_id}")
        else:
            # Order may still go through even if we get no ID back — count proceeds
            total_proceeds += est_val
            failed += 1
            logger.warning(f"    ⚠️  {sym}: no order ID returned (order may still execute)")

    logger.info(f"\nLiquidation complete: {sold} confirmed, {failed} uncertain")
    logger.info(f"Estimated proceeds:   ₹{total_proceeds:,.0f}")

    if total_proceeds <= 0:
        logger.error("Proceeds are ₹0 — NOT writing real_state.json. "
                     "Check Groww holdings/orders manually before trading.")
        return

    _write_real_state(total_proceeds)
    DONE_FLAG.touch()
    logger.info("Done-flag written — script will skip on next invocation.")


def _find_col(df, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def _write_real_state(capital: float):
    existing_trades = []
    existing_orders = 0
    if REAL_STATE_FILE.exists():
        try:
            s = json.loads(REAL_STATE_FILE.read_text())
            existing_trades = s.get("trades", [])
            existing_orders = s.get("order_count", 0)
        except Exception:
            pass

    state = {
        "capital":               round(capital, 2),
        "positions":             {},
        "pending":               {},
        "trades":                existing_trades,
        "order_count":           existing_orders,
        "initialized_at":        datetime.now().isoformat(),
        "liquidation_proceeds":  round(capital, 2),
    }
    REAL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    REAL_STATE_FILE.write_text(json.dumps(state, indent=2))
    logger.info(f"real_state.json written → capital ₹{capital:,.0f}")


if __name__ == "__main__":
    run()
