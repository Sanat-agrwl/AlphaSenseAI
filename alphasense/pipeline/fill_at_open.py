"""
Fill Pending Orders + Gap Fade Signals at Market Open
======================================================
Runs at 09:30 IST (04:00 UTC) — 15 minutes after NSE opens at 09:15.

Two jobs in one pass:
  1. Fill any pending BUY orders staged last evening at today's actual open.
  2. Scan universe for Gap Fade signals using live Groww OHLC:
       gap_pct = (today_open - yesterday_close) / yesterday_close
       Signal fires when gap_pct <= -3% AND quality_score >= 65 AND VIX < 28.
     Gap Fade is the only strategy that MUST use the real open price — all other
     strategies use end-of-day close and are handled in post-close.

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


def _load_universe() -> pd.DataFrame:
    p = cfg.data_dir / "universe.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def _load_vix() -> float:
    try:
        vdf = pd.read_parquet(cfg.nse_dir / "INDIAVIX.parquet")
        return float(vdf["vix_close"].dropna().iloc[-1])
    except Exception:
        return 18.0


def _prev_close(sym: str) -> float | None:
    """Return yesterday's closing price from parquet."""
    try:
        df = pd.read_parquet(cfg.nse_dir / f"{sym}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        today = pd.Timestamp.now().normalize()
        hist  = df[df["date"] < today].sort_values("date")
        if hist.empty:
            return None
        return float(hist["close"].iloc[-1])
    except Exception:
        return None


def _load_sentiment() -> dict[str, float]:
    """Load today's FinBERT sentiment scores if available."""
    sent_dir = cfg.data_dir / "sentiment"
    records: list[dict] = []
    if sent_dir.exists():
        today_str = datetime.now().strftime("%Y%m%d")
        for f in sorted(sent_dir.glob(f"ensemble_{today_str}*.json"), reverse=True)[:1]:
            try:
                import json
                records.extend(json.load(open(f)))
            except Exception:
                pass
    if not records:
        return {}
    df = pd.DataFrame(records)
    if "symbol" not in df.columns or "final_score" not in df.columns:
        return {}
    return df.groupby("symbol")["final_score"].mean().to_dict()


def _scan_gap_fades(groww, universe: pd.DataFrame,
                    existing_syms: set, vix: float) -> list[dict]:
    """
    Fetch live OHLC for all universe symbols, detect gap-down ≥ 3%,
    return list of gap-fade signal dicts ready to stage as BUY orders.
    """
    sc = cfg.signal
    if vix >= sc.vix_max:
        logger.info(f"VIX {vix:.1f} ≥ {sc.vix_max} — gap fade halted")
        return []

    if not groww.available:
        logger.warning("Groww unavailable — skipping gap fade scan")
        return []

    all_syms = universe["symbol"].tolist()
    logger.info(f"── Gap Fade scan: {len(all_syms)} symbols via Groww OHLC ──")

    # Fetch in batches of 50 (Groww limit)
    ohlc_map: dict[str, dict] = {}
    for i in range(0, len(all_syms), 50):
        batch = all_syms[i:i + 50]
        try:
            ohlc_map.update(groww.get_ohlc(batch))
        except Exception as e:
            logger.warning(f"  Groww OHLC batch {i//50 + 1}: {e}")

    logger.info(f"  OHLC fetched for {len(ohlc_map)}/{len(all_syms)} symbols")

    sentiment = _load_sentiment()
    signals   = []
    qual_map  = (universe.set_index("symbol")["quality_score"].to_dict()
                 if "quality_score" in universe.columns else {})

    for _, row in universe.iterrows():
        sym = row["symbol"]
        if sym in existing_syms:
            continue
        if sym not in ohlc_map:
            continue

        quality = float(qual_map.get(sym, 50.0))
        if quality < sc.gap_fade_quality_min:
            continue

        today_open = ohlc_map[sym].get("open", 0)
        if not today_open or today_open <= 0:
            continue

        prev_c = _prev_close(sym)
        if prev_c is None or prev_c <= 0:
            continue

        gap_pct = (today_open - prev_c) / prev_c
        if gap_pct > sc.gap_fade_threshold:   # threshold is -0.03 (negative)
            continue

        # Sentiment gate — skip if bad news caused the gap (score < -0.2)
        sent_score = sentiment.get(sym)
        if sent_score is not None and sent_score < -0.2:
            logger.info(f"  SKIP {sym}: gap={gap_pct*100:.1f}% but sentiment={sent_score:.2f} "
                        f"(likely fundamental — not a fade candidate)")
            continue

        # Dynamic stop/profit scaled to gap size
        gap_abs    = abs(gap_pct)
        stop_pct   = -round(max(0.02, min(gap_abs * 0.5, 0.05)), 3)   # 50% of gap, capped 2-5%
        profit_pct =  round(max(0.02, min(gap_abs * 0.7, 0.10)), 3)   # 70% of gap, capped 2-10%

        signals.append({
            "symbol":     sym,
            "gap_pct":    round(gap_pct * 100, 2),
            "prev_close": round(prev_c, 2),
            "open":       round(today_open, 2),
            "ltp":        round(today_open, 2),   # entry at today's open
            "quality":    round(quality, 1),
            "vix":        round(vix, 1),
            "stop_pct":   stop_pct,
            "profit_pct": profit_pct,
            "sentiment":  sent_score,
        })
        logger.info(
            f"  GAP FADE {sym}: gap={gap_pct*100:.1f}%  "
            f"prev_close=₹{prev_c:.2f} → open=₹{today_open:.2f}  "
            f"quality={quality:.0f}  stop={stop_pct*100:.1f}%  "
            f"profit={profit_pct*100:.1f}%"
            + (f"  sent={sent_score:.2f}" if sent_score is not None else "")
        )

    logger.info(f"Gap Fade signals found: {len(signals)}")
    return signals


def run():
    logger.info(f"\n{'#'*55}")
    logger.info(f"FILL-AT-OPEN  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")

    from alphasense.broker.kite import Broker
    from alphasense.data.groww_client import get_groww_client

    broker  = Broker()
    groww   = get_groww_client()

    # ── 1. Fill pending BUY orders ────────────────────────────────────────────
    paper        = broker._engine if hasattr(broker, "_engine") else None
    pending_syms = list(paper.pending.keys()) if paper else []

    if pending_syms:
        logger.info(f"Pending orders to fill: {pending_syms}")
        open_prices: dict[str, float] = {}

        if groww.available:
            ohlc = groww.get_ohlc(pending_syms)
            for sym, d in ohlc.items():
                op = d.get("open", 0)
                if op and op > 0:
                    open_prices[sym] = float(op)
                    logger.info(f"  Groww open {sym}: ₹{op:.2f}")
            logger.info(f"  Groww OHLC: {len(open_prices)}/{len(pending_syms)} fetched")
        else:
            logger.info("  Groww unavailable — using parquet fallback for pending fills")

        # Parquet fallback for any Groww misses
        for sym in [s for s in pending_syms if s not in open_prices]:
            prev = _prev_close(sym)
            if prev:
                open_prices[sym] = prev
                logger.info(f"  Parquet close (proxy) {sym}: ₹{prev:.2f}")

        if open_prices:
            filled = broker.fill_pending(open_prices)
            logger.info(f"Fill complete: {filled} orders filled")
        else:
            logger.warning("No open prices obtained — pending orders unchanged")
    else:
        logger.info("No pending orders to fill")

    # ── 2. Gap Fade scan at open ──────────────────────────────────────────────
    universe = _load_universe()
    if universe.empty:
        logger.warning("Universe empty — skipping gap fade scan")
        return

    vix            = _load_vix()
    paper          = broker._engine if hasattr(broker, "_engine") else None
    existing_syms  = set(paper.positions.keys()) if paper else set()
    n_open         = len(existing_syms)
    max_new        = max(0, cfg.signal.max_positions - n_open)

    if max_new == 0:
        logger.info(f"Max positions ({cfg.signal.max_positions}) reached — skipping gap fade")
        return

    gap_signals = _scan_gap_fades(groww, universe, existing_syms, vix)

    if not gap_signals:
        logger.info("No gap fade signals today")
        return

    # Sort by largest gap (most oversold first), cap at available slots
    gap_signals.sort(key=lambda x: x["gap_pct"])
    gap_signals = gap_signals[:max_new]

    # Stage each signal then immediately fill at today's open (same-day entry)
    fill_map: dict[str, float] = {}
    for sig in gap_signals:
        sym = sig["symbol"]
        ltp = sig["ltp"]
        qty = _calc_qty(ltp)
        if qty <= 0:
            logger.warning(f"  {sym}: qty=0 at ₹{ltp:.2f} — skipping")
            continue
        order_id = broker.stage(
            symbol=sym, qty=qty, signal_price=ltp,
            signal_id=f"GAP-{datetime.now().strftime('%Y%m%d')}-{sym}",
            stop_pct=sig["stop_pct"], profit_pct=sig["profit_pct"],
            strategy="gap_fade",
        )
        if order_id:
            fill_map[sym] = ltp   # fill at today's open immediately
            logger.info(
                f"  STAGED gap_fade {sym} × {qty} @ ₹{ltp:.2f} "
                f"(gap={sig['gap_pct']:.1f}%  stop={sig['stop_pct']*100:.1f}%  "
                f"profit={sig['profit_pct']*100:.1f}%)"
            )

    if fill_map:
        filled = broker.fill_pending(fill_map)
        logger.info(f"Gap Fade: {filled} positions filled at open")
    else:
        logger.info("Gap Fade: no valid orders staged")


def _calc_qty(price: float) -> int:
    """Position size: min(max_pct_capital * paper_capital, kelly) / price."""
    sc          = cfg.signal
    max_capital = sc.paper_capital * sc.max_pct_capital
    qty         = int(max_capital / price) if price > 0 else 0
    return max(0, qty)


if __name__ == "__main__":
    run()
