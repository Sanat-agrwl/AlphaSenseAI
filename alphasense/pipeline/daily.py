"""
Daily Pipeline Orchestrator
============================
Ties every module together into a scheduled daily run.

Schedule:
  08:30 IST  Pre-market  — fetch fresh news + BSE announcements
  15:45 IST  Post-close  — score sentiment → generate signals → execute

Usage:
    python alphasense/pipeline/daily.py --pre-market
    python alphasense/pipeline/daily.py --post-close
    python alphasense/pipeline/daily.py --full
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

# Setup log file
cfg.logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    cfg.logs_dir / "daily_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days", level="INFO",
)


# ─── Step 1: Fetch data ───────────────────────────────────────────────────────

def step_fetch():
    logger.info("── STEP 1: Fetch data ──────────────────────────────────")

    # Update NSE prices first — signals depend on fresh close prices
    try:
        from alphasense.data.nse_client import update_prices
        n = update_prices(days=5)
        logger.info(f"Prices: {n} stocks updated")
    except Exception as e:
        logger.error(f"Price update failed: {e}")

    try:
        from alphasense.data.bse_client import fetch_day
        anns = fetch_day(datetime.now())
        logger.info(f"BSE: {len(anns)} announcements")
    except Exception as e:
        logger.error(f"BSE fetch failed: {e}")

    try:
        from alphasense.data.news_client import fetch_and_save
        articles = fetch_and_save()
        logger.info(f"News: {len(articles)} articles")
    except Exception as e:
        logger.error(f"News fetch failed: {e}")


# ─── Step 2: Score sentiment ─────────────────────────────────────────────────

def step_sentiment() -> dict[str, float]:
    logger.info("── STEP 2: Score sentiment ─────────────────────────────")
    try:
        from alphasense.sentiment.text import TextSentimentPipeline, aggregate_sentiment_by_stock
        pipe   = TextSentimentPipeline(use_finetuned=True)
        scored = pipe.score_today()
        if scored.empty:
            return {}
        sentiment = aggregate_sentiment_by_stock(scored)
        for sym, s in sorted(sentiment.items(), key=lambda x: x[1]):
            if s < -0.4:
                logger.info(f"  🔴 {sym}: {s:.3f}")
            elif s > 0.3:
                logger.info(f"  🟢 {sym}: {s:.3f}")
        logger.info(f"Sentiment: {len(sentiment)} stocks scored")
        return sentiment
    except Exception as e:
        logger.error(f"Sentiment failed: {e}")
        return {}


# ─── Step 3: Generate signals ────────────────────────────────────────────────

def step_signals(sentiment: dict[str, float]) -> list:
    logger.info("── STEP 3: Generate signals ────────────────────────────")
    try:
        from alphasense.screener.fundamental import load_universe
        from alphasense.data.nse_client      import load_prices, load_vix
        from alphasense.signal.engine        import SignalEngine

        universe = load_universe()
        if universe.empty:
            logger.warning("No universe — run build_universe.py first")
            return []

        symbols   = universe["symbol"].tolist()
        prices    = load_prices(symbols)
        vix_df    = load_vix()
        india_vix = float(vix_df["vix_close"].iloc[-1]) if not vix_df.empty else 15.0

        engine  = SignalEngine()
        signals = engine.generate(
            date=pd.Timestamp.now(),
            universe=universe,
            prices=prices,
            sentiment=sentiment,
            india_vix=india_vix,
        )

        for s in signals:
            icon = "🟢" if s.direction == "BUY" else "🔴"
            logger.info(f"  {icon} {s.direction} {s.symbol} @ ₹{s.entry_price:,.2f} "
                        f"sent={s.sentiment_score:.2f} z={s.zscore:.2f}")

        logger.info(f"Signals: {len(signals)} generated")
        return signals

    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return []


# ─── Step 4: Stage signals (post-close) ──────────────────────────────────────

def step_execute(signals: list):
    """Queue BUY signals as pending orders — filled at next-day open in pre-market."""
    logger.info("── STEP 4: Stage signals (fill tomorrow's open) ────────")
    try:
        from alphasense.broker.kite   import Broker
        from alphasense.signal.engine import position_size

        broker  = Broker()
        capital = cfg.backtest.capital
        staged  = 0

        for sig in signals:
            if sig.direction != "BUY":
                continue
            adv = 100_000  # fallback ADV
            qty = position_size(capital, sig.entry_price, adv)
            if qty <= 0:
                continue
            sid = f"SIG-{datetime.now().strftime('%Y%m%d')}-{sig.symbol}"
            result = broker.stage(sig.symbol, qty, sig.entry_price, sid)
            if result:
                staged += 1

        logger.info(f"Staging complete: {staged} orders pending next-day open fill")
    except Exception as e:
        logger.error(f"Execution failed: {e}")


# ─── Step 4b: Fill pending orders at today's open (pre-market) ───────────────

def step_fill_pending():
    """Fill any pending orders using today's open price from parquet."""
    logger.info("── STEP 4b: Fill pending orders at today's open ────────")
    try:
        from alphasense.broker.kite import Broker

        broker = Broker()
        if not broker.paper or not broker.paper.pending:
            logger.info("No pending orders to fill")
            return

        # Load today's open price from parquet for each pending symbol
        nse_dir    = cfg.data_dir / "nse"
        open_prices: dict[str, float] = {}
        for sym in list(broker.paper.pending.keys()):
            p = nse_dir / f"{sym}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    if "open" in df.columns and not df.empty:
                        today_str = pd.Timestamp.now().normalize()
                        today_row = df[pd.to_datetime(df["date"]) >= today_str]
                        if not today_row.empty:
                            open_prices[sym] = float(today_row["open"].iloc[0])
                        else:
                            # Market holiday or data not yet available — skip today
                            logger.info(f"  {sym}: no open price for today yet, stays pending")
                except Exception as e:
                    logger.warning(f"  {sym}: could not read open price: {e}")

        filled = broker.paper.fill_pending(open_prices)
        logger.info(f"Filled {filled} pending orders at today's open")
    except Exception as e:
        logger.error(f"Fill pending failed: {e}")


# ─── Orchestrators ────────────────────────────────────────────────────────────

def run_pre_market():
    logger.info(f"\n{'#'*55}")
    logger.info(f"PRE-MARKET  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")
    step_fetch()          # update prices first (get today's open)
    step_fill_pending()   # fill any pending orders at today's open


def run_post_close():
    logger.info(f"\n{'#'*55}")
    logger.info(f"POST-CLOSE  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")
    step_fetch()
    sentiment = step_sentiment()
    signals   = step_signals(sentiment)
    step_execute(signals)   # stages pending, fills tomorrow


def run_full():
    logger.info(f"\n{'#'*55}")
    logger.info(f"FULL RUN    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")
    step_fetch()
    step_fill_pending()
    sentiment = step_sentiment()
    signals   = step_signals(sentiment)
    step_execute(signals)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AlphaSense daily pipeline")
    p.add_argument("--pre-market",  action="store_true")
    p.add_argument("--post-close",  action="store_true")
    p.add_argument("--full",        action="store_true")
    args = p.parse_args()

    if   args.pre_market: run_pre_market()
    elif args.post_close:  run_post_close()
    elif args.full:        run_full()
    else:
        print("Usage: python daily.py [--pre-market | --post-close | --full]")