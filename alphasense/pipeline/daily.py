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

def step_fetch(extra_symbols: list[str] = None):
    """
    extra_symbols: pass Z-score candidates so we can fetch targeted
    Google News articles for stocks not covered by general RSS feeds.
    """
    logger.info("── STEP 1: Fetch data ──────────────────────────────────")

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
        articles = fetch_and_save(extra_symbols=extra_symbols)
        logger.info(f"News: {len(articles)} articles")
    except Exception as e:
        logger.error(f"News fetch failed: {e}")


# ─── Step 2: Score sentiment (ensemble: FinBERT + Claude if key available) ────

def step_sentiment() -> dict[str, float]:
    logger.info("── STEP 2: Score sentiment ─────────────────────────────")
    import os

    use_ensemble = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    if use_ensemble:
        logger.info("   Mode: Ensemble (FinBERT + LLM)")
        try:
            from alphasense.sentiment.ensemble import EnsembleScorer, save_ensemble_scores
            from alphasense.data.news_client   import load_news

            news = load_news(days=1)
            if news.empty:
                return {}

            articles = []
            for _, row in news.iterrows():
                syms = row.get("matched_symbols", [])
                if isinstance(syms, list) and syms:
                    articles.append({
                        "headline":        row.get("headline", ""),
                        "body":            row.get("body", ""),
                        "matched_symbols": syms,
                    })

            if not articles:
                logger.warning("No stock-matched articles for ensemble scoring")
                # Fall through to FinBERT below
                use_ensemble = False
            else:
                scorer  = EnsembleScorer()
                results = scorer.score_batch(articles)
                save_ensemble_scores(results)
                sentiment = scorer.aggregate_by_symbol(results)

                for sym, s in sorted(sentiment.items(), key=lambda x: x[1]):
                    if s < -0.4:
                        logger.info(f"  🔴 {sym}: {s:.3f}")
                    elif s > 0.3:
                        logger.info(f"  🟢 {sym}: {s:.3f}")
                logger.info(f"Sentiment (ensemble): {len(sentiment)} stocks scored")
                return sentiment
        except Exception as e:
            logger.error(f"Ensemble sentiment failed: {e} — falling back to FinBERT")

    # FinBERT-only fallback (no API keys, or ensemble failed)
    logger.info("   Mode: FinBERT only")
    try:
        from alphasense.sentiment.text import TextSentimentPipeline, aggregate_sentiment_by_stock
        pipe      = TextSentimentPipeline(use_finetuned=True)
        scored    = pipe.score_today()
        if scored.empty:
            return {}
        sentiment = aggregate_sentiment_by_stock(scored)
        for sym, s in sorted(sentiment.items(), key=lambda x: x[1]):
            if s < -0.4:
                logger.info(f"  🔴 {sym}: {s:.3f}")
            elif s > 0.3:
                logger.info(f"  🟢 {sym}: {s:.3f}")
        logger.info(f"Sentiment (FinBERT): {len(sentiment)} stocks scored")
        return sentiment
    except Exception as e:
        logger.error(f"Sentiment failed: {e}")
        return {}


# ─── Step 2b: Pre-signal Z-score scan (get candidates for targeted news fetch) ─

def _get_zscore_candidates() -> list[str]:
    """Quick Z-score scan to find candidate symbols before fetching targeted news."""
    try:
        from alphasense.screener.fundamental import load_universe
        from alphasense.data.nse_client      import load_prices
        from alphasense.signal.engine        import rolling_zscore
        from config.settings                 import cfg as _cfg

        universe = load_universe()
        if universe.empty:
            return []
        prices   = load_prices(universe["symbol"].tolist())
        sc       = _cfg.signal
        cands    = []
        for sym in universe["symbol"].tolist():
            if sym not in prices or prices[sym].empty:
                continue
            df = prices[sym]
            if len(df) < sc.rolling_window + 10:
                continue
            z = rolling_zscore(df["close"], sc.rolling_window, sc.return_period).iloc[-1]
            if not pd.isna(z) and z < sc.zscore_threshold + 0.5:   # slightly wider net
                cands.append(sym)
        logger.info(f"Z-score candidates for targeted news: {len(cands)} stocks")
        return cands
    except Exception:
        return []


# ─── Step 3: Generate signals ────────────────────────────────────────────────

def step_signals(sentiment: dict[str, float]) -> tuple[list, list]:
    """Returns (buy_signals, sell_signals)."""
    logger.info("── STEP 3: Generate signals ────────────────────────────")
    try:
        from alphasense.screener.fundamental import load_universe
        from alphasense.data.nse_client      import load_prices, load_vix
        from alphasense.signal.engine        import SignalEngine
        from alphasense.broker.kite          import Broker
        from alphasense.data.bse_signal      import analyze_today, load_recent

        universe = load_universe()
        if universe.empty:
            logger.warning("No universe — run build_universe.py first")
            return [], []

        symbols   = universe["symbol"].tolist()
        prices    = load_prices(symbols)
        vix_df    = load_vix()
        india_vix = float(vix_df["vix_close"].iloc[-1]) if not vix_df.empty else 15.0

        # BSE signals: today + last 5 days rolling
        try:
            analyze_today()
            bse_signals = load_recent(days=5)
            logger.info(f"BSE signals loaded: "
                        f"{len(bse_signals['blocklist_hits'])} blocked, "
                        f"{len(bse_signals['sentiment_boost'])} boosted, "
                        f"{len(bse_signals['sentiment_drag'])} dragged")
        except Exception as e:
            logger.warning(f"BSE signals failed (non-fatal): {e}")
            bse_signals = {}

        broker = Broker()
        broker_positions = broker.paper.positions if broker.paper else {}
        pending_count    = len(broker.paper.pending) if broker.paper else 0

        engine  = SignalEngine()
        signals = engine.generate(
            date=pd.Timestamp.now(),
            universe=universe,
            prices=prices,
            sentiment=sentiment,
            india_vix=india_vix,
            broker_positions=broker_positions,
            pending_count=pending_count,
            bse_signals=bse_signals,
        )

        buy_signals  = [s for s in signals if s.direction == "BUY"]
        sell_signals = [s for s in signals if s.direction == "SELL"]

        for s in sell_signals:
            logger.info(f"  🔴 SELL {s.symbol} @ ₹{s.exit_price:.2f} | "
                        f"PnL={s.pnl_pct*100:.1f}% | {s.exit_reason}")
        for s in buy_signals:
            sent_str = f"{s.sentiment_score:.2f}" if s.sentiment_score != 0.0 else "N/A (no news)"
            logger.info(f"  🟢 BUY {s.symbol} @ ₹{s.entry_price:.2f} | "
                        f"sent={sent_str} z={s.zscore:.2f}")

        logger.info(f"Signals: {len(buy_signals)} BUY, {len(sell_signals)} SELL")
        return buy_signals, sell_signals

    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return [], []


# ─── Step 3b: Execute SELL signals ───────────────────────────────────────────

def step_sell(sell_signals: list):
    """Execute SELL signals — exit open positions."""
    if not sell_signals:
        return
    logger.info("── STEP 3b: Execute SELL signals ───────────────────────")
    try:
        from alphasense.broker.kite import Broker
        broker = Broker()
        sold = 0
        for sig in sell_signals:
            pos = broker.paper.positions.get(sig.symbol)
            if pos is None:
                logger.warning(f"  {sig.symbol}: no open position to sell")
                continue
            result = broker.place(sig.symbol, "SELL", pos.qty, sig.exit_price,
                                  signal_id=pos.signal_id,
                                  exit_reason=sig.exit_reason or "")
            if result:
                sold += 1
        logger.info(f"Sold {sold}/{len(sell_signals)} positions")
    except Exception as e:
        logger.error(f"Sell execution failed: {e}")


# ─── Step 4: Stage signals (post-close) ──────────────────────────────────────

def step_execute(buy_signals: list):
    """Queue BUY signals as pending orders — filled at next post-close open price."""
    logger.info("── STEP 4: Stage BUY signals (fill at tomorrow's open) ─")
    try:
        from alphasense.broker.kite   import Broker
        from alphasense.signal.engine import position_size

        broker        = Broker()
        paper_capital = cfg.signal.paper_capital   # ₹10L (env: PAPER_CAPITAL)
        staged        = 0

        for sig in buy_signals:
            adv = 100_000
            qty = position_size(paper_capital, sig.entry_price, adv)
            if qty <= 0:
                continue
            sid = f"SIG-{datetime.now().strftime('%Y%m%d')}-{sig.symbol}"
            result = broker.stage(sig.symbol, qty, sig.entry_price, sid)
            if result:
                staged += 1

        logger.info(f"Staging complete: {staged} orders pending next-day open fill "
                    f"(paper capital ₹{paper_capital:,.0f})")
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
    step_fetch()   # update prices only — today's open not available yet (pre-market)


def run_post_close():
    logger.info(f"\n{'#'*55}")
    logger.info(f"POST-CLOSE  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")
    candidates = _get_zscore_candidates()
    step_fetch(extra_symbols=candidates)
    step_fill_pending()                      # fill yesterday's pending at today's open
    sentiment = step_sentiment()
    buy_signals, sell_signals = step_signals(sentiment)
    step_sell(sell_signals)                  # execute SELLs immediately
    step_execute(buy_signals)                # stage BUYs for tomorrow's open


def run_full():
    logger.info(f"\n{'#'*55}")
    logger.info(f"FULL RUN    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")
    candidates = _get_zscore_candidates()
    step_fetch(extra_symbols=candidates)
    step_fill_pending()
    sentiment = step_sentiment()
    buy_signals, sell_signals = step_signals(sentiment)
    step_sell(sell_signals)
    step_execute(buy_signals)


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