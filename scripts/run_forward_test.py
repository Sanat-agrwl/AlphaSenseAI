"""Run forward validation test (2025–present, fully out-of-sample)."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphasense.backtest.forward_test import ForwardTester
from config.settings import cfg

p = argparse.ArgumentParser(description="Forward validation test")
p.add_argument("--start", default="2025-01-01")
p.add_argument("--end",   default="2026-04-11")
p.add_argument("--with-sentiment", action="store_true",
               help="Load latest ensemble sentiment scores")
args = p.parse_args()

tester = ForwardTester()
tester.load()

sentiment = None
if args.with_sentiment:
    # Try ensemble scores first (Claude + GPT-4o + FinBERT)
    from alphasense.sentiment.ensemble import aggregate_sentiment
    sentiment = aggregate_sentiment(days=30)

    # Fall back to FinBERT scored_news.parquet if no ensemble scores
    if not sentiment:
        from alphasense.sentiment.text import load_scored_news, aggregate_sentiment_by_stock
        scored = load_scored_news()
        if not scored.empty:
            sentiment = aggregate_sentiment_by_stock(scored, lookback_days=30)
            print(f"Using FinBERT scores (no ensemble scores found)")

    print(f"Loaded sentiment for {len(sentiment)} stocks")

result = tester.run(start=args.start, end=args.end,
                    sentiment_override=sentiment)
tester.save(result)

if result.signals:
    import pandas as pd
    sdf = pd.DataFrame([s.__dict__ for s in result.signals])
    print("\nTop 10 best trades:")
    print(sdf.nlargest(10, "return_pct")[["date","symbol","zscore","return_pct","exit_reason"]].to_string(index=False))
    print("\nTop 10 worst trades:")
    print(sdf.nsmallest(10, "return_pct")[["date","symbol","zscore","return_pct","exit_reason"]].to_string(index=False))
