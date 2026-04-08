"""Score news sentiment using FinBERT pipeline."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import cfg

p = argparse.ArgumentParser(description="Score news sentiment with FinBERT")
p.add_argument("--score-today", action="store_true",
               help="Score today's news articles")
p.add_argument("--finetune",    action="store_true",
               help="Fine-tune FinBERT on labeled_events (price-as-label)")
p.add_argument("--aggregate",   action="store_true",
               help="Print per-stock sentiment aggregation")
p.add_argument("--use-base",    action="store_true",
               help="Use base FinBERT (skip fine-tuned model)")
args = p.parse_args()

use_finetuned = not args.use_base

if args.finetune:
    from alphasense.sentiment.text import SentimentFineTuner
    import pandas as pd
    events_path = cfg.data_dir / "labeled_events.parquet"
    if not events_path.exists():
        print("labeled_events.parquet not found — run build_labels.py --build first")
        sys.exit(1)
    df = pd.read_parquet(events_path)
    tuner = SentimentFineTuner()
    tuner.train(df)
    print("Fine-tuning complete")

elif args.score_today:
    from alphasense.sentiment.text import TextSentimentPipeline, aggregate_sentiment_by_stock
    pipe   = TextSentimentPipeline(use_finetuned=use_finetuned)
    scored = pipe.score_today()
    if scored.empty:
        print("No news to score today — run fetch_news.py --fetch first")
    else:
        print(f"Scored {len(scored)} articles")
        cols = [c for c in ["published_at", "symbol", "headline", "sentiment_score"] if c in scored.columns]
        print(scored[cols].sort_values("sentiment_score").head(20).to_string())

        if args.aggregate:
            sentiment = aggregate_sentiment_by_stock(scored)
            print("\nPer-stock aggregation:")
            for sym, s in sorted(sentiment.items(), key=lambda x: x[1]):
                bar = "🔴" if s < -0.4 else ("🟢" if s > 0.3 else "⚪")
                print(f"  {bar} {sym:20s}  {s:+.3f}")

elif args.aggregate:
    from alphasense.sentiment.text import load_scored_news, aggregate_sentiment_by_stock
    df = load_scored_news()
    if df.empty:
        print("No scored news found — run --score-today first")
    else:
        sentiment = aggregate_sentiment_by_stock(df)
        print(f"Aggregated {len(sentiment)} stocks from {len(df)} rows")
        for sym, s in sorted(sentiment.items(), key=lambda x: x[1]):
            bar = "🔴" if s < -0.4 else ("🟢" if s > 0.3 else "⚪")
            print(f"  {bar} {sym:20s}  {s:+.3f}")

else:
    p.print_help()