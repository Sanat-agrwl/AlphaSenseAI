"""Fetch news articles from RSS feeds and match to NIFTY 500 stocks."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphasense.data.news_client import fetch_and_save, load_news
from config.settings import cfg

p = argparse.ArgumentParser(description="Fetch news articles")
p.add_argument("--fetch", action="store_true", help="Fetch latest articles from all RSS feeds")
p.add_argument("--load",  action="store_true", help="Load and print stored articles")
p.add_argument("--tail",  type=int, default=20, help="Rows to print with --load")
p.add_argument("--symbol", type=str, help="Filter by stock symbol")
args = p.parse_args()

if args.fetch:
    articles = fetch_and_save()
    print(f"Saved {len(articles)} articles")
    if articles:
        import pandas as pd
        df = pd.DataFrame(articles)
        print(df[["published_at", "matched_symbols", "headline"]].tail(args.tail).to_string())

elif args.load:
    df = load_news()
    if df.empty:
        print("No news found — run --fetch first")
    else:
        print(f"Total rows: {len(df)}")
        if args.symbol:
            df = df[df["matched_symbols"].apply(
                lambda x: args.symbol in (x if isinstance(x, list) else [])
            )]
            print(f"Filtered to {len(df)} rows for {args.symbol}")
        print(df[["published_at", "matched_symbols", "headline"]].tail(args.tail).to_string())

else:
    p.print_help()