"""Build labeled events parquet — news/BSE events paired with forward price returns."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphasense.data.labeler import build_and_save
from config.settings import cfg

p = argparse.ArgumentParser(description="Build labeled_events.parquet for backtesting/fine-tuning")
p.add_argument("--build", action="store_true",
               help="Rebuild labeled_events.parquet from BSE + news + prices")
p.add_argument("--stats", action="store_true",
               help="Print statistics on the existing labeled_events file")
args = p.parse_args()

if args.build:
    import time
    t0 = time.time()
    df = build_and_save()
    elapsed = time.time() - t0
    print(f"\nLabeled events: {len(df)} rows in {elapsed:.1f}s")
    print(f"Saved to: {cfg.data_dir / 'labeled_events.parquet'}")
    if not df.empty:
        print("\nSample:")
        cols = [c for c in ["date", "symbol", "headline", "zscore",
                             "return_5d", "return_10d", "return_20d"] if c in df.columns]
        print(df[cols].head(10).to_string())

elif args.stats:
    import pandas as pd
    path = cfg.data_dir / "labeled_events.parquet"
    if not path.exists():
        print("labeled_events.parquet not found — run --build first")
    else:
        df = pd.read_parquet(path)
        print(f"Total events:       {len(df)}")
        print(f"Unique stocks:      {df['symbol'].nunique()}")
        print(f"Date range:         {df['date'].min()} → {df['date'].max()}")
        if "return_20d" in df.columns:
            print(f"Avg 20d return:     {df['return_20d'].mean()*100:.2f}%")
            print(f"Win rate (20d>0):   {(df['return_20d']>0).mean()*100:.1f}%")
        if "zscore" in df.columns:
            print(f"Avg Z-score:        {df['zscore'].mean():.2f}")
            below = (df["zscore"] < -2.0).sum()
            print(f"Events z<-2.0:      {below} ({below/len(df)*100:.1f}%)")
else:
    p.print_help()