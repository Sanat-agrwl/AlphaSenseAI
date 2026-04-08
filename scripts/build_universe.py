"""Build the quality fundamental universe from NSE price data."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphasense.screener.fundamental import build_and_save_universe, load_universe
from config.settings import cfg

p = argparse.ArgumentParser(description="Build NIFTY 500 quality universe")
p.add_argument("--build",  action="store_true", help="Build and save universe.parquet")
p.add_argument("--load",   action="store_true", help="Print the saved universe")
p.add_argument("--tail",   type=int, default=30, help="Rows to print with --load")
args = p.parse_args()

if args.build:
    df = build_and_save_universe()
    print(f"\nUniverse built: {len(df)} stocks")
    print(df[["symbol", "tier", "quality_score"]].head(30).to_string())

elif args.load:
    df = load_universe()
    if df.empty:
        print("No universe found — run --build first")
    else:
        print(f"Universe: {len(df)} stocks")
        cols = [c for c in ["symbol", "tier", "quality_score", "sector"] if c in df.columns]
        print(df[cols].head(args.tail).to_string())

else:
    p.print_help()