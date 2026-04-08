"""Fetch NSE price data for all NIFTY 500 stocks."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphasense.data.nse_client import NSEClient, fetch_constituents, fetch_vix, backfill
from config.settings import cfg

p = argparse.ArgumentParser()
p.add_argument("--backfill",      action="store_true", help="Download all stocks since 2015")
p.add_argument("--constituents",  action="store_true", help="Refresh NIFTY 500 list only")
p.add_argument("--vix",           action="store_true", help="Refresh India VIX")
p.add_argument("--symbol",        type=str, help="Fetch a single symbol")
p.add_argument("--from-date",     default="01-01-2015")
args = p.parse_args()

client = NSEClient()

if args.constituents:
    df = fetch_constituents(client)
    df.to_parquet(cfg.nse_dir / "nifty500_constituents.parquet", index=False)
    print(df.head(10).to_string())

elif args.vix:
    df = fetch_vix(client)
    df.to_parquet(cfg.nse_dir / "india_vix.parquet", index=False)
    print(f"VIX: {len(df)} rows saved")

elif args.symbol:
    from datetime import datetime
    from alphasense.data.nse_client import fetch_history
    df = fetch_history(client, args.symbol, args.from_date,
                       datetime.now().strftime("%d-%m-%Y"))
    print(df.tail(10).to_string())

elif args.backfill:
    backfill(from_date=args.from_date)

else:
    p.print_help()