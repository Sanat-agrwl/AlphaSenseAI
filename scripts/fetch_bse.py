"""Fetch BSE corporate announcements."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphasense.data.bse_client import fetch_today, backfill, load_announcements
from config.settings import cfg

p = argparse.ArgumentParser(description="Fetch BSE announcements")
p.add_argument("--today",    action="store_true", help="Fetch today's announcements")
p.add_argument("--backfill", action="store_true", help="Backfill from 2015")
p.add_argument("--from-date", default="2015-01-01", help="Backfill start date (YYYY-MM-DD)")
p.add_argument("--load",     action="store_true", help="Load and print stored announcements")
p.add_argument("--tail",     type=int, default=20, help="Rows to print with --load")
args = p.parse_args()

if args.today:
    anns = fetch_today()
    print(f"Fetched {len(anns)} announcements")
    if anns:
        import pandas as pd
        print(pd.DataFrame(anns).tail(args.tail).to_string())

elif args.backfill:
    backfill(from_date=args.from_date)

elif args.load:
    df = load_announcements()
    if df.empty:
        print("No announcements found — run --today or --backfill first")
    else:
        print(f"Total rows: {len(df)}")
        print(df.tail(args.tail).to_string())

else:
    p.print_help()