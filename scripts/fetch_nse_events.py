"""
NSE Corporate Events Historical Scraper
========================================
Scrapes NSE's event-calendar API month-by-month for Financial Results
and Buyback announcements. Saves to data/nse_events/ as JSON files.

Usage:
    python scripts/fetch_nse_events.py --from-date 2024-01-01  # default
    python scripts/fetch_nse_events.py --from-date 2023-01-01  # 2-year pull
"""

import sys, json, time, argparse
from pathlib import Path
from datetime import datetime, timedelta

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg

EVENTS_DIR = cfg.data_dir / "nse_events"
NSE_EVENTS_URL = "https://www.nseindia.com/api/event-calendar"
NSE_HOME = "https://www.nseindia.com"

RELEVANT_PURPOSES = {
    "Financial Results",
    "Financial Results/Dividend",
    "Financial Results/Buyback",
    "Financial Results/Dividend/Bonus",
    "Financial Results/Dividend/Fund Raising",
    "Financial Results/Dividend/Other business matters",
    "Financial Results/Fund Raising",
    "Financial Results/Fund Raising/Other business matters",
    "Financial Results/Other business matters",
    "Buyback",
    "Financial Results/Buyback",
}


def _make_session() -> requests.Session:
    s = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/",
        "Origin":          "https://www.nseindia.com",
    }
    s.headers.update(headers)
    try:
        s.get(NSE_HOME, timeout=15)
        time.sleep(1)
        s.get("https://www.nseindia.com/market-data/live-equity-market",
              timeout=10)
        time.sleep(0.5)
    except Exception as e:
        logger.warning(f"NSE session warmup: {e}")
    return s


def _fetch_month(session: requests.Session,
                 from_dt: datetime, to_dt: datetime,
                 retries: int = 3) -> list[dict]:
    from_s = from_dt.strftime("%d-%m-%Y")
    to_s   = to_dt.strftime("%d-%m-%Y")
    for i in range(retries):
        try:
            r = session.get(
                NSE_EVENTS_URL,
                params={"from_date": from_s, "to_date": to_s},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning(f"NSE events attempt {i+1} [{from_s}→{to_s}]: {e}")
            time.sleep(2 ** i)
            if i == 1:
                # Re-warm session on second retry
                session = _make_session()
    return []


def backfill(start_date: str = "2024-01-01"):
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.now()

    session  = _make_session()
    cur      = start.replace(day=1)
    total    = 0

    logger.info(f"NSE events backfill: {start_date} → {end.date()}")

    while cur <= end:
        # Month window
        next_month = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end  = min(next_month - timedelta(days=1), end)

        out_file = EVENTS_DIR / f"nse_events_{cur.strftime('%Y%m')}.json"
        if out_file.exists():
            logger.debug(f"  {cur.strftime('%Y-%m')}: already exists, skipping")
            cur = next_month
            continue

        raw = _fetch_month(session, cur, month_end)
        # Filter to relevant purposes
        relevant = [
            e for e in raw
            if any(rp in e.get("purpose", "") for rp in
                   ["Financial Results", "Buyback"])
        ]

        out_file.write_text(json.dumps(relevant, indent=2, ensure_ascii=False))
        total += len(relevant)
        logger.info(f"  {cur.strftime('%Y-%m')}: {len(raw)} events → "
                    f"{len(relevant)} relevant saved")

        cur = next_month
        time.sleep(1.2)   # be polite to NSE

    logger.success(f"NSE events backfill complete. {total} relevant events saved.")


def load_events(purpose_filter: str = None) -> list[dict]:
    """Load all saved NSE events into a flat list."""
    records = []
    for f in sorted(EVENTS_DIR.glob("nse_events_*.json")):
        try:
            records.extend(json.loads(f.read_text()))
        except Exception:
            pass
    if purpose_filter:
        records = [r for r in records if purpose_filter in r.get("purpose", "")]
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NSE corporate events scraper")
    p.add_argument("--from-date", default="2024-01-01",
                   help="Backfill start (YYYY-MM-DD)")
    p.add_argument("--list", action="store_true",
                   help="List saved events summary")
    args = p.parse_args()

    if args.list:
        evs = load_events()
        purposes = {}
        for e in evs:
            p_ = e.get("purpose", "unknown")
            purposes[p_] = purposes.get(p_, 0) + 1
        print(f"Total events: {len(evs)}")
        for k, v in sorted(purposes.items(), key=lambda x: -x[1])[:15]:
            print(f"  {v:5d}  {k}")
    else:
        backfill(start_date=args.from_date)
