"""
BSE Corporate Announcements Client
====================================
Free XML/JSON feed covering all listed company disclosures.
Saves daily JSON files to data/bse/.

Run:
    python scripts/fetch_bse.py --backfill --from 2015-01-01
    python scripts/fetch_bse.py --today
"""

import sys, json, time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

import requests
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

BSE_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.bseindia.com/",
    "Origin":     "https://www.bseindia.com",
}

RELEVANT_CATEGORIES = {
    "Board Meeting", "Financial Results", "Acquisition",
    "Resignation / Appointment", "Credit Rating", "Fraud / Default",
    "SEBI / Regulatory", "Dividend", "Buyback", "Change in Directors",
    "Scheme of Arrangement", "Insider Trading", "AGM / EGM",
}


@dataclass
class Announcement:
    scrip_code:   str
    company_name: str
    subject:      str
    category:     str
    date:         datetime
    body:         str = ""


def _parse_dt(s: str) -> Optional[datetime]:
    for fmt in ["%d/%m/%Y %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def _fetch_raw(from_dt: str, to_dt: str, retries: int = 3) -> list[dict]:
    params = {
        "strCat": "", "strPrevDate": from_dt, "strScrip": "",
        "strSearch": "P", "strToDate": to_dt, "strType": "C",
    }
    for i in range(retries):
        try:
            r = requests.get(BSE_URL, params=params, headers=_HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "Table" in data:
                return data["Table"]
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning(f"BSE attempt {i+1}: {e}")
            time.sleep(2 ** i)
    return []


def fetch_day(date: datetime) -> list[Announcement]:
    ds   = date.strftime("%d/%m/%Y")
    raw  = _fetch_raw(ds, ds)
    anns = []
    for item in raw:
        dt = _parse_dt(item.get("DisssemDT") or item.get("NEWS_DT", ""))
        if dt is None:
            continue
        anns.append(Announcement(
            scrip_code=str(item.get("SCRIP_CD", "")),
            company_name=item.get("SLONGNAME", item.get("COMPANY_NAME", "")),
            subject=item.get("HEADLINE", item.get("NEWSSUB", "")),
            category=item.get("CATEGORYNAME", item.get("CAT_NAME", "")),
            date=dt,
            body=item.get("NEWSDET", item.get("NEWS_BODY", ""))[:2000],
        ))
    logger.info(f"BSE {ds}: {len(anns)} announcements")
    return anns


def backfill(start: datetime, end: datetime = None):
    """Download day-by-day, skip weekends and existing files."""
    end = end or datetime.now()
    cfg.bse_dir.mkdir(parents=True, exist_ok=True)
    cur = start
    while cur <= end:
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue
        out = cfg.bse_dir / f"bse_{cur.strftime('%Y%m%d')}.json"
        if not out.exists():
            anns = fetch_day(cur)
            records = [{"scrip_code": a.scrip_code, "company_name": a.company_name,
                        "subject": a.subject, "category": a.category,
                        "date": a.date.isoformat(), "body": a.body}
                       for a in anns]
            with open(out, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            time.sleep(1.5)
        cur += timedelta(days=1)
    logger.success("BSE backfill complete")


def load_announcements(relevant_only: bool = True) -> pd.DataFrame:
    """Load all BSE JSON files into a DataFrame."""
    records = []
    for f in sorted(cfg.bse_dir.glob("bse_*.json")):
        with open(f) as fh:
            records.extend(json.load(fh))
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if relevant_only:
        df = df[df["category"].isin(RELEVANT_CATEGORIES)].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} BSE announcements")
    return df