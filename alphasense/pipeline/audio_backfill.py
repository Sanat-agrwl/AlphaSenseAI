"""
Earnings Audio Backfill
========================
One-time script to scan last 4 quarters of BSE announcements for
conference call / earnings call audio links and download them.

Covers all universe stocks, building the historical audio archive
from day one for future transcription and sentiment analysis.

Usage:
    python alphasense/pipeline/audio_backfill.py [--dry-run] [--quarters N]
"""

import sys, re, json, time
from pathlib import Path
from datetime import datetime, timedelta

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

cfg.logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    cfg.logs_dir / "audio_backfill_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="7 days", level="INFO",
)

AUDIO_DIR     = cfg.data_dir / "transcripts" / "audio"
MANIFEST_FILE = cfg.data_dir / "transcripts" / "manifest.json"

# Patterns that indicate a conference call audio link in BSE announcement text
AUDIO_URL_PATTERNS = [
    r"https?://[^\s\"'<>]+\.mp3[^\s\"'<>]*",
    r"https?://[^\s\"'<>]*(?:konfer|earncast|vcplayer|webcast|investor)[^\s\"'<>]*",
    r"https?://[^\s\"'<>]*(?:conf(?:erence)?[-_]?call|earnings[-_]?call)[^\s\"'<>]*\.(?:mp3|wav|aac)",
]

# Keywords in headline/body that suggest this is an earnings/conference call announcement
CALL_KEYWORDS = [
    "conference call", "earnings call", "investor call", "analyst call",
    "q1 results", "q2 results", "q3 results", "q4 results",
    "quarterly results", "annual results", "q1fy", "q2fy", "q3fy", "q4fy",
    "konfer", "earncast",
]

QUARTER_MAP = {
    1: "Q1", 2: "Q1", 3: "Q1",
    4: "Q4", 5: "Q4", 6: "Q4",   # actually Q4 is Jan-Mar, but approximate
    7: "Q1", 8: "Q1", 9: "Q1",   # Q1 = Apr-Jun
    10: "Q2", 11: "Q2", 12: "Q2",
}


def _quarter_label(date: datetime) -> str:
    m = date.month
    fy = date.year if date.month >= 4 else date.year - 1
    q  = "Q1" if m in (4,5,6) else "Q2" if m in (7,8,9) else "Q3" if m in (10,11,12) else "Q4"
    return f"{q}FY{str(fy+1)[-2:]}"


def _extract_audio_urls(text: str) -> list[str]:
    urls = []
    for pat in AUDIO_URL_PATTERNS:
        urls.extend(re.findall(pat, text, re.IGNORECASE))
    return list(dict.fromkeys(urls))   # dedup preserving order


def _is_call_announcement(headline: str, body: str) -> bool:
    combined = (headline + " " + body).lower()
    return any(kw in combined for kw in CALL_KEYWORDS)


def _load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except Exception:
            pass
    return {"entries": {}}


def _save_manifest(manifest: dict):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))


def _download_audio(url: str, dest: Path, dry_run: bool = False) -> bool:
    if dry_run:
        logger.info(f"    [dry-run] would download → {dest.name}")
        return True
    if dest.exists() and dest.stat().st_size > 10_000:
        logger.debug(f"    already exists: {dest.name}")
        return True
    try:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"    Downloading → {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        logger.info(f"    ✅ {dest.name} ({size_kb} KB)")
        return True
    except Exception as e:
        logger.warning(f"    ❌ Download failed {url}: {e}")
        return False


def run(quarters: int = 4, dry_run: bool = False):
    logger.info(f"\n{'#'*55}")
    logger.info(f"AUDIO BACKFILL  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Quarters: {quarters}  dry_run: {dry_run}")
    logger.info(f"{'#'*55}")

    # ── Load universe symbols ─────────────────────────────────────────────────
    try:
        import pandas as pd
        csv_path = cfg.data_dir / "nse" / "constituents.csv"
        if csv_path.exists():
            universe_syms = set(pd.read_csv(csv_path)["symbol"].tolist())
        else:
            universe_syms = {f.stem for f in (cfg.data_dir / "nse").glob("*.parquet")
                             if f.stem not in {"INDIAVIX", "india_vix"}}
        logger.info(f"Universe: {len(universe_syms)} symbols")
    except Exception as e:
        logger.warning(f"Could not load universe ({e}) — scanning all BSE data")
        universe_syms = None

    # ── Scan BSE announcement JSON files within the date window ──────────────
    cutoff = datetime.now() - timedelta(days=quarters * 92)
    bse_dir = cfg.data_dir / "bse"
    ann_files = sorted(bse_dir.glob("*.json")) if bse_dir.exists() else []
    logger.info(f"BSE announcement files found: {len(ann_files)}")

    manifest = _load_manifest()
    found, downloaded, skipped = 0, 0, 0

    for ann_file in ann_files:
        # Date filter from filename (bse_YYYYMMDD.json)
        try:
            file_date_str = ann_file.stem.replace("bse_", "")
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
            if file_date < cutoff:
                continue
        except Exception:
            pass  # if can't parse date, still scan it

        try:
            announcements = json.loads(ann_file.read_text())
        except Exception as e:
            logger.warning(f"Could not read {ann_file.name}: {e}")
            continue

        for ann in announcements:
            scrip_code = str(ann.get("scrip_code", ann.get("symbol", "")))
            symbol     = str(ann.get("company_name", scrip_code))
            headline   = str(ann.get("subject", ann.get("headline", "")))
            body       = str(ann.get("body", ""))
            ann_date   = ann.get("date", "")

            if not _is_call_announcement(headline, body):
                continue

            audio_urls = _extract_audio_urls(headline + " " + body)
            if not audio_urls:
                # No direct URL — log as a call without downloadable audio
                logger.debug(f"  {scrip_code} [{ann_date}]: call announcement but no audio URL")
                found += 1
                continue

            for url in audio_urls:
                try:
                    ann_dt = datetime.fromisoformat(str(ann_date)) if ann_date else datetime.now()
                except Exception:
                    ann_dt = datetime.now()
                quarter = _quarter_label(ann_dt)
                fname   = f"{scrip_code}_{quarter}.mp3"
                dest    = AUDIO_DIR / fname
                entry_key = f"{scrip_code}_{quarter}"

                if entry_key in manifest["entries"]:
                    skipped += 1
                    continue

                found += 1
                ok = _download_audio(url, dest, dry_run=dry_run)
                if ok:
                    manifest["entries"][entry_key] = {
                        "symbol":    scrip_code,
                        "quarter":   quarter,
                        "url":       url,
                        "file":      str(dest),
                        "date":      str(ann_dt.date()),
                        "status":    "downloaded" if not dry_run else "dry_run",
                        "fetched_at": datetime.now().isoformat(),
                    }
                    downloaded += 1
                    if not dry_run:
                        _save_manifest(manifest)
                    time.sleep(0.5)
                break   # one URL per announcement is enough

    _save_manifest(manifest)
    logger.info(f"\nBackfill complete:")
    logger.info(f"  Call announcements found:  {found}")
    logger.info(f"  Audio files downloaded:    {downloaded}")
    logger.info(f"  Already in manifest:       {skipped}")
    logger.info(f"  Manifest entries total:    {len(manifest['entries'])}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Earnings audio backfill")
    p.add_argument("--quarters", type=int, default=4,
                   help="Number of quarters to scan back (default 4)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Scan and report without downloading")
    args = p.parse_args()
    run(quarters=args.quarters, dry_run=args.dry_run)
