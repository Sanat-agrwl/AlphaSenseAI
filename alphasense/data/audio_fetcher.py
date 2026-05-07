"""
Audio Fetcher
=============
Monitors BSE announcements for earnings call audio links, downloads them,
and updates the transcript manifest for the AudioPipeline to process.

Sources scanned:
  • BSE announcement bodies — regex extracts mp3/wav/webcast URLs
  • Konfer.in public recordings (Indian earnings call platform)
  • EarnCast / BodhiTree platforms
  • Direct company IR page URLs

Output:
  data/transcripts/audio/{SYMBOL}_{QUARTER}.mp3   — downloaded audio
  data/transcripts/manifest.json                   — fetch status log

Usage:
    python alphasense/data/audio_fetcher.py --today
    python alphasense/data/audio_fetcher.py --symbols RELIANCE,INFY --quarter 2025Q1
"""

import json, re, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

MANIFEST_FILE = cfg.transcripts_dir / "manifest.json"
AUDIO_DIR     = cfg.transcripts_dir / "audio"

# Announcement categories / subject keywords that signal a conference call
_CALL_CATEGORIES = {"Audio/Visual Proceedings", "Transcript"}
_CALL_SUBJECT_KW = [
    "conference call", "analyst call", "investor call", "earnings call",
    "q1 call", "q2 call", "q3 call", "q4 call", "audio", "webcast",
    "konfer", "earncast", "bodhi", "recording",
]

# Regex patterns that pick up audio / webcast URLs from announcement bodies
_URL_PATTERNS = [
    # Direct audio files
    r'https?://[^\s"<>]+\.mp3',
    r'https?://[^\s"<>]+\.wav',
    r'https?://[^\s"<>]+\.m4a',
    # Known platforms
    r'https?://(?:www\.)?konfer\.in/[^\s"<>]+',
    r'https?://(?:www\.)?earncast\.in/[^\s"<>]+',
    r'https?://(?:www\.)?bodhitree\.com/[^\s"<>]+',
    r'https?://(?:www\.)?bluejeans\.com/[^\s"<>]+',
    # YouTube (some companies post there)
    r'https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+',
    r'https?://youtu\.be/[A-Za-z0-9_-]+',
    # Generic BSE / NSE hosted
    r'https?://[^\s"<>]*bseindia[^\s"<>]+\.(?:mp3|wav|mp4)',
    r'https?://[^\s"<>]*nseindia[^\s"<>]+\.(?:mp3|wav|mp4)',
]
_URL_RE = re.compile("|".join(_URL_PATTERNS), re.IGNORECASE)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


# ── Quarter helper ────────────────────────────────────────────────────────────

def _current_quarter(dt: datetime = None) -> str:
    """Returns e.g. '2025Q1' for Jan–Mar 2025."""
    dt  = dt or datetime.now()
    q   = (dt.month - 1) // 3 + 1
    # Indian FY: Q1 = Apr–Jun, Q2 = Jul–Sep, Q3 = Oct–Dec, Q4 = Jan–Mar
    fy  = dt.year if dt.month >= 4 else dt.year - 1
    fy_q = ((dt.month - 4) % 12) // 3 + 1
    return f"FY{fy}Q{fy_q}"


# ── Manifest I/O ──────────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except Exception:
            pass
    return {}   # {"{symbol}_{quarter}": {...}}


def _save_manifest(m: dict):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(m, indent=2, ensure_ascii=False))


def _manifest_key(symbol: str, quarter: str) -> str:
    return f"{symbol}_{quarter}"


def _record(manifest: dict, symbol: str, quarter: str,
            url: str = "", audio_path: str = "",
            status: str = "pending") -> None:
    key = _manifest_key(symbol, quarter)
    manifest[key] = {
        "symbol":     symbol,
        "quarter":    quarter,
        "url":        url,
        "audio_path": audio_path,
        "status":     status,
        "fetched_at": datetime.now().isoformat(),
    }


# ── URL extraction ────────────────────────────────────────────────────────────

def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text)


def _is_audio_url(url: str) -> bool:
    return bool(re.search(r'\.(mp3|wav|m4a)(\?|$)', url, re.IGNORECASE))


# ── Audio downloader ──────────────────────────────────────────────────────────

def _download_audio(url: str, dest: Path, timeout: int = 120) -> bool:
    """
    Downloads an audio file. Returns True on success.
    Skips non-audio URLs (webcast pages, YouTube — can't download those here).
    """
    if not _is_audio_url(url):
        logger.info(f"  Webcast URL (not a direct download): {url[:80]}")
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, headers=_HEADERS, timeout=timeout, stream=True)
        r.raise_for_status()
        size = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)
                size += len(chunk)
        logger.info(f"  Downloaded {dest.name} ({size/1024/1024:.1f} MB)")
        return True
    except Exception as e:
        logger.warning(f"  Download failed {url[:60]}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── BSE announcement scanner ──────────────────────────────────────────────────

def _scan_bse_file(bse_path: Path, manifest: dict,
                   target_symbols: set = None) -> list[dict]:
    """
    Scans one BSE JSON file for conference call links.
    Returns list of newly found items.
    """
    found = []
    try:
        records = json.loads(bse_path.read_text())
    except Exception:
        return found

    quarter = _current_quarter()

    for rec in records:
        category = rec.get("category", "")
        subject  = rec.get("subject", "")
        body     = rec.get("body", "")
        company  = rec.get("company_name", "")

        # Map to symbol — reuse bse_signal helper
        from alphasense.data.bse_signal import _build_scrip_map, _company_to_symbol
        scrip_map = _build_scrip_map()
        symbol    = _company_to_symbol(company, scrip_map)
        if not symbol:
            continue
        if target_symbols and symbol not in target_symbols:
            continue

        key = _manifest_key(symbol, quarter)
        if manifest.get(key, {}).get("status") in ("done", "transcribed"):
            continue

        is_call = (category in _CALL_CATEGORIES or
                   any(kw in subject.lower() for kw in _CALL_SUBJECT_KW) or
                   any(kw in body.lower()    for kw in _CALL_SUBJECT_KW[:5]))

        if not is_call:
            continue

        text = f"{subject} {body}"
        urls = _extract_urls(text)
        url  = urls[0] if urls else ""
        found.append({"symbol": symbol, "quarter": quarter, "url": url, "company": company})

    return found


# ── Main fetch routine ────────────────────────────────────────────────────────

def fetch_for_symbols(symbols: list[str], quarter: str = None,
                      days_lookback: int = 5) -> list[dict]:
    """
    Scan BSE files from the last N days for any conference call audio
    mentioning the given symbols. Downloads audio where possible.

    Returns list of manifest entries updated this run.
    """
    quarter  = quarter or _current_quarter()
    manifest = _load_manifest()
    target   = set(symbols)
    updated  = []

    for i in range(days_lookback):
        ds  = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        bsf = cfg.bse_dir / f"bse_{ds}.json"
        if not bsf.exists():
            continue
        items = _scan_bse_file(bsf, manifest, target_symbols=target)
        for item in items:
            sym  = item["symbol"]
            url  = item["url"]
            dest = AUDIO_DIR / f"{sym}_{quarter}.mp3"

            if url and _is_audio_url(url) and not dest.exists():
                ok = _download_audio(url, dest)
                if ok:
                    _record(manifest, sym, quarter, url, str(dest), "downloaded")
                    logger.info(f"Audio ready: {sym} {quarter}")
                else:
                    _record(manifest, sym, quarter, url, "", "url_found_no_download")
            elif url:
                # Webcast / YouTube — log it so user can manually download
                _record(manifest, sym, quarter, url, "", "webcast_url")
                logger.info(f"Webcast URL for {sym}: {url[:80]}")
            else:
                # Call announced but no URL found yet
                _record(manifest, sym, quarter, "", "", "pending")
                logger.info(f"Conference call noted for {sym} — no URL yet")

            updated.append(manifest[_manifest_key(sym, quarter)])

    _save_manifest(manifest)
    return updated


def fetch_today(extra_symbols: list[str] = None) -> list[dict]:
    """Convenience: fetch for today's earnings-due symbols + any extras."""
    from alphasense.data.bse_signal import load_recent
    signals  = load_recent(days=1)
    symbols  = list(signals.get("earnings_due", []))
    if extra_symbols:
        symbols += [s for s in extra_symbols if s not in symbols]
    if not symbols:
        logger.info("No earnings-due symbols today — nothing to fetch")
        return []
    logger.info(f"Audio fetch: {len(symbols)} earnings-due symbols")
    return fetch_for_symbols(symbols, days_lookback=3)


def get_downloadable_entries() -> list[dict]:
    """Return manifest entries with status='downloaded' that aren't yet transcribed."""
    manifest = _load_manifest()
    return [v for v in manifest.values()
            if v.get("status") == "downloaded"
            and Path(v.get("audio_path", "")).exists()]


def mark_transcribed(symbol: str, quarter: str):
    manifest = _load_manifest()
    key = _manifest_key(symbol, quarter)
    if key in manifest:
        manifest[key]["status"] = "transcribed"
        _save_manifest(manifest)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--today",   action="store_true", help="Fetch for today's earnings symbols")
    p.add_argument("--symbols", help="Comma-separated NSE symbols")
    p.add_argument("--quarter", help="e.g. FY2025Q4  (default: current quarter)")
    args = p.parse_args()

    if args.today:
        results = fetch_today()
    elif args.symbols:
        syms    = [s.strip().upper() for s in args.symbols.split(",")]
        results = fetch_for_symbols(syms, quarter=args.quarter, days_lookback=30)
    else:
        p.print_help()
        sys.exit(1)

    print(f"\n{len(results)} entries updated in manifest")
    for r in results:
        print(f"  {r['symbol']} {r['quarter']} — {r['status']}  {r.get('url','')[:60]}")
