"""
Quarterly Pipeline Orchestrator
=================================
Runs during Indian earnings seasons (April, July, October, January).

Jobs:
  1. XBRL refresh   — update quarterly financials for all universe stocks
  2. Audio fetch    — scan BSE for conference call audio links, download them
  3. Audio process  — transcribe + score downloaded audio via AudioPipeline
  4. Summary report — log confidence scores and fundamental signals

Schedule:
  Weekly (Sundays) : XBRL refresh
  Daily (earnings season): Audio fetch for today's earnings-due symbols
  On demand       : full run for a specific symbol / quarter

Usage:
    python alphasense/pipeline/quarterly.py --xbrl          # refresh XBRL only
    python alphasense/pipeline/quarterly.py --audio         # fetch + process audio
    python alphasense/pipeline/quarterly.py --full          # both
    python alphasense/pipeline/quarterly.py --symbol INFY   # single stock
"""

import sys
from pathlib import Path
from datetime import datetime

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

cfg.logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    cfg.logs_dir / "quarterly_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="60 days", level="INFO",
)


# ── Step 1: XBRL refresh ─────────────────────────────────────────────────────

def step_xbrl(symbols: list[str] = None, force: bool = False):
    logger.info("── STEP 1: XBRL / Quarterly financials refresh ─────────")
    try:
        from alphasense.data.xbrl_client import refresh_universe, refresh_symbol

        if symbols:
            for sym in symbols:
                logger.info(f"  Refreshing {sym}")
                refresh_symbol(sym, force=force)
        else:
            refresh_universe(force=force)

        logger.info("XBRL refresh complete")
    except Exception as e:
        logger.error(f"XBRL refresh failed: {e}")


# ── Step 2: Audio fetch ───────────────────────────────────────────────────────

def step_audio_fetch(symbols: list[str] = None) -> list[dict]:
    logger.info("── STEP 2: Audio fetch (BSE scan) ──────────────────────")
    try:
        from alphasense.data.audio_fetcher import fetch_today, fetch_for_symbols

        if symbols:
            results = fetch_for_symbols(symbols, days_lookback=7)
        else:
            results = fetch_today()

        downloaded = [r for r in results if r.get("status") == "downloaded"]
        webcast    = [r for r in results if r.get("status") == "webcast_url"]
        pending    = [r for r in results if r.get("status") == "pending"]

        logger.info(f"Audio fetch: {len(downloaded)} downloaded, "
                    f"{len(webcast)} webcast URLs, {len(pending)} pending")
        for r in webcast:
            logger.info(f"  Webcast (manual download needed): "
                        f"{r['symbol']} {r['quarter']} — {r['url'][:70]}")
        return results
    except Exception as e:
        logger.error(f"Audio fetch failed: {e}")
        return []


# ── Step 3: Audio transcribe + score ─────────────────────────────────────────

def step_audio_process():
    logger.info("── STEP 3: Audio transcription + scoring ───────────────")
    try:
        from alphasense.data.audio_fetcher import get_downloadable_entries, mark_transcribed
        from alphasense.sentiment.audio    import AudioPipeline

        entries = get_downloadable_entries()
        if not entries:
            logger.info("No downloaded audio to process")
            return

        pipeline = AudioPipeline()
        for entry in entries:
            sym     = entry["symbol"]
            quarter = entry["quarter"]
            path    = Path(entry["audio_path"])
            if not path.exists():
                continue
            try:
                logger.info(f"  Processing {sym} {quarter}...")
                result = pipeline.process(path, sym, quarter)
                mark_transcribed(sym, quarter)
                logger.info(f"  {sym} {quarter}: confidence={result.get('confidence_score')}/100")
            except Exception as e:
                logger.error(f"  {sym} {quarter} processing failed: {e}")

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")


# ── Step 4: Summary ───────────────────────────────────────────────────────────

def step_summary():
    logger.info("── STEP 4: Fundamental signals summary ─────────────────")
    try:
        from alphasense.screener.fundamental import load_universe
        from alphasense.signal.fundamental_signal import get_modifiers

        universe = load_universe()
        if universe.empty:
            return

        symbols   = universe["symbol"].tolist()
        modifiers = get_modifiers(symbols)

        boosted  = [(s, m) for s, m in modifiers.items() if m.sentiment_delta > 0 and not m.blocked]
        tightened = [(s, m) for s, m in modifiers.items() if m.sentiment_delta < 0 and not m.blocked]
        blocked  = [(s, m) for s, m in modifiers.items() if m.blocked]

        if blocked:
            logger.info("  Blocked stocks:")
            for s, m in blocked:
                logger.info(f"    🔴 {s}: {m.reason}")
        if boosted:
            logger.info("  Boosted stocks:")
            for s, m in boosted:
                logger.info(f"    🟢 {s}: {m.reason}")
        if tightened:
            logger.info("  Tightened stocks:")
            for s, m in tightened:
                logger.info(f"    🟡 {s}: {m.reason}")

    except Exception as e:
        logger.error(f"Summary failed: {e}")


# ── Orchestrators ─────────────────────────────────────────────────────────────

def run_xbrl_weekly():
    """Sunday XBRL refresh — refresh all universe fundamentals."""
    logger.info(f"\n{'#'*55}")
    logger.info(f"QUARTERLY-XBRL  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")
    step_xbrl()
    step_summary()


def run_audio_daily(symbols: list[str] = None):
    """Daily audio fetch (earnings season) — scan BSE + transcribe new audio."""
    logger.info(f"\n{'#'*55}")
    logger.info(f"QUARTERLY-AUDIO  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")
    step_audio_fetch(symbols)
    step_audio_process()


def run_full(symbols: list[str] = None):
    """Full quarterly run: XBRL + audio."""
    logger.info(f"\n{'#'*55}")
    logger.info(f"QUARTERLY-FULL  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")
    step_xbrl(symbols)
    step_audio_fetch(symbols)
    step_audio_process()
    step_summary()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AlphaSense quarterly pipeline")
    p.add_argument("--xbrl",   action="store_true", help="XBRL refresh only")
    p.add_argument("--audio",  action="store_true", help="Audio fetch + process only")
    p.add_argument("--full",   action="store_true", help="XBRL + audio")
    p.add_argument("--symbol", help="Single NSE symbol (skips full universe)")
    p.add_argument("--force",  action="store_true", help="Ignore XBRL cache")
    args = p.parse_args()

    syms = [args.symbol.upper()] if args.symbol else None

    if   args.xbrl:  step_xbrl(syms, force=args.force); step_summary()
    elif args.audio: run_audio_daily(syms)
    elif args.full:  run_full(syms)
    else:
        p.print_help()
