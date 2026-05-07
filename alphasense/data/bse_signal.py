"""
BSE Announcement → Trading Signal Converter
=============================================
Turns raw BSE corporate disclosures into four actionable outputs:

  1. blocklist_hits   — symbols to block from new BUY signals
                        (fraud, default, CEO/CFO resignation, regulatory probe)

  2. sentiment_boost  — symbols where BSE news is strongly positive
                        (buyback, special dividend, QIP at premium, rating upgrade)
                        → relaxes sentiment_threshold by +0.2 for 5 days

  3. sentiment_drag   — symbols where BSE news is negative
                        (credit downgrade, poor results flag)
                        → tightens zscore_threshold by -0.5 for 5 days

  4. earnings_due     — symbols reporting results today/tomorrow
                        → fetch audio for pipeline if available

Runs as part of daily post-close pipeline (step_bse_signals called in daily.py).
Results written to data/results/bse_signals_YYYYMMDD.json.
"""

import json, re, sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg


# ── Category → action map ─────────────────────────────────────────────────────

_BLOCKLIST_CATEGORIES = {
    "Fraud / Default",
    "SEBI / Regulatory",
    "Insider Trading",
}

_BLOCKLIST_SUBJECT_KW = [
    "fraud", "default", "nclt", "insolvency", "ed raid", "cbi",
    "enforcement directorate", "forensic audit", "whistleblower",
    "sebi order", "sebi ban", "debarment", "promoter pledge",
    "resignation of ceo", "resignation of md", "resignation of cfo",
    "ceo resigned", "md resigned", "cfo resigned",
    "debt restructur", "npa",
]

_POSITIVE_SUBJECT_KW = [
    "buyback", "buy-back", "special dividend", "interim dividend",
    "rating upgrade", "upgraded to", "record profit", "highest ever",
    "beat estimates", "strong growth", "order win", "large order",
    "qip at premium", "rights issue at premium",
]

_NEGATIVE_SUBJECT_KW = [
    "rating downgrade", "downgraded to", "profit warning", "results below",
    "below estimates", "loss reported", "widening loss", "revenue decline",
    "margin pressure", "guidance cut",
]

_EARNINGS_SUBJECT_KW = [
    "financial results", "quarterly results", "q1 results", "q2 results",
    "q3 results", "q4 results", "unaudited results", "audited results",
    "board meeting.*results", "results.*board meeting",
]


def _match_kw(text: str, keywords: list) -> bool:
    tl = text.lower()
    for kw in keywords:
        if re.search(kw, tl):
            return True
    return False


# ── Symbol lookup (BSE scrip_code → NSE symbol) ───────────────────────────────

def _build_scrip_map() -> dict[str, str]:
    """
    Map BSE scrip codes to NSE symbols.
    Uses NSE constituents CSV; falls back to company name matching.
    """
    mapping: dict[str, str] = {}
    csv_path = cfg.data_dir / "nse" / "constituents.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # BSE scrip code column may be present
        if "isin" in df.columns and "symbol" in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row.get("isin")):
                    mapping[str(row["isin"])] = str(row["symbol"])
    # Also load our parquet universe for name-based matching
    universe_path = cfg.data_dir / "universe.parquet"
    if universe_path.exists():
        udf = pd.read_parquet(universe_path)
        if "symbol" in udf.columns and "company_name" in udf.columns:
            for _, row in udf.iterrows():
                mapping[str(row["company_name"]).upper()] = str(row["symbol"])
    return mapping


def _company_to_symbol(company_name: str, scrip_map: dict) -> str:
    """Best-effort: match company name to NSE symbol."""
    cn = company_name.upper().strip()
    if cn in scrip_map:
        return scrip_map[cn]
    # Partial match — strip "LIMITED", "LTD", "INDIA", etc.
    clean = re.sub(r"\b(LIMITED|LTD|INDIA|INDUSTRIES|ENTERPRISES|CORP)\b", "", cn).strip()
    for key, sym in scrip_map.items():
        if clean and clean in key:
            return sym
    return ""


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_today(date: datetime = None) -> dict:
    """
    Load BSE announcements for today, classify each, return signal dict.
    """
    date = date or datetime.now()
    ds   = date.strftime("%Y%m%d")
    bse_file = cfg.bse_dir / f"bse_{ds}.json"

    if not bse_file.exists():
        logger.info(f"BSE file not found for {ds} — fetching now")
        try:
            from alphasense.data.bse_client import fetch_day
            anns    = fetch_day(date)
            records = [{"scrip_code": a.scrip_code, "company_name": a.company_name,
                        "subject": a.subject, "category": a.category,
                        "date": a.date.isoformat(), "body": a.body}
                       for a in anns]
            bse_file.parent.mkdir(parents=True, exist_ok=True)
            bse_file.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"BSE fetch failed: {e}")
            return _empty_result(date)

    try:
        records = json.loads(bse_file.read_text())
    except Exception:
        return _empty_result(date)

    scrip_map = _build_scrip_map()

    blocklist_hits:  dict[str, str]   = {}   # sym → reason
    sentiment_boost: dict[str, float] = {}   # sym → +threshold_delta
    sentiment_drag:  dict[str, float] = {}   # sym → -threshold_delta
    earnings_due:    list[str]        = []

    for rec in records:
        company  = rec.get("company_name", "")
        subject  = rec.get("subject", "")
        category = rec.get("category", "")
        body     = rec.get("body", "")
        text     = f"{subject} {body}"

        sym = _company_to_symbol(company, scrip_map)
        if not sym:
            continue

        # ── Blocklist ─────────────────────────────────────────────────────────
        if category in _BLOCKLIST_CATEGORIES or _match_kw(text, _BLOCKLIST_SUBJECT_KW):
            reason = f"BSE:{category}:{subject[:60]}"
            blocklist_hits[sym] = reason
            logger.warning(f"BSE blocklist: {sym} — {reason}")
            continue

        # ── Sentiment boost (positive) ────────────────────────────────────────
        if _match_kw(text, _POSITIVE_SUBJECT_KW):
            sentiment_boost[sym] = 0.2   # relax sentiment_threshold by 0.2
            logger.info(f"BSE boost: {sym} — {subject[:60]}")

        # ── Sentiment drag (negative) ─────────────────────────────────────────
        if _match_kw(text, _NEGATIVE_SUBJECT_KW):
            sentiment_drag[sym] = -0.3   # tighten zscore_threshold by 0.3
            logger.info(f"BSE drag: {sym} — {subject[:60]}")

        # ── Earnings due ──────────────────────────────────────────────────────
        if category == "Financial Results" or _match_kw(subject, _EARNINGS_SUBJECT_KW):
            if sym not in earnings_due:
                earnings_due.append(sym)

    result = {
        "date":            date.strftime("%Y-%m-%d"),
        "total_parsed":    len(records),
        "blocklist_hits":  blocklist_hits,
        "sentiment_boost": sentiment_boost,
        "sentiment_drag":  sentiment_drag,
        "earnings_due":    earnings_due,
    }

    out = cfg.results_dir / f"bse_signals_{ds}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info(f"BSE signals: {len(blocklist_hits)} blocked | "
                f"{len(sentiment_boost)} boosted | {len(sentiment_drag)} dragged | "
                f"{len(earnings_due)} earnings")
    return result


def load_recent(days: int = 5) -> dict:
    """
    Merge BSE signal files from the last N days.
    Used by SignalEngine to apply rolling blocklist/modifiers.
    """
    merged: dict = {
        "blocklist_hits":  {},
        "sentiment_boost": {},
        "sentiment_drag":  {},
        "earnings_due":    [],
    }
    for i in range(days):
        ds  = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        f   = cfg.results_dir / f"bse_signals_{ds}.json"
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
            merged["blocklist_hits"].update(d.get("blocklist_hits", {}))
            merged["sentiment_boost"].update(d.get("sentiment_boost", {}))
            merged["sentiment_drag"].update(d.get("sentiment_drag", {}))
            for sym in d.get("earnings_due", []):
                if sym not in merged["earnings_due"]:
                    merged["earnings_due"].append(sym)
        except Exception:
            pass
    return merged


def _empty_result(date: datetime) -> dict:
    return {
        "date":            date.strftime("%Y-%m-%d"),
        "total_parsed":    0,
        "blocklist_hits":  {},
        "sentiment_boost": {},
        "sentiment_drag":  {},
        "earnings_due":    [],
    }
