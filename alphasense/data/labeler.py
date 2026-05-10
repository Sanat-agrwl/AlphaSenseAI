"""
FinBERT Training Data Labeler
==============================
Incrementally labels news + BSE announcements with multi-source signals.
Outputs JSONL for FinBERT fine-tuning.

Label sources (highest → lowest confidence):
  price_outcome  — 5d + 20d forward return agreement    (high / medium)
  bse_category   — earnings/dividend/legal event type   (medium)
  keyword        — keyword heuristics                   (low)
  pseudo         — neutral catch-all                    (low)

Run:
    python alphasense/data/labeler.py [--backfill] [--report]

Output:
    data/labels/labeled_YYYYMMDD.jsonl   — daily incremental file
    data/labels/index.json               — hash tracker (prevents re-labeling)
"""

import sys, json, hashlib
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg
from alphasense.data.nse_client import load_prices

LABELS_DIR = cfg.data_dir / "labels"
INDEX_FILE  = LABELS_DIR / "index.json"

POSITIVE_KEYWORDS = [
    "record profit", "beat estimate", "strong earnings", "revenue growth",
    "raised guidance", "buyback", "dividend declared", "rating upgrade",
    "outperform", "strong results", "beats expectations", "new contract",
    "board approval", "bonus shares", "demerger", "stake sale", "capacity expansion",
]
NEGATIVE_KEYWORDS = [
    "net loss", "profit warning", "missed estimate", "revenue decline",
    "rating downgrade", "underperform", "fraud", "regulatory action", "penalty",
    "sebi notice", "default", "debt restructuring", "insolvency",
    "md resigns", "ceo quits", "plant shutdown", "asset sale under pressure",
]

BSE_POSITIVE_CATS = {"dividend", "buyback", "bonus", "merger-amalgamation", "results"}
BSE_NEGATIVE_CATS = {"insider-trading", "legal", "penalty", "fraud", "suspension"}


# ── Text helpers ──────────────────────────────────────────────────────────────

def _text_for_model(headline: str, body: str) -> str:
    """headline + first 200 + last 200 chars of body — fits 512-token FinBERT limit."""
    body = body or ""
    if len(body) <= 400:
        return f"{headline}. {body}".strip()[:600]
    return f"{headline}. {body[:200]} ... {body[-200:]}".strip()


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


# ── Label sources ──────────────────────────────────────────────────────────────

def _keyword_label(text: str) -> tuple[str, float]:
    text_l = text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_l)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_l)
    if pos > neg and pos >= 1:
        return "positive", 0.6
    if neg > pos and neg >= 1:
        return "negative", 0.6
    return "neutral", 0.4


def _bse_category_label(category: str) -> tuple[str, float] | None:
    cat_l = (category or "").lower()
    for c in BSE_POSITIVE_CATS:
        if c in cat_l:
            return "positive", 0.75
    for c in BSE_NEGATIVE_CATS:
        if c in cat_l:
            return "negative", 0.75
    return None


def _price_label(sym: str, date: pd.Timestamp,
                 prices: dict) -> tuple | None:
    """
    Returns (label, label_5d, label_20d, return_5d, return_20d, confidence) or None.
    confidence="high" if 5d+20d agree, "medium" if only one horizon available.
    """
    if sym not in prices:
        return None
    df = prices[sym]
    if df.empty or "close" not in df.columns:
        return None

    date_col   = pd.to_datetime(df["date"])
    future_idx = date_col[date_col >= date].index
    if len(future_idx) < 2:
        return None

    close = df["close"]
    entry = float(close.iloc[future_idx[0]])
    if entry == 0:
        return None

    def _fwd(n):
        i = future_idx[0] + n
        return float((close.iloc[i] - entry) / entry) if i < len(close) else None

    r5  = _fwd(5)
    r20 = _fwd(20)

    def _lbl(r):
        if r is None:  return None
        if r >  0.03:  return "positive"
        if r < -0.03:  return "negative"
        return "neutral"

    l5, l20 = _lbl(r5), _lbl(r20)
    if l5 is None:
        return None

    if l5 == l20:
        conf, label = "high", l5
    elif l20 is None:
        conf, label = "medium", l5
    else:
        conf, label = "medium", l5   # 5d is primary

    return label, l5, l20, r5, r20, conf


# ── Index (dedup tracker) ──────────────────────────────────────────────────────

def _load_index() -> set[str]:
    if not INDEX_FILE.exists():
        return set()
    try:
        return set(json.loads(INDEX_FILE.read_text()).get("hashes", []))
    except Exception:
        return set()


def _save_index(hashes: set[str]):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps({"hashes": sorted(hashes)}, indent=2))


# ── Core labeler ───────────────────────────────────────────────────────────────

def label_batch(articles: list[dict], prices: dict) -> list[dict]:
    """
    Label a batch of article dicts.
    Each dict needs: symbol, date (or published_at), headline, body, source.
    Returns list of labeled record dicts for JSONL output.
    """
    labeled = []
    for art in articles:
        sym      = str(art.get("symbol", "")).strip()
        raw_date = art.get("date") or art.get("published_at")
        try:
            date = pd.to_datetime(raw_date)
        except Exception:
            continue
        if pd.isnull(date):
            continue

        headline = str(art.get("headline", ""))
        body     = str(art.get("body", ""))
        text     = _text_for_model(headline, body)
        if not text or not sym:
            continue

        record = {
            "text":         text,
            "headline":     headline,
            "symbol":       sym,
            "date":         str(date.date()),
            "source":       str(art.get("source", "")),
            "url":          str(art.get("url", "")),
            "return_5d":    None,
            "return_20d":   None,
            "label":        None,
            "label_source": None,
            "confidence":   None,
        }

        # 1. Price outcome (strongest signal)
        pr = _price_label(sym, date, prices)
        if pr:
            lbl, l5, l20, r5, r20, conf = pr
            record.update({
                "label":        lbl,
                "label_source": "price_outcome",
                "confidence":   conf,
                "return_5d":    round(r5,  5) if r5  is not None else None,
                "return_20d":   round(r20, 5) if r20 is not None else None,
            })

        # 2. BSE event category (upgrades confidence if it agrees)
        if art.get("category"):
            br = _bse_category_label(art["category"])
            if br:
                bse_lbl, _ = br
                if record["label"] is None:
                    record.update({"label": bse_lbl, "label_source": "bse_category",
                                   "confidence": "medium"})
                elif record["label"] == bse_lbl:
                    record["confidence"] = "high"

        # 3. Keyword heuristic (fills gaps, low confidence)
        if record["label"] is None:
            kw_lbl, _ = _keyword_label(text)
            if kw_lbl != "neutral":
                record.update({"label": kw_lbl, "label_source": "keyword",
                               "confidence": "low"})

        # 4. Pseudo neutral catch-all
        if record["label"] is None:
            record.update({"label": "neutral", "label_source": "pseudo",
                           "confidence": "low"})

        labeled.append(record)

    return labeled


# ── Daily runner ───────────────────────────────────────────────────────────────

def run_daily(backfill: bool = False) -> list[dict]:
    """
    Incremental daily labeling.
    backfill=True  — process all historical news files
    backfill=False — process today's file (or yesterday if today is missing)
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    labeled_hashes = _load_index()

    from alphasense.data.bse_client  import load_announcements
    news_dir = cfg.news_dir

    if backfill:
        news_files = sorted(news_dir.glob("news_*.json"))
        logger.info(f"Backfill mode: {len(news_files)} news files")
    else:
        today     = datetime.now().strftime("%Y%m%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        news_files = []
        for d in [today, yesterday]:
            f = news_dir / f"news_{d}.json"
            if f.exists():
                news_files.append(f)
                break

    all_articles: list[dict] = []

    # BSE announcements
    try:
        anns = load_announcements(relevant_only=True)
        if not anns.empty:
            for _, row in anns.iterrows():
                headline = str(row.get("subject", ""))
                sym      = str(row.get("scrip_code", ""))
                date_val = row.get("date")
                ch = _content_hash(headline + sym + str(date_val))
                if ch not in labeled_hashes:
                    all_articles.append({
                        "symbol":   sym,
                        "date":     date_val,
                        "headline": headline,
                        "body":     str(row.get("body", "")),
                        "source":   "BSE",
                        "category": str(row.get("category", "")),
                        "url":      str(row.get("url", "")),
                        "_hash":    ch,
                    })
    except Exception as e:
        logger.warning(f"BSE announcements load failed: {e}")

    # News files
    for nf in news_files:
        try:
            items = json.loads(nf.read_text(encoding="utf-8"))
            for item in items:
                syms = item.get("matched_symbols", [])
                if not isinstance(syms, list) or not syms:
                    continue
                headline = str(item.get("headline", ""))
                for sym in syms:
                    ch = _content_hash(headline + sym)
                    if ch not in labeled_hashes:
                        all_articles.append({
                            "symbol":   sym,
                            "date":     item.get("published_at"),
                            "headline": headline,
                            "body":     str(item.get("body", "")),
                            "source":   str(item.get("source", "")),
                            "category": "",
                            "url":      str(item.get("url", "")),
                            "_hash":    ch,
                        })
        except Exception as e:
            logger.warning(f"Could not load {nf}: {e}")

    if not all_articles:
        logger.info("No new articles to label.")
        return []

    logger.info(f"Articles to label: {len(all_articles)} "
                f"(already labeled: {len(labeled_hashes)})")

    # Load prices for all unique symbols
    symbols = list({a["symbol"] for a in all_articles})
    prices  = load_prices(symbols)
    logger.info(f"Prices loaded for {len(prices)} symbols")

    # Label
    labeled = label_batch(all_articles, prices)

    # Write incremental JSONL
    today_str = datetime.now().strftime("%Y%m%d")
    out_file  = LABELS_DIR / f"labeled_{today_str}.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        for rec in labeled:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # Update index
    new_hashes = {a["_hash"] for a in all_articles if "_hash" in a}
    labeled_hashes.update(new_hashes)
    _save_index(labeled_hashes)

    label_counts  = Counter(r["label"]        for r in labeled)
    source_counts = Counter(r["label_source"] for r in labeled)
    conf_counts   = Counter(r["confidence"]   for r in labeled)

    logger.info(f"Labeled {len(labeled)} articles → {out_file}")
    logger.info(f"  Labels:     {dict(label_counts)}")
    logger.info(f"  Sources:    {dict(source_counts)}")
    logger.info(f"  Confidence: {dict(conf_counts)}")

    return labeled


# ── Dataset loader ─────────────────────────────────────────────────────────────

def build_full_dataset() -> pd.DataFrame:
    """Load all JSONL label files into a single DataFrame."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for jf in sorted(LABELS_DIR.glob("labeled_*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    logger.info(f"Full dataset: {len(df)} labeled samples")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AlphaSense FinBERT data labeler")
    p.add_argument("--backfill", action="store_true",
                   help="Process all historical news files")
    p.add_argument("--report",   action="store_true",
                   help="Print dataset stats only (no labeling)")
    args = p.parse_args()

    if args.report:
        df = build_full_dataset()
        if df.empty:
            print("No labeled data yet. Run with --backfill first.")
        else:
            print(f"\nTotal samples: {len(df)}")
            print("\nLabel distribution:")
            print(df["label"].value_counts().to_string())
            print("\nLabel source:")
            print(df["label_source"].value_counts().to_string())
            print("\nConfidence:")
            print(df["confidence"].value_counts().to_string())
    else:
        run_daily(backfill=args.backfill)
