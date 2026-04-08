"""
Financial News RSS Client
==========================
Polls ET Markets, MoneyControl, LiveMint RSS feeds.
Maps articles to NSE symbols. Saves daily JSON files.

Run:
    python scripts/fetch_news.py
"""

import sys, re, json, time
from pathlib import Path
from datetime import datetime

import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

RSS_FEEDS = {
    "et_markets":   ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "ET"),
    "et_companies": ("https://economictimes.indiatimes.com/news/company/rssfeeds/2143429.cms", "ET"),
    "mc_latest":    ("https://www.moneycontrol.com/rss/latestnews.xml", "MC"),
    "mc_markets":   ("https://www.moneycontrol.com/rss/marketreports.xml", "MC"),
    "livemint":     ("https://www.livemint.com/rss/markets", "LM"),
}


class StockMatcher:
    """Maps free text → NSE symbols using NIFTY 500 constituent names."""

    def __init__(self):
        self.lookup: dict[str, str] = {}
        self.symbols: set[str]      = set()
        path = cfg.nse_dir / "nifty500_constituents.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                sym = row["symbol"]
                self.symbols.add(sym)
                self.lookup[sym.lower()] = sym
                name = str(row.get("company_name", "")).lower().strip()
                if name:
                    self.lookup[name] = sym
                    first = name.split()[0]
                    if len(first) > 3:
                        self.lookup.setdefault(first, sym)
            logger.info(f"StockMatcher: {len(self.symbols)} symbols")
        else:
            logger.warning("No constituents parquet. Run fetch_prices --constituents first.")

    def match(self, text: str) -> list[str]:
        if not self.symbols:
            return []
        matched: set[str] = set()
        text_lower = text.lower()
        for sym in self.symbols:
            if re.search(rf"\b{re.escape(sym)}\b", text, re.IGNORECASE):
                matched.add(sym)
        for name, sym in self.lookup.items():
            if len(name) > 4 and name in text_lower:
                matched.add(sym)
        return list(matched)


def _parse_feed(name: str, url: str, source: str) -> list[dict]:
    try:
        feed  = feedparser.parse(url)
        items = []
        for e in feed.entries:
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime(*e.published_parsed[:6])
            elif hasattr(e, "updated_parsed") and e.updated_parsed:
                pub = datetime(*e.updated_parsed[:6])
            else:
                pub = datetime.now()
            body = BeautifulSoup(
                e.get("summary", e.get("description", "")), "html.parser"
            ).get_text(strip=True)
            items.append({
                "headline":     e.get("title", ""),
                "body":         body[:1000],
                "url":          e.get("link", ""),
                "source":       source,
                "published_at": pub.isoformat(),
            })
        logger.debug(f"  {name}: {len(items)} articles")
        return items
    except Exception as ex:
        logger.error(f"Feed {name}: {ex}")
        return []


def fetch_all_feeds() -> list[dict]:
    all_items = []
    for name, (url, source) in RSS_FEEDS.items():
        all_items.extend(_parse_feed(name, url, source))
        time.sleep(0.5)
    seen, unique = set(), []
    for item in all_items:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)
    logger.info(f"News: {len(unique)} unique articles")
    return unique


def fetch_and_save() -> list[dict]:
    """Fetch today's news, match stocks, persist to disk."""
    matcher  = StockMatcher()
    articles = fetch_all_feeds()
    for a in articles:
        a["matched_symbols"] = matcher.match(f"{a['headline']} {a['body']}")
    matched = sum(1 for a in articles if a["matched_symbols"])
    logger.info(f"  {matched}/{len(articles)} matched to stocks")

    cfg.news_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.news_dir / f"news_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved: {out}")
    return articles


def load_news(date_str: str = None) -> pd.DataFrame:
    """Load news. date_str='20260407' for a single day, None for all."""
    records = []
    if date_str:
        f = cfg.news_dir / f"news_{date_str}.json"
        if f.exists():
            records = json.load(open(f))
    else:
        for f in sorted(cfg.news_dir.glob("news_*.json")):
            records.extend(json.load(open(f)))
    df = pd.DataFrame(records)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
        df = df.sort_values("published_at").reset_index(drop=True)
    return df