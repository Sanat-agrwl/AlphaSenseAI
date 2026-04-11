"""
Financial News RSS Client
==========================
Polls ET Markets, MoneyControl, LiveMint RSS feeds.
Maps articles to NSE symbols using symbol + company-name matching.
"""
import sys, re, json, time
from pathlib import Path
from datetime import datetime

import feedparser
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

RSS_FEEDS = {
    "et_markets":    ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",  "ET"),
    "et_companies":  ("https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",   "ET"),
    "mc_latest":     ("https://www.moneycontrol.com/rss/latestnews.xml",                       "MC"),
    "mc_markets":    ("https://www.moneycontrol.com/rss/marketreports.xml",                    "MC"),
    "livemint":      ("https://www.livemint.com/rss/markets",                                  "LM"),
}

# ── Well-known company name aliases (NSE symbol → search terms) ───────────────
# Covers common short names used in headlines that differ from NSE symbol
COMPANY_ALIASES: dict[str, list[str]] = {
    "RELIANCE":     ["reliance industries", "reliance jio", "mukesh ambani"],
    "TCS":          ["tata consultancy", "tcs"],
    "HDFCBANK":     ["hdfc bank", "hdfcbank"],
    "INFY":         ["infosys"],
    "ICICIBANK":    ["icici bank"],
    "WIPRO":        ["wipro"],
    "HINDUNILVR":   ["hindustan unilever", "hul"],
    "BAJFINANCE":   ["bajaj finance"],
    "BAJAJFINSV":   ["bajaj finserv"],
    "ADANIENT":     ["adani enterprises"],
    "ADANIPORTS":   ["adani ports"],
    "TATAMOTORS":   ["tata motors"],
    "TATASTEEL":    ["tata steel"],
    "AXISBANK":     ["axis bank"],
    "KOTAKBANK":    ["kotak mahindra", "kotak bank"],
    "SBILIFE":      ["sbi life"],
    "HDFCLIFE":     ["hdfc life"],
    "SUNPHARMA":    ["sun pharma", "sun pharmaceutical"],
    "DRREDDY":      ["dr reddy", "dr. reddy"],
    "CIPLA":        ["cipla"],
    "ONGC":         ["ongc", "oil and natural gas"],
    "NTPC":         ["ntpc"],
    "POWERGRID":    ["power grid"],
    "COALINDIA":    ["coal india"],
    "MARUTI":       ["maruti suzuki", "maruti"],
    "HEROMOTOCO":   ["hero motocorp", "hero moto"],
    "M&M":          ["mahindra & mahindra", "m&m"],
    "ULTRACEMCO":   ["ultratech cement"],
    "GRASIM":       ["grasim"],
    "ASIANPAINT":   ["asian paints"],
    "NESTLEIND":    ["nestle india", "nestle"],
    "TITAN":        ["titan company", "titan"],
    "DIVISLAB":     ["divi's laboratories", "divis lab"],
    "TECHM":        ["tech mahindra"],
    "HCLTECH":      ["hcl technologies", "hcl tech"],
    "LT":           ["larsen & toubro", "l&t"],
    "INDUSINDBK":   ["indusind bank"],
    "SBIN":         ["state bank of india", "sbi"],
    "BPCL":         ["bharat petroleum", "bpcl"],
    "IOC":          ["indian oil", "ioc"],
    "BHARTIARTL":   ["bharti airtel", "airtel"],
    "JSWSTEEL":     ["jsw steel"],
    "HINDALCO":     ["hindalco"],
    "VEDL":         ["vedanta"],
    "TATACONSUM":   ["tata consumer"],
}


class StockMatcher:
    """Maps free text → NSE symbols."""

    def __init__(self):
        self.symbols: set[str] = set()
        # symbol → list of lowercase search terms
        self.terms: dict[str, list[str]] = {}

        # Load from constituents CSV (written by nse_client.fetch_constituents)
        csv_path = cfg.data_dir / "nse" / "constituents.csv"
        if csv_path.exists():
            syms = pd.read_csv(csv_path)["symbol"].tolist()
            for sym in syms:
                self.symbols.add(sym)
                self.terms[sym] = [sym.lower()]
        else:
            # Fall back to parquet-derived list
            for f in (cfg.data_dir / "nse").glob("*.parquet"):
                if f.stem != "INDIAVIX":
                    self.symbols.add(f.stem)
                    self.terms[f.stem] = [f.stem.lower()]

        # Add well-known aliases
        for sym, aliases in COMPANY_ALIASES.items():
            if sym not in self.terms:
                self.terms[sym] = [sym.lower()]
                self.symbols.add(sym)
            self.terms[sym].extend(aliases)

        logger.info(f"StockMatcher: {len(self.symbols)} symbols loaded")

    def match(self, text: str) -> list[str]:
        if not self.symbols:
            return []
        text_l = text.lower()
        matched: set[str] = set()
        for sym, terms in self.terms.items():
            for term in terms:
                # Use word-boundary match for short terms, substring for long
                if len(term) <= 6:
                    if re.search(rf"\b{re.escape(term)}\b", text_l):
                        matched.add(sym)
                        break
                else:
                    if term in text_l:
                        matched.add(sym)
                        break
        return sorted(matched)


# ── Feed parser ───────────────────────────────────────────────────────────────

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

            # Strip HTML from summary without BeautifulSoup dependency
            raw_body = e.get("summary", e.get("description", ""))
            body = re.sub(r"<[^>]+>", " ", raw_body).strip()[:1000]

            items.append({
                "headline":     e.get("title", "").strip(),
                "body":         body,
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
        time.sleep(0.3)
    # Deduplicate by URL
    seen, unique = set(), []
    for item in all_items:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)
    logger.info(f"Fetched {len(unique)} unique articles from {len(RSS_FEEDS)} feeds")
    return unique


def fetch_and_save() -> list[dict]:
    """Fetch today's news, match to stocks, save to disk."""
    matcher  = StockMatcher()
    articles = fetch_all_feeds()

    for a in articles:
        a["matched_symbols"] = matcher.match(f"{a['headline']} {a['body']}")

    matched = sum(1 for a in articles if a["matched_symbols"])
    logger.info(f"Matched {matched}/{len(articles)} articles to stocks")

    cfg.news_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.news_dir / f"news_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved → {out}")
    return articles


def load_news(days: int = None) -> pd.DataFrame:
    """Load saved news. days=7 → last 7 files, None → all."""
    records = []
    files = sorted(cfg.news_dir.glob("news_*.json"), reverse=True)
    if days:
        files = files[:days]
    for f in files:
        try:
            records.extend(json.load(open(f, encoding="utf-8")))
        except Exception:
            pass
    df = pd.DataFrame(records)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
        df = df.sort_values("published_at").reset_index(drop=True)
    return df
