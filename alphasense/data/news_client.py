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


def fetch_stock_news(symbols: list[str], max_per_stock: int = 3) -> list[dict]:
    """
    Fetch Google News RSS for specific stocks — fills the gap for mid/small caps
    that never appear in general ET/MC feeds.
    Free, no API key needed. Called for signal candidates only.
    """
    import urllib.parse
    articles = []
    seen_urls: set[str] = set()

    for sym in symbols:
        # Use company alias if available, else just symbol
        query_terms = COMPANY_ALIASES.get(sym, [sym.lower()])
        query = query_terms[0]   # most descriptive alias
        url   = f"https://news.google.com/rss/search?q={urllib.parse.quote(query + ' NSE India stock')}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            feed  = feedparser.parse(url)
            count = 0
            for e in feed.entries[:10]:
                link = e.get("link", "")
                if link in seen_urls:
                    continue
                if hasattr(e, "published_parsed") and e.published_parsed:
                    pub = datetime(*e.published_parsed[:6])
                else:
                    pub = datetime.now()
                articles.append({
                    "headline":       e.get("title", "").strip(),
                    "body":           "",
                    "url":            link,
                    "source":         "GoogleNews",
                    "published_at":   pub.isoformat(),
                    "matched_symbols": [sym],
                })
                seen_urls.add(link)
                count += 1
                if count >= max_per_stock:
                    break
            logger.debug(f"  GoogleNews {sym}: {count} articles")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"  GoogleNews {sym}: {e}")

    logger.info(f"Stock-specific news: {len(articles)} articles for {len(symbols)} stocks")
    return articles


def fetch_deep_news(symbols: list[str]) -> list[dict]:
    """
    Firecrawl search for full article text for z-score candidates + held positions.
    Uses ~2 credits per symbol. Only runs when FIRECRAWL_API_KEY is set.
    Returns articles with full body text (up to 2000 chars vs RSS 1000 char truncation).
    """
    import os
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key or not symbols:
        return []
    try:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=api_key)
    except ImportError:
        logger.warning("firecrawl-py not installed — skipping deep news fetch")
        return []

    articles   = []
    seen_urls: set[str] = set()

    for sym in symbols:
        query_terms  = COMPANY_ALIASES.get(sym, [sym.lower()])
        company_name = query_terms[0]
        query        = f'"{company_name}" NSE India stock news'
        try:
            result = app.search(
                query,
                params={
                    "limit": 5,
                    "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
                },
            )
            items = result.get("data", []) if isinstance(result, dict) else (result or [])
            for item in items:
                url = item.get("url", "")
                if not url or url in seen_urls:
                    continue
                body = (item.get("markdown") or item.get("content")
                        or item.get("description", ""))
                articles.append({
                    "headline":        (item.get("title") or "").strip(),
                    "body":            body[:2000] if body else "",
                    "url":             url,
                    "source":          "Firecrawl",
                    "published_at":    datetime.now().isoformat(),
                    "matched_symbols": [sym],
                })
                seen_urls.add(url)
            logger.debug(f"  Firecrawl {sym}: {len(items)} articles")
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"  Firecrawl {sym}: {e}")

    logger.info(f"Firecrawl deep news: {len(articles)} articles for {len(symbols)} stocks")
    return articles


def fetch_regulatory() -> list[dict]:
    """
    Scrape SEBI enforcement orders + RBI press releases for any company mentions.
    Uses ~4 credits/day flat. Only runs when FIRECRAWL_API_KEY is set.
    """
    import os
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return []
    try:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=api_key)
    except ImportError:
        return []

    REGULATORY_SOURCES = [
        ("https://www.sebi.gov.in/enforcement/orders/latest.html", "SEBI"),
        ("https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",  "RBI"),
    ]

    articles = []
    matcher  = StockMatcher()

    for url, source in REGULATORY_SOURCES:
        try:
            result  = app.scrape_url(
                url,
                params={"formats": ["markdown"], "onlyMainContent": True},
            )
            content = result.get("markdown", "") if isinstance(result, dict) else ""
            if not content:
                continue
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
            for para in paragraphs[:20]:
                syms = matcher.match(para)
                if syms:
                    articles.append({
                        "headline":        para[:120],
                        "body":            para[:1000],
                        "url":             url,
                        "source":          source,
                        "published_at":    datetime.now().isoformat(),
                        "matched_symbols": syms,
                    })
            logger.debug(f"  Regulatory {source}: {len(articles)} matched items")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"  Regulatory {source}: {e}")

    logger.info(f"Regulatory news: {len(articles)} matched items")
    return articles


def fetch_and_save(extra_symbols: list[str] = None) -> list[dict]:
    """
    Fetch today's news, match to stocks, save to disk.
    extra_symbols: specific stocks to search for (signal candidates + held positions).
      - Google News RSS (always, free)
      - Firecrawl deep search (if FIRECRAWL_API_KEY set; full article text)
    Also appends SEBI/RBI regulatory items when Firecrawl key is set.
    """
    import os
    matcher  = StockMatcher()
    articles = fetch_all_feeds()

    for a in articles:
        a["matched_symbols"] = matcher.match(f"{a['headline']} {a['body']}")

    existing_urls = {a["url"] for a in articles}

    # Google News RSS for candidates (free)
    if extra_symbols:
        stock_articles = fetch_stock_news(extra_symbols)
        new_ones = [a for a in stock_articles if a["url"] not in existing_urls]
        articles.extend(new_ones)
        existing_urls.update(a["url"] for a in new_ones)
        logger.info(f"Added {len(new_ones)} stock-specific articles from Google News")

    # Firecrawl deep search for candidates (full article body, uses credits)
    if extra_symbols and os.getenv("FIRECRAWL_API_KEY"):
        deep = fetch_deep_news(extra_symbols)
        new_deep = [a for a in deep if a["url"] not in existing_urls]
        articles.extend(new_deep)
        existing_urls.update(a["url"] for a in new_deep)
        logger.info(f"Added {len(new_deep)} Firecrawl deep articles")

    # Regulatory scrape (SEBI/RBI — flat ~4 credits/day)
    if os.getenv("FIRECRAWL_API_KEY"):
        reg = fetch_regulatory()
        new_reg = [a for a in reg if a["url"] not in existing_urls]
        articles.extend(new_reg)
        logger.info(f"Added {len(new_reg)} regulatory articles")

    matched = sum(1 for a in articles if a.get("matched_symbols"))
    logger.info(f"Matched {matched}/{len(articles)} articles to stocks")

    cfg.news_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.news_dir / f"news_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved → {out}")
    return articles


def load_news(days=None) -> pd.DataFrame:
    """Load saved news. days=7 → last 7 files, None → all, days='YYYYMMDD' → specific date."""
    records = []
    if isinstance(days, str):
        # Specific date string like "20260411"
        files = [cfg.news_dir / f"news_{days}.json"]
    else:
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
