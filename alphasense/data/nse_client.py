"""
NSE Data Client
===============
Downloads NIFTY 500 constituent list, OHLCV price history, and India VIX.
Saves one parquet per stock in data/nse/.

Run:
    python scripts/fetch_prices.py --backfill
    python scripts/fetch_prices.py --symbol RELIANCE
    python scripts/fetch_prices.py --vix
"""

import sys, time
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

NSE_BASE = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": NSE_BASE,
}


class NSEClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self._refresh_cookies()

    def _refresh_cookies(self):
        try:
            self.session.get(NSE_BASE, timeout=10)
            time.sleep(1)
        except Exception as e:
            logger.warning(f"NSE cookie init: {e}")

    def get(self, url: str, params: dict = None, retries: int = 3) -> dict | None:
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=20)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 401:
                    self._refresh_cookies()
                else:
                    logger.warning(f"HTTP {r.status_code}: {url}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
        return None


def fetch_constituents(client: NSEClient) -> pd.DataFrame:
    url  = f"{NSE_BASE}/api/equity-stockIndices?index=NIFTY%20500"
    data = client.get(url)
    if not data or "data" not in data:
        logger.error("Failed to fetch NIFTY 500 list")
        return pd.DataFrame()

    rows = [
        {
            "symbol":        d["symbol"],
            "company_name":  d.get("meta", {}).get("companyName", d["symbol"]),
            "isin":          d.get("meta", {}).get("isin", ""),
            "industry":      d.get("meta", {}).get("industry", ""),
            "last_price":    d.get("lastPrice", 0),
            "market_cap_cr": d.get("ffmc", 0),
        }
        for d in data["data"] if d.get("symbol") != "NIFTY 500"
    ]
    df = pd.DataFrame(rows)
    logger.info(f"Constituents: {len(df)} stocks")
    return df


def fetch_history(client: NSEClient, symbol: str,
                  from_date: str, to_date: str) -> pd.DataFrame:
    """from_date / to_date in DD-MM-YYYY format."""
    url = f"{NSE_BASE}/api/historical/cm/equity"
    data = client.get(url, params={
        "symbol": symbol, "series": '["EQ"]',
        "from": from_date, "to": to_date,
    })
    if not data or "data" not in data:
        return pd.DataFrame()

    rows = [{
        "date":         r.get("CH_TIMESTAMP", ""),
        "open":         float(r.get("CH_OPENING_PRICE", 0)),
        "high":         float(r.get("CH_TRADE_HIGH_PRICE", 0)),
        "low":          float(r.get("CH_TRADE_LOW_PRICE", 0)),
        "close":        float(r.get("CH_CLOSING_PRICE", 0)),
        "volume":       float(r.get("CH_TOT_TRADED_QTY", 0)),
        "delivery_pct": float(r.get("COP_DELIV_PERC", 0)),
    } for r in data["data"]]

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_vix(client: NSEClient, days: int = 3650) -> pd.DataFrame:
    url = f"{NSE_BASE}/api/historical/vixhistory"
    data = client.get(url, params={
        "from": (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y"),
        "to":   datetime.now().strftime("%d-%m-%Y"),
    })
    if not data or "data" not in data:
        return pd.DataFrame()

    rows = [{
        "date":      r.get("EOD_TIMESTAMP", ""),
        "vix_open":  float(r.get("EOD_OPEN_INDEX_VAL", 0)),
        "vix_high":  float(r.get("EOD_HIGH_INDEX_VAL", 0)),
        "vix_low":   float(r.get("EOD_LOW_INDEX_VAL", 0)),
        "vix_close": float(r.get("EOD_CLOSE_INDEX_VAL", 0)),
    } for r in data["data"]]

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def backfill(from_date: str = "01-01-2015", sleep: float = 2.0):
    """Download all NIFTY 500 stocks from from_date to today."""
    client   = NSEClient()
    out_dir  = cfg.nse_dir
    const_path = out_dir / "nifty500_constituents.parquet"

    if const_path.exists():
        constituents = pd.read_parquet(const_path)
    else:
        constituents = fetch_constituents(client)
        if constituents.empty:
            return
        constituents.to_parquet(const_path, index=False)

    start = datetime.strptime(from_date, "%d-%m-%Y")
    end   = datetime.now()
    n     = len(constituents)

    for i, row in constituents.iterrows():
        sym  = row["symbol"]
        path = out_dir / f"{sym}.parquet"
        if path.exists():
            logger.debug(f"[{i+1}/{n}] Skip {sym}")
            continue

        logger.info(f"[{i+1}/{n}] {sym}")
        chunks, cursor = [], start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=365), end)
            df = fetch_history(client, sym,
                               cursor.strftime("%d-%m-%Y"),
                               chunk_end.strftime("%d-%m-%Y"))
            if not df.empty:
                chunks.append(df)
            cursor = chunk_end + timedelta(days=1)
            time.sleep(sleep)

        if chunks:
            out = pd.concat(chunks).drop_duplicates("date") \
                    .sort_values("date").reset_index(drop=True)
            out.to_parquet(path, index=False)
            logger.info(f"  → {len(out)} rows")

    # VIX
    vix = fetch_vix(client)
    if not vix.empty:
        vix.to_parquet(out_dir / "india_vix.parquet", index=False)
        logger.info(f"VIX saved: {len(vix)} rows")

    logger.success("Backfill complete")


def load_prices(symbols: list[str] = None) -> dict[str, pd.DataFrame]:
    """Load price parquets → {symbol: DataFrame indexed by date}."""
    prices = {}
    for f in cfg.nse_dir.glob("*.parquet"):
        if f.stem in ("nifty500_constituents", "india_vix"):
            continue
        if symbols and f.stem not in symbols:
            continue
        df = pd.read_parquet(f)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            prices[f.stem] = df.set_index("date").sort_index()
    return prices


def load_vix() -> pd.DataFrame:
    p = cfg.nse_dir / "india_vix.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()