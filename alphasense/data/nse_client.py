"""
NSE data client — uses yfinance for reliable OHLCV + VIX data.
Replaces the old cookie-based NSE scraper which broke when NSE changed its API.
"""
import time
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg


def _yf():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError("yfinance not installed — run: pip install yfinance")


# ── Constituents ──────────────────────────────────────────────────────────────

def fetch_constituents() -> list[str]:
    """Return NIFTY 500 symbols (no .NS suffix). Saves to constituents.csv."""
    import requests
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, */*",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        s.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.5)
        r = s.get(
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
            timeout=15,
        )
        if r.status_code == 200:
            data    = r.json().get("data", [])
            symbols = [d["symbol"] for d in data
                       if d.get("symbol") and d["symbol"] != "NIFTY 500"]
            logger.info(f"Fetched {len(symbols)} NIFTY 500 constituents")
            out = cfg.data_dir / "nse" / "constituents.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"symbol": symbols}).to_csv(out, index=False)
            return symbols
    except Exception as e:
        logger.warning(f"NSE constituents fetch failed: {e}")

    return load_constituents()


def load_constituents() -> list[str]:
    cached = cfg.data_dir / "nse" / "constituents.csv"
    if cached.exists():
        return pd.read_csv(cached)["symbol"].tolist()
    # fallback: derive from existing parquet files — skip known non-stock files
    _SKIP = {"INDIAVIX", "india_vix", "nifty500_constituents"}
    return [f.stem for f in (cfg.data_dir / "nse").glob("*.parquet")
            if f.stem not in _SKIP and f.stem.replace("&","").replace("-","").replace("_","").isalnum()]


# ── Price history ─────────────────────────────────────────────────────────────

def fetch_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    yf = _yf()
    try:
        df = yf.Ticker(f"{symbol}.NS").history(period=period, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return df[["date","open","high","low","close","volume"]].sort_values("date")
    except Exception as e:
        logger.warning(f"{symbol}: {e}")
        return pd.DataFrame()


def fetch_vix(period: str = "2y") -> pd.DataFrame:
    yf = _yf()
    try:
        df = yf.Ticker("^INDIAVIX").history(period=period, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return df.rename(columns={"close": "vix_close"})[["date","vix_close"]].sort_values("date")
    except Exception as e:
        logger.warning(f"VIX: {e}")
        return pd.DataFrame()


# ── Backfill ──────────────────────────────────────────────────────────────────

def backfill(symbols: list[str] = None, period: str = "max",
             delay: float = 0.3) -> None:
    """Download full history for all NIFTY 500 stocks."""
    if symbols is None:
        symbols = load_constituents()

    out_dir = cfg.data_dir / "nse"
    out_dir.mkdir(parents=True, exist_ok=True)
    yf = _yf()

    total = len(symbols)
    ok = fail = 0
    batch_size = 50

    for i in range(0, total, batch_size):
        batch  = symbols[i:i + batch_size]
        tickers = [f"{s}.NS" for s in batch]
        logger.info(f"Batch {i//batch_size+1}/{-(-total//batch_size)}: {len(batch)} stocks")

        try:
            data = yf.download(tickers, period=period, auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)
        except Exception as e:
            logger.error(f"Batch error: {e}")
            fail += len(batch); continue

        for sym in batch:
            try:
                df = data[f"{sym}.NS"].copy() if len(batch) > 1 else data.copy()
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                df = df[["date","open","high","low","close","volume"]].dropna().sort_values("date")
                if len(df) < 10:
                    fail += 1; continue
                df.to_parquet(out_dir / f"{sym}.parquet", index=False)
                ok += 1
            except Exception as e:
                logger.warning(f"  {sym}: {e}"); fail += 1

        time.sleep(delay)

    # VIX
    vdf = fetch_vix(period="10y")
    if not vdf.empty:
        vdf.to_parquet(out_dir / "INDIAVIX.parquet", index=False)
        logger.info(f"VIX: {len(vdf)} rows saved")

    logger.info(f"Backfill done: {ok} OK / {fail} failed of {total}")


def update_prices(symbols: list[str] = None, days: int = 7) -> int:
    """Incremental daily update — append last N days to existing parquets."""
    if symbols is None:
        symbols = load_constituents()

    # Strip any non-stock symbols that may have crept in
    _SKIP = {"INDIAVIX", "india_vix", "nifty500_constituents"}
    symbols = [s for s in symbols if s not in _SKIP]

    out_dir    = cfg.data_dir / "nse"
    yf         = _yf()
    period     = f"{max(days, 7)}d"
    updated    = 0
    batch_size = 100

    for i in range(0, len(symbols), batch_size):
        batch   = symbols[i:i + batch_size]
        tickers = [f"{s}.NS" for s in batch]
        try:
            data = yf.download(tickers, period=period, auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)
        except Exception as e:
            logger.warning(f"Update batch error: {e}")
            continue

        for sym in batch:
            path = out_dir / f"{sym}.parquet"
            try:
                # group_by="ticker" puts ticker at level 0 of MultiIndex columns
                if len(batch) > 1:
                    ticker_key = f"{sym}.NS"
                    if ticker_key not in data.columns.get_level_values(0):
                        continue
                    new = data[ticker_key].copy()
                else:
                    new = data.copy()

                new = new.reset_index()
                new.columns = [c.lower() for c in new.columns]
                new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None).dt.normalize()
                new = new[["date","open","high","low","close","volume"]].dropna()
                if new.empty:
                    continue
                if path.exists():
                    old = pd.read_parquet(path)
                    new = pd.concat([old, new]).drop_duplicates("date").sort_values("date")
                new.to_parquet(path, index=False)
                updated += 1
            except Exception:
                pass
        time.sleep(0.2)

    # VIX update
    vdf = fetch_vix(period=period)
    vpath = out_dir / "INDIAVIX.parquet"
    if not vdf.empty:
        if vpath.exists():
            vdf = pd.concat([pd.read_parquet(vpath), vdf]).drop_duplicates("date").sort_values("date")
        vdf.to_parquet(vpath, index=False)
        logger.info(f"VIX updated: last date={vdf['date'].iloc[-1].date()}, value={vdf['vix_close'].iloc[-1]:.1f}")
    else:
        logger.warning("VIX update returned empty — ^INDIAVIX may be unavailable")

    logger.info(f"Daily price update complete: {updated}/{len(symbols)} stocks refreshed")
    return updated


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_prices(symbols: list[str] = None) -> dict[str, pd.DataFrame]:
    nse_dir = cfg.data_dir / "nse"
    if not nse_dir.exists():
        return {}
    files = [f for f in nse_dir.glob("*.parquet") if f.stem != "INDIAVIX"]
    if symbols:
        sym_set = set(symbols)
        files = [f for f in files if f.stem in sym_set]
    result = {}
    for f in files:
        try:
            df = pd.read_parquet(f).sort_values("date")
            df["date"] = pd.to_datetime(df["date"])
            result[f.stem] = df.set_index("date")
        except Exception:
            pass
    return result


def load_vix() -> pd.DataFrame:
    p = cfg.data_dir / "nse" / "INDIAVIX.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def current_vix() -> float:
    vdf = load_vix()
    if vdf.empty:
        return 18.0
    return float(vdf["vix_close"].iloc[-1])
