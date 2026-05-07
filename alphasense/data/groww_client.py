"""
Groww Trade API Client
======================
Wraps the growwapi Python SDK for:
  - Historical OHLCV candles (replaces yfinance for daily prices)
  - Live batch LTP / OHLC (up to 50 symbols per call)
  - Order placement (used when ACTUAL_TRADE=true)

Auth flow (TOTP — permanent, no daily expiry):
  GROWW_API_KEY   = long-lived JWT from Groww API portal
  GROWW_TOTP_SECRET = base32 TOTP seed from Groww 2FA setup
  → get_access_token(api_key=JWT, totp=6-digit-code) → session token each time

Groww symbol format for LTP/OHLC: "NSE_SYMBOL" (underscore separator)
Historical uses: trading_symbol="SYMBOL", exchange="NSE", segment="CASH"

Install: pip install growwapi pyotp
"""

import os, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

_EXCHANGE = "NSE"
_SEGMENT  = "CASH"


def _to_groww_sym(symbol: str) -> str:
    """RELIANCE → NSE_RELIANCE  (Groww LTP/OHLC format)"""
    return f"NSE_{symbol}"


def _from_groww_sym(groww_sym: str) -> str:
    """NSE_RELIANCE → RELIANCE"""
    return groww_sym.replace("NSE_", "").replace("BSE_", "")


def _groww_sdk():
    try:
        from growwapi import GrowwAPI
        return GrowwAPI
    except ImportError:
        raise ImportError("growwapi not installed — run: pip install growwapi pyotp")


class GrowwClient:
    """
    Single authenticated client. Instantiate once per process.
    Falls back gracefully when credentials are missing.

    Auth requires BOTH:
      GROWW_API_KEY    — long-lived JWT from Groww API portal
      GROWW_TOTP_SECRET — base32 seed (shown during 2FA setup)
    """

    def __init__(self):
        self._api = None
        self._instruments: Optional[pd.DataFrame] = None
        self._init()

    def _init(self):
        gc = cfg.groww
        if not gc.api_key:
            logger.debug("Groww: GROWW_API_KEY not set — client inactive")
            return
        try:
            GrowwAPI = _groww_sdk()
            if gc.totp_secret:
                import pyotp
                totp_code     = pyotp.TOTP(gc.totp_secret).now()
                session_token = GrowwAPI.get_access_token(api_key=gc.api_key, totp=totp_code)
                self._api     = GrowwAPI(session_token)
                logger.info("Groww client initialised via TOTP")
            elif gc.api_secret:
                session_token = GrowwAPI.get_access_token(api_key=gc.api_key, secret=gc.api_secret)
                self._api     = GrowwAPI(session_token)
                logger.info("Groww client initialised via API key+secret")
            else:
                logger.debug("Groww: need GROWW_TOTP_SECRET or GROWW_API_SECRET — client inactive")
        except Exception as e:
            logger.warning(f"Groww init failed: {e}")
            self._api = None

    @property
    def available(self) -> bool:
        return self._api is not None

    # ── Instruments ──────────────────────────────────────────────────────────

    def load_instruments(self) -> pd.DataFrame:
        if self._instruments is not None:
            return self._instruments
        if not self.available:
            return pd.DataFrame()
        try:
            self._instruments = self._api.get_all_instruments()
            logger.info(f"Groww instruments: {len(self._instruments)} loaded")
        except Exception as e:
            logger.warning(f"Groww instruments load failed: {e}")
            self._instruments = pd.DataFrame()
        return self._instruments

    # ── Historical candles ───────────────────────────────────────────────────

    def get_historical(self, symbol: str, days: int = 400,
                       interval: str = "1d") -> pd.DataFrame:
        """
        Fetch daily OHLCV for one NSE equity symbol.
        Returns DataFrame with columns: date, open, high, low, close, volume
        interval: "1d" maps to no interval_in_minutes (daily candles)
        """
        if not self.available:
            return pd.DataFrame()
        end   = datetime.now()
        start = end - timedelta(days=days)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # suppress deprecation warning
                result = self._api.get_historical_candle_data(
                    trading_symbol=symbol,
                    exchange=_EXCHANGE,
                    segment=_SEGMENT,
                    start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
                    end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
                )
            if not result or not isinstance(result, dict):
                return pd.DataFrame()
            raw = result.get("candles", [])
            if not raw:
                return pd.DataFrame()
            # Candle format: [timestamp_ms_or_s, open, high, low, close, volume]
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            # Normalise timestamps: if ts > 1e10 it's milliseconds, else seconds
            ts = df["ts"].iloc[0]
            df["date"] = pd.to_datetime(df["ts"], unit="ms" if ts > 1e10 else "s").dt.normalize()
            df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
            # Keep only daily rows (last row per date) when interval is 1d
            if interval == "1d":
                df = df.groupby("date").last().reset_index()
            return df.reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Groww historical {symbol}: {e}")
            return pd.DataFrame()

    # ── Live prices ──────────────────────────────────────────────────────────

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """
        Batch LTP for up to 50 NSE symbols.
        Input:  ["RELIANCE", "TCS", ...]
        Output: {"RELIANCE": 1436.2, "TCS": 2401.4, ...}
        """
        if not self.available or not symbols:
            return {}
        results = {}
        for i in range(0, len(symbols), 50):
            batch = tuple(_to_groww_sym(s) for s in symbols[i:i + 50])
            try:
                resp = self._api.get_ltp(
                    exchange_trading_symbols=batch,
                    segment=_SEGMENT,
                )
                for groww_sym, data in resp.items():
                    sym   = _from_groww_sym(groww_sym)
                    price = data.get("ltp") or data.get("last_traded_price") \
                            if isinstance(data, dict) else data
                    if price:
                        results[sym] = float(price)
            except Exception as e:
                logger.warning(f"Groww LTP batch: {e}")
            time.sleep(0.1)
        return results

    def get_ohlc(self, symbols: list[str]) -> dict[str, dict]:
        """
        Batch OHLC snapshot for up to 50 symbols.
        Input:  ["RELIANCE", "HDFCBANK"]
        Output: {"RELIANCE": {"open": x, "high": x, "low": x, "close": x}, ...}
        """
        if not self.available or not symbols:
            return {}
        results = {}
        for i in range(0, len(symbols), 50):
            batch = tuple(_to_groww_sym(s) for s in symbols[i:i + 50])
            try:
                resp = self._api.get_ohlc(
                    exchange_trading_symbols=batch,
                    segment=_SEGMENT,
                )
                for groww_sym, data in resp.items():
                    sym = _from_groww_sym(groww_sym)
                    if isinstance(data, dict):
                        results[sym] = {
                            "open":  float(data.get("open",  0)),
                            "high":  float(data.get("high",  0)),
                            "low":   float(data.get("low",   0)),
                            "close": float(data.get("close", data.get("ltp", 0))),
                        }
            except Exception as e:
                logger.warning(f"Groww OHLC batch: {e}")
            time.sleep(0.1)
        return results

    # ── Order placement ──────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, qty: int,
                           direction: str) -> Optional[str]:
        """
        Place a MARKET CNC order on NSE.
        direction: "BUY" or "SELL"
        Returns order_id or None on failure.
        """
        if not self.available:
            logger.warning("Groww not available — order not placed")
            return None
        try:
            result = self._api.place_order(
                trading_symbol=symbol,
                quantity=qty,
                price=0.0,
                validity="DAY",
                exchange=_EXCHANGE,
                segment=_SEGMENT,
                product="CNC",
                order_type="MARKET",
                transaction_type=direction,
            )
            order_id = result.get("data", {}).get("order_id") or str(result)
            logger.info(f"Groww {direction} {symbol} ×{qty} placed → {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Groww order failed {direction} {symbol}: {e}")
            return None

    def get_holdings(self) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        try:
            result = self._api.get_holdings_for_user()
            data   = result.get("data") or result.get("holdings") or \
                     (result if isinstance(result, list) else [])
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Groww holdings: {e}")
            return pd.DataFrame()


# ── Module-level singleton ────────────────────────────────────────────────────

_client: Optional[GrowwClient] = None


def get_groww_client() -> GrowwClient:
    global _client
    if _client is None:
        _client = GrowwClient()
    return _client
