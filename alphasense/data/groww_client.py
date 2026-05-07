"""
Groww Trade API Client
======================
Wraps the growwapi Python SDK for:
  - Historical OHLCV candles (replaces yfinance for daily prices)
  - Live batch LTP / OHLC (up to 50 symbols per call)
  - Order placement (used when ACTUAL_TRADE=true)

Auth (priority order):
  1. TOTP — GROWW_TOTP_SECRET in .env (preferred for automation, no daily expiry)
  2. API key/secret — GROWW_API_KEY + GROWW_API_SECRET (requires daily token refresh)

Groww symbol format: "NSE-SYMBOL" (we strip the prefix internally).
NSE segment is always CASH for equity.

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
    """RELIANCE → NSE-RELIANCE"""
    return f"NSE-{symbol}"


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
    """

    def __init__(self):
        self._api = None
        self._instruments: Optional[pd.DataFrame] = None
        self._init()

    def _init(self):
        gc = cfg.groww
        if not (gc.api_key or gc.totp_secret):
            logger.debug("Groww: no credentials configured — client inactive")
            return
        try:
            GrowwAPI = _groww_sdk()
            if gc.totp_secret:
                import pyotp
                totp  = pyotp.TOTP(gc.totp_secret)
                token = totp.now()
                self._api = GrowwAPI(token)
                logger.info("Groww client initialised via TOTP")
            elif gc.api_key and gc.api_secret:
                token = GrowwAPI.get_access_token(api_key=gc.api_key, secret=gc.api_secret)
                self._api = GrowwAPI(token)
                logger.info("Groww client initialised via API key")
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
        """
        if not self.available:
            return pd.DataFrame()
        end   = datetime.now()
        start = end - timedelta(days=days)
        try:
            candles = self._api.get_historical_candles(
                exchange=_EXCHANGE,
                segment=_SEGMENT,
                trading_symbol=symbol,
                start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval,
            )
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["ts"], unit="s").dt.normalize()
            df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
            return df.reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Groww historical {symbol}: {e}")
            return pd.DataFrame()

    # ── Live prices ──────────────────────────────────────────────────────────

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """
        Batch LTP for up to 50 NSE symbols.
        Returns {symbol: ltp_price}.
        """
        if not self.available or not symbols:
            return {}
        results = {}
        # API accepts max 50 per call
        for i in range(0, len(symbols), 50):
            batch = symbols[i:i + 50]
            try:
                resp = self._api.get_ltp(
                    exchange_trading_symbols=batch,
                    segment=_SEGMENT,
                )
                # resp is a dict: {symbol: {"ltp": price, ...}}
                for sym, data in resp.items():
                    clean_sym = sym.replace(f"{_EXCHANGE}:", "").replace(f"{_EXCHANGE}_", "")
                    price = data.get("ltp") or data.get("last_traded_price")
                    if price:
                        results[clean_sym] = float(price)
            except Exception as e:
                logger.warning(f"Groww LTP batch: {e}")
            time.sleep(0.1)
        return results

    def get_ohlc(self, symbols: list[str]) -> dict[str, dict]:
        """
        Batch OHLC snapshot for up to 50 symbols.
        Returns {symbol: {"open": x, "high": x, "low": x, "close": x}}.
        """
        if not self.available or not symbols:
            return {}
        results = {}
        for i in range(0, len(symbols), 50):
            batch = symbols[i:i + 50]
            try:
                resp = self._api.get_ohlc(
                    exchange_trading_symbols=batch,
                    segment=_SEGMENT,
                )
                for sym, data in resp.items():
                    clean_sym = sym.replace(f"{_EXCHANGE}:", "").replace(f"{_EXCHANGE}_", "")
                    results[clean_sym] = {
                        "open":  float(data.get("open", 0)),
                        "high":  float(data.get("high", 0)),
                        "low":   float(data.get("low", 0)),
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
        Place a MARKET order on NSE CNC (delivery).
        direction: "BUY" or "SELL"
        Returns order_id or None on failure.
        """
        if not self.available:
            logger.warning("Groww not available — order not placed")
            return None
        try:
            order_id = self._api.place_order(
                trading_symbol=symbol,
                quantity=qty,
                price=0,            # market order
                trigger_price=0,
                validity="DAY",
                exchange=_EXCHANGE,
                segment=_SEGMENT,
                product="CNC",
                order_type="MARKET",
                transaction_type=direction,
            )
            logger.info(f"Groww {direction} {symbol} ×{qty} placed → {order_id}")
            return str(order_id)
        except Exception as e:
            logger.error(f"Groww order failed {direction} {symbol}: {e}")
            return None

    def get_holdings(self) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        try:
            holdings = self._api.get_holdings_for_user()
            return pd.DataFrame(holdings) if holdings else pd.DataFrame()
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
