"""
Signal Engine
=============
Generates BUY/SELL signals by combining:
  • Fundamental quality universe (pre-filter)
  • Text sentiment score
  • 5-day return Z-score
  • Structural blocklist check
  • India VIX market regime filter

BUY when ALL conditions true simultaneously:
  1. Stock is in quality universe
  2. Sentiment score < -0.4  (significant negative news)
  3. 5-day Z-score   < -2.0  (2+ sigma price drop)
  4. No blocklist hit         (no fraud / regulatory probe / CEO departure)
  5. India VIX       < 28    (not a systemic crisis)

SELL when ANY exit condition:
  • Sentiment recovers  > +0.1
  • Z-score recovers    > -1.0
  • Stop-loss           -8% from entry
  • Time-stop           20 trading days

Position sizing: min of (1% ADV, 10% capital, 2%-risk-based shares).
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

# ─── Blocklist ────────────────────────────────────────────────────────────────

BLOCKLIST_KW = [
    "fraud", "scam", "default", "insolvency", "nclt",
    "sebi ban", "regulatory probe", "cbi inquiry",
    "ed raid", "enforcement directorate", "forensic audit", "whistleblower",
    "resignation of ceo", "resignation of md", "resignation of cfo",
    "promoter pledge", "promoter selling", "debt restructuring",
]


def check_blocklist(headlines: list[str]) -> tuple[bool, str]:
    for h in headlines:
        hl = h.lower()
        for kw in BLOCKLIST_KW:
            if kw in hl:
                return True, kw
    return False, ""


# ─── Z-Score ─────────────────────────────────────────────────────────────────

def rolling_zscore(prices: pd.Series,
                   window: int = 252,
                   period: int = 5) -> pd.Series:
    """Z-score of N-day returns over a rolling window."""
    ret   = prices.pct_change(period)
    mu    = ret.rolling(window).mean()
    sigma = ret.rolling(window).std().replace(0, np.nan)
    return (ret - mu) / sigma


# ─── Signal dataclass ────────────────────────────────────────────────────────

class ExitReason(str, Enum):
    SENTIMENT_RECOVERY = "sentiment_recovery"
    PRICE_RECOVERY     = "price_recovery"
    STOP_LOSS          = "stop_loss"
    TIME_STOP          = "time_stop"
    VIX_BREACH         = "vix_breach"


@dataclass
class Signal:
    symbol:          str
    date:            pd.Timestamp
    direction:       str            # "BUY" | "SELL"
    sentiment_score: float
    zscore:          float
    quality_score:   float
    india_vix:       float
    entry_price:     Optional[float] = None
    exit_price:      Optional[float] = None
    exit_date:       Optional[pd.Timestamp] = None
    exit_reason:     Optional[str]          = None
    pnl_pct:         Optional[float]        = None


# ─── Engine ───────────────────────────────────────────────────────────────────

class SignalEngine:
    def __init__(self):
        self.sc = cfg.signal
        self.open_positions: dict[str, Signal] = {}

    def generate(
        self,
        date:         pd.Timestamp,
        universe:     pd.DataFrame,              # symbol, quality_score
        prices:       dict[str, pd.DataFrame],   # symbol → OHLCV (indexed by date)
        sentiment:    dict[str, float],          # symbol → score
        india_vix:    float,
        recent_news:  dict[str, list[str]] = None,  # symbol → headlines
    ) -> list[Signal]:
        sc       = self.sc
        signals  = []
        recent_news = recent_news or {}

        # ── Check exits ──────────────────────────────────────────────────────
        for sym in list(self.open_positions.keys()):
            exit_sig = self._check_exit(self.open_positions[sym], date, prices, sentiment, india_vix)
            if exit_sig:
                signals.append(exit_sig)
                del self.open_positions[sym]

        # ── Market regime ────────────────────────────────────────────────────
        if india_vix > sc.vix_max:
            logger.debug(f"VIX {india_vix:.1f} > {sc.vix_max} — no new entries")
            return signals

        if len(self.open_positions) >= sc.max_positions:
            return signals

        # ── New BUY signals ──────────────────────────────────────────────────
        for _, row in universe.iterrows():
            sym = row["symbol"]
            if sym in self.open_positions:
                continue

            # Only apply sentiment filter when we have a score for this stock.
            # Stocks with no news today (score defaults to None) pass through.
            sent = sentiment.get(sym, None)
            if sent is not None and sent >= sc.sentiment_threshold:
                continue
            has_news = sent is not None
            sent     = sent if has_news else 0.0   # 0.0 stored in signal; N/A in log

            if sym not in prices or prices[sym].empty:
                continue

            price_df = prices[sym]
            if len(price_df) < sc.rolling_window + 10:
                continue

            z_series = rolling_zscore(price_df["close"], sc.rolling_window, sc.return_period)
            z        = z_series.iloc[-1]
            if pd.isna(z) or z >= sc.zscore_threshold:
                continue

            blocked, reason = check_blocklist(recent_news.get(sym, []))
            if blocked:
                logger.debug(f"  {sym} blocked: {reason}")
                continue

            entry_price = float(price_df["close"].iloc[-1])
            sig = Signal(
                symbol=sym, date=date, direction="BUY",
                sentiment_score=sent, zscore=z,
                quality_score=float(row.get("quality_score", 50)),
                india_vix=india_vix, entry_price=entry_price,
            )
            signals.append(sig)
            self.open_positions[sym] = sig
            sent_str = f"{sent:.2f}" if has_news else "N/A (no news)"
            logger.info(f"  BUY {sym} @ ₹{entry_price:.2f} | sent={sent_str} z={z:.2f}")

            if len(self.open_positions) >= sc.max_positions:
                break

        return signals

    def _check_exit(self, pos: Signal, date: pd.Timestamp,
                    prices: dict[str, pd.DataFrame],
                    sentiment: dict[str, float],
                    india_vix: float) -> Optional[Signal]:
        sc  = self.sc
        sym = pos.symbol
        if sym not in prices or prices[sym].empty:
            return None

        curr_price = float(prices[sym]["close"].iloc[-1])
        pnl        = (curr_price - pos.entry_price) / pos.entry_price
        days_held  = (date - pos.date).days
        curr_sent  = sentiment.get(sym, 0.0)

        z_series = rolling_zscore(prices[sym]["close"])
        curr_z   = float(z_series.iloc[-1]) if not z_series.empty else 0.0

        reason = None
        if curr_sent > sc.recovery_sentiment:
            reason = ExitReason.SENTIMENT_RECOVERY
        elif not pd.isna(curr_z) and curr_z > sc.recovery_zscore:
            reason = ExitReason.PRICE_RECOVERY
        elif pnl <= sc.stop_loss_pct:
            reason = ExitReason.STOP_LOSS
        elif days_held >= sc.time_stop_days:
            reason = ExitReason.TIME_STOP

        if reason:
            logger.info(f"  SELL {sym} @ ₹{curr_price:.2f} | PnL={pnl*100:.1f}% | {reason.value}")
            return Signal(
                symbol=sym, date=date, direction="SELL",
                sentiment_score=curr_sent, zscore=curr_z if not pd.isna(curr_z) else 0.0,
                quality_score=pos.quality_score, india_vix=india_vix,
                entry_price=pos.entry_price, exit_price=curr_price,
                exit_date=date, exit_reason=reason.value, pnl_pct=pnl,
            )
        return None

    def reset(self):
        self.open_positions.clear()


# ─── Position Sizer ───────────────────────────────────────────────────────────

def position_size(capital: float, entry_price: float,
                  avg_daily_volume: float) -> int:
    """
    Returns number of shares: min of three constraints:
      - 1% of avg daily volume (liquidity)
      - 10% of capital (concentration)
      - 2% capital at risk based on 8% stop-loss
    """
    sc = cfg.signal
    liq  = int(avg_daily_volume * sc.max_pct_adv)
    cap  = int((capital * sc.max_pct_capital) / entry_price)
    risk = int((capital * sc.risk_per_trade_pct) / (entry_price * abs(sc.stop_loss_pct))) \
           if entry_price and sc.stop_loss_pct else 0
    return max(0, min(liq, cap, risk))