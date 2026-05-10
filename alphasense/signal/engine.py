"""
Signal Engine
=============
Multi-strategy signal generator. Four strategies share one position pool:

  MEAN_REVERSION    — contrarian buy on 2σ price shock + negative sentiment
  MOMENTUM          — trend-follow when price + sentiment both rising
  EARNINGS_SURPRISE — post-beat momentum (XBRL beat + audio confidence spike)
  BUYBACK_ARBIT     — buy on BSE buyback announcement before price catches up

Entries gated by: quality universe, India VIX < 28, max_positions limit.
Each strategy has its own entry thresholds, stop-loss, and time-stop.
Position sizing uses Kelly Criterion (half-Kelly, capped at max_pct_capital).
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
    SENTIMENT_RECOVERY  = "sentiment_recovery"
    PRICE_RECOVERY      = "price_recovery"
    STOP_LOSS           = "stop_loss"
    TIME_STOP           = "time_stop"
    VIX_BREACH          = "vix_breach"
    PROFIT_WITH_PENDING = "profit_with_pending"
    MOMENTUM_REVERSAL   = "momentum_reversal"


class StrategyType(str, Enum):
    MEAN_REVERSION    = "mean_reversion"
    MOMENTUM          = "momentum"
    EARNINGS_SURPRISE = "earnings_surprise"
    BUYBACK_ARBIT     = "buyback_arbit"
    VIX_SPIKE_BUY     = "vix_spike_buy"     # OOS Sharpe 2.74, WR 72.2%
    QUALITY_MOMENTUM  = "quality_momentum"  # OOS Sharpe 2.90, WR 59.6%
    GAP_FADE          = "gap_fade"          # OOS Sharpe 2.19, WR 60.1%


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
    strategy:        str                    = StrategyType.MEAN_REVERSION


# ─── Engine ───────────────────────────────────────────────────────────────────

class SignalEngine:
    def __init__(self):
        self.sc = cfg.signal
        self.open_positions: dict[str, Signal] = {}
        self._vix_prev: float = 0.0   # previous day's VIX for spike detection

    def generate(
        self,
        date:          pd.Timestamp,
        universe:      pd.DataFrame,
        prices:        dict[str, pd.DataFrame],
        sentiment:     dict[str, float],
        india_vix:     float,
        recent_news:   dict[str, list[str]] = None,
        broker_positions: dict = None,     # Position objects from PaperEngine
        pending_count: int = 0,            # number of pending orders waiting
        bse_signals:   dict = None,        # from bse_signal.load_recent()
        fundamental_modifiers: dict = None,  # from fundamental_signal.get_modifiers()
    ) -> list[Signal]:
        sc          = self.sc
        signals     = []
        recent_news = recent_news or {}
        # Track VIX for spike detection (updated at end of this call)
        vix_prev_snapshot = self._vix_prev
        bse         = bse_signals or {}
        bse_block   = bse.get("blocklist_hits", {})
        bse_boost   = bse.get("sentiment_boost", {})   # sym → threshold delta (+)
        bse_drag    = bse.get("sentiment_drag", {})     # sym → threshold delta (-)
        fund_mods   = fundamental_modifiers or {}       # sym → FundamentalModifier

        # Restore open positions from broker state (engine is stateless across runs)
        if broker_positions:
            for sym, pos in broker_positions.items():
                if sym not in self.open_positions:
                    self.open_positions[sym] = Signal(
                        symbol=sym,
                        date=pd.Timestamp(pos.entry_date),
                        direction="BUY",
                        sentiment_score=0.0,
                        zscore=-2.5,
                        quality_score=50.0,
                        india_vix=india_vix,
                        entry_price=pos.entry_price,
                    )

        # ── Check exits ──────────────────────────────────────────────────────
        for sym in list(self.open_positions.keys()):
            exit_sig = self._check_exit(
                self.open_positions[sym], date, prices, sentiment, india_vix, pending_count
            )
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

            # ── Hard gates (apply to ALL strategies) ─────────────────────────
            if sym in bse_block:
                logger.debug(f"  {sym} BSE-blocked: {bse_block[sym]}")
                continue

            fmod = fund_mods.get(sym)
            if fmod and fmod.blocked:
                logger.debug(f"  {sym} fundamental-blocked: {fmod.reason}")
                continue

            blocked, reason = check_blocklist(recent_news.get(sym, []))
            if blocked:
                logger.debug(f"  {sym} blocklist: {reason}")
                continue

            if sym not in prices or prices[sym].empty:
                continue
            price_df = prices[sym]
            if len(price_df) < sc.rolling_window + 10:
                continue

            sent      = sentiment.get(sym, None)
            has_news  = sent is not None
            sent      = sent if has_news else 0.0
            fund_sent = fmod.sentiment_delta if fmod else 0.0
            fund_z    = fmod.zscore_delta    if fmod else 0.0

            z_series = rolling_zscore(price_df["close"], sc.rolling_window, sc.return_period)
            z        = z_series.iloc[-1]
            if pd.isna(z):
                continue

            entry_price = float(price_df["close"].iloc[-1])
            bse_tag  = (" [BSE+]" if sym in bse_boost else
                        " [BSE-]" if sym in bse_drag  else "")
            fund_tag = (" [F+]"   if fmod and fmod.sentiment_delta > 0 else
                        " [F-]"   if fmod and fmod.sentiment_delta < 0 else "")

            # ── Strategy selection (highest-priority match wins) ──────────────
            strategy = self._match_strategy(
                sym=sym, z=z, sent=sent, has_news=has_news,
                fmod=fmod, fund_sent=fund_sent, fund_z=fund_z,
                bse_boost=bse_boost, bse_drag=bse_drag,
                price_df=price_df, sc=sc,
                india_vix=india_vix, vix_prev=self._vix_prev,
                quality_score=float(row.get("quality_score", 50)),
            )
            if strategy is None:
                continue

            sig = Signal(
                symbol=sym, date=date, direction="BUY",
                sentiment_score=sent, zscore=z,
                quality_score=float(row.get("quality_score", 50)),
                india_vix=india_vix, entry_price=entry_price,
                strategy=strategy,
            )
            signals.append(sig)
            self.open_positions[sym] = sig
            sent_str = f"{sent:.2f}" if has_news else "N/A"
            logger.info(f"  BUY {sym} @ ₹{entry_price:.2f} | "
                        f"strat={strategy} sent={sent_str} z={z:.2f}{bse_tag}{fund_tag}")

            if len(self.open_positions) >= sc.max_positions:
                break

        # Update VIX for next day's spike detection
        self._vix_prev = india_vix
        return signals

    def _match_strategy(self, sym, z, sent, has_news, fmod, fund_sent, fund_z,
                        bse_boost, bse_drag, price_df, sc,
                        india_vix=0.0, vix_prev=0.0,
                        quality_score=50.0) -> Optional[str]:
        """
        Returns the strategy name for this symbol, or None if no strategy fires.
        Priority order (highest → lowest):
          gap_fade > vix_spike_buy > quality_momentum >
          earnings_surprise(disabled) > buyback_arbit(disabled) > mean_reversion
        """
        # ── 1. GAP FADE — OOS Sharpe 2.19, WR 60.1%, Ann 18.2% ────────────────
        # Gap down ≥ 3% at today's open vs yesterday's close → fade at open price
        if "open" in price_df.columns and len(price_df) >= 2:
            prev_close = float(price_df["close"].iloc[-2])
            today_open = float(price_df["open"].iloc[-1])
            if prev_close > 0:
                gap_pct = (today_open - prev_close) / prev_close
                if (gap_pct <= sc.gap_fade_threshold
                        and quality_score >= sc.gap_fade_quality_min):
                    return StrategyType.GAP_FADE

        # ── 2. VIX SPIKE BUY — OOS Sharpe 2.74, WR 72.2%, Ann 6.5% ────────────
        # India VIX jumps ≥ 15% in 1 day → buy oversold stocks
        if (vix_prev > 0 and india_vix > 0
                and (india_vix - vix_prev) / vix_prev >= sc.vix_spike_jump_pct
                and z <= sc.vix_spike_z_thresh):
            return StrategyType.VIX_SPIKE_BUY

        # ── 3. QUALITY MOMENTUM — OOS Sharpe 2.90, WR 59.6%, Ann 33.2% ─────────
        # z > +2.0 AND quality ≥ 75 AND price at 20d high → ride strong trend
        if (z >= sc.qmom_z_thresh
                and quality_score >= sc.qmom_quality_min
                and len(price_df) >= sc.qmom_lookback_high + 1):
            lookback = price_df["close"].iloc[-(sc.qmom_lookback_high + 1):-1]
            today_c  = float(price_df["close"].iloc[-1])
            if today_c >= float(lookback.max()):
                return StrategyType.QUALITY_MOMENTUM

        # ── 4. EARNINGS_SURPRISE — DISABLED (OOS: Sharpe -0.16, WR 46%) ────────
        # Needs real audio sentiment_delta. Re-enable when fmod data is validated.
        # if (fmod and fmod.sentiment_delta >= sc.earnings_sent_delta_min
        #          and z >= -1.5):
        #     return StrategyType.EARNINGS_SURPRISE

        # ── 5. BUYBACK_ARBIT — DISABLED (OOS: Sharpe -3.40, only 7 trades) ─────
        # Too few events. Re-enable if buyback frequency increases.
        # if sym in bse_boost and bse_boost[sym] >= 0.2 and z < 1.0:
        #     return StrategyType.BUYBACK_ARBIT

        # ── 6. MEAN_REVERSION — OOS Sharpe 2.85, WR 55%, Ann 8.6% (primary) ────
        # Volume gate: skip if today's volume is below average (weak conviction)
        if "volume" in price_df.columns and len(price_df) >= 20:
            vol_today = float(price_df["volume"].iloc[-1])
            vol_avg   = float(price_df["volume"].iloc[-20:].mean())
            if vol_avg > 0 and vol_today < 1.5 * vol_avg:
                return None

        sent_thresh = (sc.sentiment_threshold
                       + bse_boost.get(sym, 0.0)
                       + bse_drag.get(sym, 0.0)
                       + fund_sent)
        z_thresh = sc.zscore_threshold + fund_z
        if (z < z_thresh
                and (not has_news or sent < sent_thresh)):
            return StrategyType.MEAN_REVERSION

        return None

    def _check_exit(self, pos: Signal, date: pd.Timestamp,
                    prices: dict[str, pd.DataFrame],
                    sentiment: dict[str, float],
                    india_vix: float,
                    pending_count: int = 0) -> Optional[Signal]:
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

        reason = self._exit_reason_for_strategy(
            strategy=getattr(pos, "strategy", StrategyType.MEAN_REVERSION),
            pnl=pnl, days_held=days_held, curr_sent=curr_sent, curr_z=curr_z,
            pending_count=pending_count, sc=sc,
        )

        if reason:
            logger.info(f"  SELL {sym} @ ₹{curr_price:.2f} | "
                        f"PnL={pnl*100:.1f}% | {reason.value} [{pos.strategy}]")
            return Signal(
                symbol=sym, date=date, direction="SELL",
                sentiment_score=curr_sent, zscore=curr_z if not pd.isna(curr_z) else 0.0,
                quality_score=pos.quality_score, india_vix=india_vix,
                entry_price=pos.entry_price, exit_price=curr_price,
                exit_date=date, exit_reason=reason.value, pnl_pct=pnl,
                strategy=getattr(pos, "strategy", StrategyType.MEAN_REVERSION),
            )
        return None

    def _exit_reason_for_strategy(self, strategy: str, pnl: float,
                                   days_held: int, curr_sent: float,
                                   curr_z: float, pending_count: int,
                                   sc) -> Optional[ExitReason]:
        """Strategy-specific exit logic."""
        if strategy == StrategyType.MOMENTUM:
            # Tighter stop, shorter hold, exit on trend reversal
            if pnl <= sc.momentum_stop_loss_pct:
                return ExitReason.STOP_LOSS
            if curr_z < 0.0 and curr_sent < 0.0:           # both trend+sent reversed
                return ExitReason.MOMENTUM_REVERSAL
            if pnl >= sc.momentum_profit_exit_pct:
                return ExitReason.PROFIT_WITH_PENDING
            if days_held >= sc.momentum_time_stop_days:
                return ExitReason.TIME_STOP

        elif strategy == StrategyType.EARNINGS_SURPRISE:
            # Short-term momentum window; exit on stop or time
            if pnl <= sc.earnings_stop_loss_pct:
                return ExitReason.STOP_LOSS
            if pnl >= sc.earnings_profit_exit_pct:
                return ExitReason.PROFIT_WITH_PENDING
            if days_held >= sc.earnings_time_stop_days:
                return ExitReason.TIME_STOP

        elif strategy == StrategyType.BUYBACK_ARBIT:
            # Hold until price recovers to buyback level or time stop
            if pnl <= sc.stop_loss_pct:
                return ExitReason.STOP_LOSS
            if pnl >= 0.08:                                  # 8% = typical buyback premium
                return ExitReason.PROFIT_WITH_PENDING
            if days_held >= sc.time_stop_days:
                return ExitReason.TIME_STOP

        elif strategy == StrategyType.VIX_SPIKE_BUY:
            # Quick recovery play on fear spike; tight stop, short hold
            if pnl <= sc.vix_spike_stop_pct:
                return ExitReason.STOP_LOSS
            if pnl >= sc.vix_spike_profit_pct:
                return ExitReason.PROFIT_WITH_PENDING
            if days_held >= sc.vix_spike_time_stop:
                return ExitReason.TIME_STOP

        elif strategy == StrategyType.QUALITY_MOMENTUM:
            # Trend-following; exit on reversal or profit target
            if pnl <= sc.qmom_stop_pct:
                return ExitReason.STOP_LOSS
            if not pd.isna(curr_z) and curr_z < 0.5:   # trend weakening
                return ExitReason.PRICE_RECOVERY
            if pnl >= sc.qmom_profit_pct:
                return ExitReason.PROFIT_WITH_PENDING
            if days_held >= sc.qmom_time_stop:
                return ExitReason.TIME_STOP

        elif strategy == StrategyType.GAP_FADE:
            # Very short hold — gap fills within 1-3 days or stop out
            if pnl <= sc.gap_fade_stop_pct:
                return ExitReason.STOP_LOSS
            if pnl >= sc.gap_fade_profit_pct:
                return ExitReason.PROFIT_WITH_PENDING
            if days_held >= sc.gap_fade_time_stop:
                return ExitReason.TIME_STOP

        else:  # MEAN_REVERSION (default)
            if curr_sent > sc.recovery_sentiment:
                return ExitReason.SENTIMENT_RECOVERY
            if not pd.isna(curr_z) and curr_z > sc.recovery_zscore:
                return ExitReason.PRICE_RECOVERY
            if pnl <= sc.stop_loss_pct:
                return ExitReason.STOP_LOSS
            if days_held >= sc.time_stop_days:
                return ExitReason.TIME_STOP
            if pnl >= sc.profit_exit_pct and pending_count > 0:
                return ExitReason.PROFIT_WITH_PENDING

        return None

    def reset(self):
        self.open_positions.clear()


# ─── Kelly Criterion ─────────────────────────────────────────────────────────

def kelly_fraction(strategy: str = StrategyType.MEAN_REVERSION) -> float:
    """
    Compute half-Kelly optimal fraction from paper trade history.
    Falls back to sc.max_pct_capital if history is insufficient.

    Formula: f* = (p*b - q) / b   where b = avg_win/avg_loss, p = win rate
    We use half-Kelly (f*/2) to reduce variance, capped at kelly_max_fraction.
    """
    sc = cfg.signal
    try:
        import json
        paper_state_file = cfg.data_dir / "paper_state.json"
        if not paper_state_file.exists():
            return sc.max_pct_capital

        state  = json.loads(paper_state_file.read_text())
        trades = state.get("trades", [])

        # Try strategy-specific history first; fall back to all trades
        strat_trades = [t for t in trades
                        if t.get("strategy", StrategyType.MEAN_REVERSION) == strategy]
        if len(strat_trades) < sc.kelly_min_trades:
            strat_trades = trades

        pnls = [float(t["pnl_pct"]) for t in strat_trades
                if t.get("pnl_pct") is not None]
        if len(pnls) < sc.kelly_min_trades:
            return sc.max_pct_capital

        wins   = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p <= 0]
        if not wins or not losses:
            return sc.max_pct_capital

        p     = len(wins) / len(pnls)
        b     = float(np.mean(wins)) / float(np.mean(losses))
        q     = 1.0 - p
        kelly = (p * b - q) / b if b > 0 else 0.0
        frac  = max(0.0, kelly / 2.0)          # half-Kelly

        capped = min(frac, sc.kelly_max_fraction)
        logger.debug(f"Kelly [{strategy}]: p={p:.2f} b={b:.2f} raw={kelly:.3f} "
                     f"half={frac:.3f} → {capped:.3f}")
        return max(0.02, capped)               # floor at 2%
    except Exception as e:
        logger.debug(f"Kelly calc failed ({e}) — using max_pct_capital")
        return sc.max_pct_capital


# ─── Position Sizer ───────────────────────────────────────────────────────────

def position_size(capital: float, entry_price: float,
                  avg_daily_volume: float,
                  strategy: str = StrategyType.MEAN_REVERSION) -> int:
    """
    Returns number of shares: min of three constraints:
      - 1% of avg daily volume (liquidity)
      - Kelly-optimal % of capital (adaptive to edge strength)
      - 2% capital at risk based on strategy stop-loss
    """
    if not entry_price or entry_price <= 0:
        return 0
    sc         = cfg.signal
    k_frac     = kelly_fraction(strategy)
    stop_pct   = (sc.momentum_stop_loss_pct   if strategy == StrategyType.MOMENTUM else
                  sc.earnings_stop_loss_pct    if strategy == StrategyType.EARNINGS_SURPRISE else
                  sc.stop_loss_pct)
    liq  = int(avg_daily_volume * sc.max_pct_adv)
    cap  = int((capital * k_frac) / entry_price)
    risk = int((capital * sc.risk_per_trade_pct) / (entry_price * abs(stop_pct))) \
           if stop_pct else 0
    return max(0, min(liq, cap, risk))