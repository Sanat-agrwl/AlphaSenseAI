"""
Broker Layer — Paper Trading + Zerodha Kite Connect
=====================================================
Two modes controlled by EXECUTION_MODE in .env:
  paper  — simulated execution with realistic slippage, state saved to disk
  live   — real orders via Zerodha Kite Connect API

IMPORTANT: Keep EXECUTION_MODE=paper until you have 4–6 months of paper
trading track record. Never push --no-verify or skip risk checks.

Usage:
    from alphasense.broker.kite import Broker
    broker = Broker()
    broker.place("RELIANCE", "BUY", qty=100, price=2850.0)
"""

import sys, json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

STATE_FILE = cfg.data_dir / "paper_state.json"


@dataclass
class Position:
    symbol:       str
    qty:          int
    entry_price:  float   # actual fill = signal_price + slippage
    entry_date:   str
    signal_id:    str   = ""
    signal_price: float = 0.0   # yesterday's close that triggered the signal
    slippage:     float = 0.0   # rupee slippage paid (entry_price - signal_price)


@dataclass
class PendingOrder:
    symbol:       str
    qty:          int
    signal_price: float   # yesterday's close that triggered the signal
    signal_date:  str     # date the signal fired (post-close)
    signal_id:    str = ""


@dataclass
class Trade:
    symbol:     str
    direction:  str
    qty:        int
    entry_price: float
    exit_price:  float
    entry_date:  str
    exit_date:   str
    pnl:         float
    pnl_pct:     float
    signal_id:   str = ""


# ─── Paper Trading Engine ────────────────────────────────────────────────────

class PaperEngine:
    """
    Simulates realistic order execution:
      • Post-close (15:45 IST): signals are queued as PENDING orders
      • Pre-market next day (08:30 IST): pending orders fill at that day's OPEN + slippage
    This mirrors real trading — you can't buy at yesterday's close after the bell.
    """

    SLIPPAGE_BPS = 10       # 0.10%
    MAX_ORDER_VALUE = 10_00_000   # ₹10L per position
    MAX_DAILY_ORDERS = 20

    def __init__(self):
        self.positions:     dict[str, Position]     = {}
        self.pending:       dict[str, PendingOrder] = {}
        self.trades:        list[Trade]             = []
        self.order_count    = 0
        self._daily_count   = 0
        self._daily_date    = None
        self._load()

    def _load(self):
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                for sym, d in state.get("positions", {}).items():
                    d.setdefault("signal_price", d.get("entry_price", 0.0))
                    self.positions[sym] = Position(**d)
                for sym, d in state.get("pending", {}).items():
                    self.pending[sym] = PendingOrder(**d)
                self.trades = [Trade(**t) for t in state.get("trades", [])]
                self.order_count = state.get("order_count", 0)
                n_pend = len(self.pending)
                logger.info(f"Paper state loaded: {len(self.positions)} positions, "
                            f"{n_pend} pending, {len(self.trades)} closed trades")
            except Exception as e:
                logger.warning(f"Could not load paper state: {e}")

    def save(self):
        state = {
            "positions":   {s: p.__dict__ for s, p in self.positions.items()},
            "pending":     {s: p.__dict__ for s, p in self.pending.items()},
            "trades":      [t.__dict__ for t in self.trades],
            "order_count": self.order_count,
        }
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def _reset_daily(self):
        today = datetime.now().date()
        if self._daily_date != today:
            self._daily_count = 0
            self._daily_date  = today

    def stage(self, symbol: str, qty: int, signal_price: float,
              signal_id: str = "") -> Optional[str]:
        """
        Queue a BUY signal as a pending order (post-close step).
        The order will fill at next-day open via fill_pending().
        """
        if symbol in self.positions:
            logger.info(f"⏭  SKIP {symbol} — already in position "
                        f"(entered {self.positions[symbol].entry_date[:10]})")
            return None
        if symbol in self.pending:
            logger.info(f"⏭  SKIP {symbol} — already pending from "
                        f"{self.pending[symbol].signal_date[:10]}")
            return None

        order_value = qty * signal_price
        if order_value > self.MAX_ORDER_VALUE:
            qty = int(self.MAX_ORDER_VALUE / signal_price)

        self.pending[symbol] = PendingOrder(
            symbol=symbol, qty=qty,
            signal_price=signal_price,
            signal_date=datetime.now().isoformat(),
            signal_id=signal_id,
        )
        logger.info(f"⏳ PENDING BUY {symbol} ×{qty} | signal @ ₹{signal_price:.2f} "
                    f"— will fill at tomorrow's open")
        self.save()
        return f"PENDING-{symbol}"

    def fill_pending(self, open_prices: dict[str, float]) -> int:
        """
        Fill all pending orders at today's open prices (pre-market step).
        open_prices: {symbol: today_open_price}
        Returns number of orders filled.
        """
        filled = 0
        for sym, order in list(self.pending.items()):
            open_px = open_prices.get(sym)
            if open_px is None or open_px <= 0:
                logger.warning(f"  No open price for {sym} — keeping pending")
                continue

            self._reset_daily()
            if self._daily_count >= self.MAX_DAILY_ORDERS:
                logger.warning("Daily order limit reached during fill_pending")
                break

            slip = open_px * (self.SLIPPAGE_BPS / 10_000)
            fill = round(open_px + slip, 2)
            self.order_count  += 1
            self._daily_count += 1
            oid = f"PAPER-{self.order_count:06d}"

            slip_rs = round(fill - open_px, 2)
            self.positions[sym] = Position(
                symbol=sym, qty=order.qty, entry_price=fill,
                entry_date=datetime.now().isoformat(),
                signal_id=order.signal_id, signal_price=order.signal_price,
                slippage=slip_rs,
            )
            del self.pending[sym]
            filled += 1
            logger.info(f"📗 FILL BUY {sym} ×{order.qty} @ ₹{fill:.2f} "
                        f"(open ₹{open_px:.2f} + slip ₹{slip_rs:.2f}) "
                        f"signal was ₹{order.signal_price:.2f} | {oid}")

        if filled:
            self.save()
        return filled

    def place(self, symbol: str, direction: str, qty: int,
              price: float, signal_id: str = "") -> Optional[str]:
        """
        SELL: exit an open position at current price + slippage.
        BUY: use stage() + fill_pending() for realistic next-open execution.
             This direct-buy path is kept for manual/emergency use only.
        """
        self._reset_daily()
        if self._daily_count >= self.MAX_DAILY_ORDERS:
            logger.warning("Daily order limit reached")
            return None

        order_value = qty * price
        if order_value > self.MAX_ORDER_VALUE:
            qty         = int(self.MAX_ORDER_VALUE / price)
            order_value = qty * price

        slip = price * (self.SLIPPAGE_BPS / 10_000)
        fill = price + slip if direction == "BUY" else price - slip
        fill = round(fill, 2)
        self.order_count  += 1
        self._daily_count += 1
        oid = f"PAPER-{self.order_count:06d}"

        if direction == "BUY":
            if symbol in self.positions:
                logger.info(f"⏭  SKIP {symbol} — already in position "
                            f"(entered {self.positions[symbol].entry_date[:10]})")
                self.order_count  -= 1
                self._daily_count -= 1
                return None
            slip_rs = round(fill - price, 2)
            self.positions[symbol] = Position(
                symbol=symbol, qty=qty, entry_price=fill,
                entry_date=datetime.now().isoformat(),
                signal_id=signal_id, signal_price=price, slippage=slip_rs,
            )
            logger.info(f"📗 BUY  {symbol} ×{qty} @ ₹{fill:,.2f} "
                        f"(signal ₹{price:.2f} + slip ₹{slip_rs:.2f}) | {oid}")

        elif direction == "SELL" and symbol in self.positions:
            pos     = self.positions.pop(symbol)
            pnl     = (fill - pos.entry_price) * pos.qty
            pnl_pct = (fill - pos.entry_price) / pos.entry_price
            self.trades.append(Trade(
                symbol=symbol, direction="SELL", qty=pos.qty,
                entry_price=pos.entry_price, exit_price=fill,
                entry_date=pos.entry_date, exit_date=datetime.now().isoformat(),
                pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 4), signal_id=signal_id,
            ))
            emoji = "📈" if pnl > 0 else "📉"
            logger.info(f"{emoji} SELL {symbol} ×{pos.qty} @ ₹{fill:,.2f} | "
                        f"PnL ₹{pnl:,.0f} ({pnl_pct*100:.1f}%) | {oid}")
        else:
            logger.warning(f"No open position for SELL {symbol}")
            return None

        self.save()
        return oid

    def portfolio_value(self, current_prices: dict[str, float] = None) -> float:
        if current_prices is None:
            return sum(p.qty * p.entry_price for p in self.positions.values())
        return sum(p.qty * current_prices.get(s, p.entry_price)
                   for s, p in self.positions.items())

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


# ─── Live Kite Connect ────────────────────────────────────────────────────────

class KiteEngine:
    """Live execution via Zerodha Kite Connect."""

    def __init__(self):
        self.kite = None

    def connect(self, request_token: str):
        try:
            from kiteconnect import KiteConnect
            kc = KiteConnect(api_key=cfg.kite.api_key)
            sess = kc.generate_session(request_token, api_secret=cfg.kite.api_secret)
            kc.set_access_token(sess["access_token"])
            self.kite = kc
            logger.info(f"Kite connected: {sess.get('user_name', '')}")
        except ImportError:
            logger.error("kiteconnect not installed — pip install kiteconnect")
        except Exception as e:
            logger.error(f"Kite connect failed: {e}")

    def place(self, symbol: str, direction: str, qty: int,
              price: float = 0, order_type: str = "MARKET") -> Optional[str]:
        if self.kite is None:
            logger.error("Not connected. Call connect(request_token) first.")
            return None
        try:
            from kiteconnect import KiteConnect
            txn  = self.kite.TRANSACTION_TYPE_BUY if direction == "BUY" \
                   else self.kite.TRANSACTION_TYPE_SELL
            otyp = self.kite.ORDER_TYPE_MARKET if order_type == "MARKET" \
                   else self.kite.ORDER_TYPE_LIMIT
            oid  = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=txn,
                quantity=qty,
                product=self.kite.PRODUCT_CNC,
                order_type=otyp,
                price=price if order_type == "LIMIT" else None,
            )
            logger.info(f"Live order placed: {direction} {symbol} ×{qty} → {oid}")
            return oid
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

    def positions(self) -> dict:
        return self.kite.positions() if self.kite else {}

    def holdings(self) -> list:
        return self.kite.holdings() if self.kite else []


# ─── Unified Broker ───────────────────────────────────────────────────────────

class Broker:
    """
    Single entry point. Delegates to PaperEngine or KiteEngine
    based on EXECUTION_MODE in .env.
    """

    def __init__(self):
        mode = cfg.kite.mode
        if mode == "live":
            logger.warning("🔴 LIVE MODE — real money at risk")
            self._engine = KiteEngine()
        else:
            logger.info("🔵 PAPER MODE")
            self._engine = PaperEngine()
        self.mode = mode

    def place(self, symbol: str, direction: str, qty: int,
              price: float, signal_id: str = "") -> Optional[str]:
        return self._engine.place(symbol, direction, qty, price, signal_id)

    def stage(self, symbol: str, qty: int, signal_price: float,
              signal_id: str = "") -> Optional[str]:
        """Queue a BUY signal as pending — paper mode only."""
        if isinstance(self._engine, PaperEngine):
            return self._engine.stage(symbol, qty, signal_price, signal_id)
        logger.warning("stage() only supported in paper mode")
        return None

    def fill_pending(self, open_prices: dict[str, float]) -> int:
        """Fill pending orders at today's open — paper mode only."""
        if isinstance(self._engine, PaperEngine):
            return self._engine.fill_pending(open_prices)
        return 0

    @property
    def paper(self) -> Optional[PaperEngine]:
        return self._engine if isinstance(self._engine, PaperEngine) else None