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
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

STATE_FILE      = cfg.data_dir / "paper_state.json"
REAL_STATE_FILE = cfg.data_dir / "real_state.json"


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
      • Post-close next day (15:45 IST): pending orders fill at that day's OPEN + slippage
        (today's open is available in parquet after update_prices runs at post-close)
    This mirrors real trading — you can't buy at yesterday's close after the bell.
    """

    SLIPPAGE_BPS = 10           # 0.10%
    MAX_ORDER_VALUE = 10_00_000 # ₹10L per position
    MAX_DAILY_ORDERS = 20
    MAX_PENDING = 10            # max concurrent pending orders
    MAX_PENDING_DAYS = 3        # cancel pending orders older than this many calendar days

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
        if len(self.pending) >= self.MAX_PENDING:
            logger.warning(f"⏭  SKIP {symbol} — max pending orders ({self.MAX_PENDING}) reached")
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

    def expire_pending(self) -> int:
        """Cancel pending orders older than MAX_PENDING_DAYS. Returns count expired."""
        expired = 0
        cutoff = datetime.now() - timedelta(days=self.MAX_PENDING_DAYS)
        for sym in list(self.pending.keys()):
            try:
                signal_dt = datetime.fromisoformat(self.pending[sym].signal_date)
                if signal_dt < cutoff:
                    logger.info(f"  EXPIRE {sym} — pending since "
                                f"{signal_dt.date()} (>{self.MAX_PENDING_DAYS} days old)")
                    del self.pending[sym]
                    expired += 1
            except Exception:
                pass
        if expired:
            self.save()
            logger.info(f"Expired {expired} stale pending orders")
        return expired

    def fill_pending(self, open_prices: dict[str, float]) -> int:
        """
        Fill all pending orders at today's open prices (post-close step).
        open_prices: {symbol: today_open_price}
        Returns number of orders filled.
        """
        self.expire_pending()
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


# ─── Real Capital Manager ─────────────────────────────────────────────────────

class RealStateManager:
    """
    Tracks real Groww positions and cash separately from paper trades.
    Capital model: `capital` = liquid cash available (not yet deployed).
      BUY  → capital -= qty × fill_price
      SELL → capital += qty × fill_price   (realises P&L automatically)

    real_state.json is written on every mutation.
    """

    MAX_PCT_CAPITAL = 0.10   # allocate at most 10% of free cash per position

    def __init__(self):
        self.capital:     float = 0.0
        self.positions:   dict  = {}
        self.pending:     dict  = {}
        self.trades:      list  = []
        self.order_count: int   = 0
        self._load()

    def _load(self):
        if REAL_STATE_FILE.exists():
            try:
                s = json.loads(REAL_STATE_FILE.read_text())
                self.capital     = float(s.get("capital", 0))
                self.positions   = s.get("positions", {})
                self.pending     = s.get("pending", {})
                self.trades      = s.get("trades", [])
                self.order_count = s.get("order_count", 0)
            except Exception as e:
                logger.warning(f"Could not load real state: {e}")

    def save(self):
        state = {
            "capital":     round(self.capital, 2),
            "positions":   self.positions,
            "pending":     self.pending,
            "trades":      self.trades,
            "order_count": self.order_count,
        }
        REAL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        REAL_STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

    @property
    def available(self) -> float:
        return max(0.0, self.capital)

    def qty_for(self, price: float) -> int:
        """Shares buyable with 10% of available cash at given price."""
        if price <= 0 or self.capital < price:
            return 0
        budget = self.capital * self.MAX_PCT_CAPITAL
        return int(budget / price)

    def record_buy(self, symbol: str, qty: int, fill_price: float,
                   order_id: str, signal_id: str = ""):
        cost = qty * fill_price
        self.capital -= cost
        self.order_count += 1
        self.positions[symbol] = {
            "symbol":      symbol,
            "qty":         qty,
            "entry_price": fill_price,
            "entry_date":  datetime.now().isoformat(),
            "order_id":    order_id,
            "signal_id":   signal_id,
        }
        self.save()
        logger.info(f"Real BUY  {symbol} ×{qty} @ ₹{fill_price:.2f} | "
                    f"cost ₹{cost:,.0f} | cash left ₹{self.capital:,.0f} | {order_id}")

    def record_sell(self, symbol: str, fill_price: float,
                    exit_reason: str = "", order_id: str = ""):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return
        qty     = pos["qty"]
        proceeds = qty * fill_price
        pnl      = proceeds - qty * pos["entry_price"]
        pnl_pct  = pnl / (qty * pos["entry_price"])
        self.capital += proceeds
        self.trades.append({
            "symbol":      symbol,
            "qty":         qty,
            "entry_price": pos["entry_price"],
            "exit_price":  fill_price,
            "entry_date":  pos["entry_date"],
            "exit_date":   datetime.now().isoformat(),
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(pnl_pct, 4),
            "exit_reason": exit_reason,
            "order_id":    order_id,
        })
        self.save()
        emoji = "📈" if pnl >= 0 else "📉"
        logger.info(f"Real SELL {symbol} ×{qty} @ ₹{fill_price:.2f} | "
                    f"PnL ₹{pnl:+,.0f} ({pnl_pct*100:+.1f}%) | "
                    f"cash ₹{self.capital:,.0f} | {order_id}")


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
        self._engine = PaperEngine()
        self.actual_trade = cfg.actual_trade
        mode_label = "PAPER + LIVE (Groww)" if self.actual_trade else "PAPER ONLY"
        logger.info(f"🔵 {mode_label}")

    def _groww_buy(self, symbol: str, qty: int) -> Optional[str]:
        """Place a real BUY via Groww if actual_trade is enabled."""
        if not self.actual_trade:
            return None
        try:
            from alphasense.data.groww_client import get_groww_client
            return get_groww_client().place_market_order(symbol, qty, "BUY")
        except Exception as e:
            logger.error(f"Groww BUY failed {symbol}: {e}")
            return None

    def _groww_sell(self, symbol: str, qty: int) -> Optional[str]:
        """Place a real SELL via Groww if actual_trade is enabled."""
        if not self.actual_trade:
            return None
        try:
            from alphasense.data.groww_client import get_groww_client
            return get_groww_client().place_market_order(symbol, qty, "SELL")
        except Exception as e:
            logger.error(f"Groww SELL failed {symbol}: {e}")
            return None

    def place(self, symbol: str, direction: str, qty: int,
              price: float, signal_id: str = "") -> Optional[str]:
        result = self._engine.place(symbol, direction, qty, price, signal_id)
        if result and direction == "SELL" and self.actual_trade:
            real = RealStateManager()
            real_pos = real.positions.get(symbol)
            if real_pos:
                real_qty = real_pos["qty"]
                oid = self._groww_sell(symbol, real_qty) or ""
                real.record_sell(symbol, price, signal_id, oid)
            else:
                self._groww_sell(symbol, qty)   # fallback: use paper qty
        return result

    def stage(self, symbol: str, qty: int, signal_price: float,
              signal_id: str = "") -> Optional[str]:
        """Queue a BUY signal as pending — paper mode only."""
        if isinstance(self._engine, PaperEngine):
            return self._engine.stage(symbol, qty, signal_price, signal_id)
        logger.warning("stage() only supported in paper mode")
        return None

    def fill_pending(self, open_prices: dict[str, float]) -> int:
        """
        Fill pending orders at today's open price.
        Paper: fills at open + slippage, using paper capital sizing.
        Real:  sizes independently from RealStateManager (10% of real cash),
               places Groww market order, records in real_state.json.
        """
        if not isinstance(self._engine, PaperEngine):
            return 0
        pre_pending = set(self._engine.pending.keys())
        filled      = self._engine.fill_pending(open_prices)
        if filled > 0 and self.actual_trade:
            post_pending = set(self._engine.pending.keys())
            filled_syms  = pre_pending - post_pending
            real = RealStateManager()
            for sym in filled_syms:
                paper_pos  = self._engine.positions.get(sym)
                fill_price = paper_pos.entry_price if paper_pos else open_prices.get(sym, 0)
                if not fill_price:
                    continue
                real_qty = real.qty_for(fill_price)
                if real_qty <= 0:
                    logger.info(f"Real {sym}: ₹{real.available:,.0f} available — "
                                f"insufficient for {fill_price:.0f}/share, skipping")
                    continue
                sid = paper_pos.signal_id if paper_pos else ""
                oid = self._groww_buy(sym, real_qty) or ""
                real.record_buy(sym, real_qty, fill_price, oid, sid)
        return filled

    @property
    def paper(self) -> Optional[PaperEngine]:
        return self._engine if isinstance(self._engine, PaperEngine) else None