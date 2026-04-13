"""
Interactive Brokers Paper Trading Broker
==========================================
Uses ib_insync to connect to IB Gateway or TWS.

Setup (one-time):
  1. Download IB Gateway: https://www.interactivebrokers.com/en/trading/ibgateway.html
  2. Log in with your IBKR paper trading credentials
  3. Enable API: Configuration → API → Settings
       - Socket port: 4002 (paper) / 4001 (live)
       - Allow connections from localhost
  4. pip install ib_insync

NSE symbols: IBKR uses "SYMBOL-NSE" format (e.g. "RELIANCE-NSE")
              We auto-convert from plain NSE symbols.

Usage:
    from alphasense.broker.ibkr import IBKRBroker
    broker = IBKRBroker()
    broker.connect()
    broker.place("RELIANCE", "BUY", qty=10, price=2850.0)
    broker.disconnect()
"""

import sys, time
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg


# ─── Connection helper ────────────────────────────────────────────────────────

def _get_ib():
    """Import ib_insync lazily so missing install doesn't break other modules."""
    try:
        from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
        return IB, Stock, MarketOrder, LimitOrder, util
    except ImportError:
        raise ImportError(
            "ib_insync not installed. Run: pip install ib_insync\n"
            "Also ensure IB Gateway is running on port 4002 (paper) or 4001 (live)."
        )


def _nse_contract(symbol: str, IB_module, Stock_cls):
    """
    Create an IBKR contract for an NSE stock.
    IBKR uses exchange='NSE' and currency='INR'.
    """
    # Strip .NS suffix if someone passes yfinance-style symbol
    sym = symbol.replace(".NS", "").upper()
    contract = Stock_cls(sym, "NSE", "INR")
    return contract


# ─── IBKR Engine ──────────────────────────────────────────────────────────────

class IBKREngine:
    """
    Live/Paper execution via Interactive Brokers.

    Paper mode:  IBKR_PORT=4002  (IB Gateway paper account)
    Live mode:   IBKR_PORT=4001  (IB Gateway live account — NEVER use until ready)

    The IB Gateway must be running on the same machine (or accessible host).
    For EC2 deployment, run IB Gateway on your laptop with SSH port-forward:
        ssh -L 4002:localhost:4002 ubuntu@15.206.123.153
    """

    def __init__(self, host: str = None, port: int = None, client_id: int = 1):
        import os
        self.host      = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port      = port or int(os.getenv("IBKR_PORT", "4002"))
        self.client_id = client_id
        self.ib        = None
        self._IB       = None
        self._Stock    = None
        self._MarketOrder = None
        self._LimitOrder  = None

    def connect(self, timeout: int = 10) -> bool:
        """Connect to IB Gateway. Returns True on success."""
        try:
            IB, Stock, MarketOrder, LimitOrder, util = _get_ib()
            self._IB          = IB
            self._Stock       = Stock
            self._MarketOrder = MarketOrder
            self._LimitOrder  = LimitOrder

            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id,
                            timeout=timeout)
            acc = self.ib.managedAccounts()
            logger.info(f"IBKR connected → {self.host}:{self.port} | accounts={acc}")
            return True
        except Exception as e:
            logger.error(f"IBKR connect failed: {e}")
            logger.error(
                "Make sure IB Gateway is running and API is enabled on "
                f"port {self.port}. Paper trading port = 4002."
            )
            return False

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKR disconnected")

    def account_summary(self) -> dict:
        """Returns key account values: NetLiquidation, AvailableFunds, etc."""
        if not self.ib:
            return {}
        vals = self.ib.accountValues()
        summary = {}
        for v in vals:
            if v.tag in ("NetLiquidation", "AvailableFunds", "UnrealizedPnL",
                         "RealizedPnL", "GrossPositionValue"):
                summary[v.tag] = float(v.value) if v.value else 0.0
        return summary

    def positions(self) -> list[dict]:
        """Current open positions."""
        if not self.ib:
            return []
        result = []
        for pos in self.ib.positions():
            result.append({
                "symbol":    pos.contract.symbol,
                "qty":       pos.position,
                "avg_cost":  pos.avgCost,
                "market_val": pos.position * pos.avgCost,
            })
        return result

    def place(self, symbol: str, direction: str, qty: int,
              price: float = 0.0, order_type: str = "MARKET",
              signal_id: str = "") -> Optional[str]:
        """
        Place an order on NSE.

        Args:
            symbol:     NSE symbol (e.g. "RELIANCE")
            direction:  "BUY" or "SELL"
            qty:        Number of shares
            price:      Limit price (ignored for MARKET orders)
            order_type: "MARKET" or "LIMIT"
            signal_id:  Optional reference ID logged with the order

        Returns:
            IBKR order ID string, or None on failure
        """
        if not self.ib or not self.ib.isConnected():
            logger.error("Not connected. Call connect() first.")
            return None

        try:
            contract = _nse_contract(symbol, self.ib, self._Stock)

            # Qualify the contract (fetches IBKR's internal conId)
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                logger.error(f"Could not qualify contract for {symbol} on NSE")
                return None

            if order_type == "LIMIT" and price > 0:
                order = self._LimitOrder(direction, qty, round(price, 2))
            else:
                order = self._MarketOrder(direction, qty)

            trade = self.ib.placeOrder(contract, order)

            # Wait briefly for order acknowledgement
            for _ in range(10):
                self.ib.sleep(0.5)
                if trade.orderStatus.status in (
                    "Submitted", "Filled", "PreSubmitted"
                ):
                    break

            oid = str(trade.order.orderId)
            status = trade.orderStatus.status
            fill   = trade.orderStatus.avgFillPrice or price

            icon = "📗" if direction == "BUY" else "📕"
            logger.info(
                f"{icon} {direction} {symbol} ×{qty} @ ₹{fill:,.2f} "
                f"| status={status} | oid={oid} | ref={signal_id}"
            )
            return oid

        except Exception as e:
            logger.error(f"IBKR order failed for {symbol}: {e}")
            return None

    def cancel_all(self):
        """Cancel all open orders."""
        if not self.ib:
            return
        self.ib.reqGlobalCancel()
        logger.info("All open orders cancelled")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()


# ─── Unified IBKR Broker (paper + live via same interface as kite.Broker) ────

class IBKRBroker:
    """
    Drop-in replacement for kite.Broker — same .place() interface.
    Mode is controlled by IBKR_PORT:
        4002 → paper (default, safe)
        4001 → live  (real money, only when ready)
    """

    def __init__(self):
        import os
        port = int(os.getenv("IBKR_PORT", "4002"))
        self.mode   = "paper" if port == 4002 else "live"
        self.engine = IBKREngine(port=port)
        self._connected = False

        if self.mode == "live":
            logger.warning("🔴 IBKR LIVE MODE — real money will be at risk")
        else:
            logger.info("🔵 IBKR PAPER MODE (port 4002)")

    def connect(self) -> bool:
        self._connected = self.engine.connect()
        return self._connected

    def disconnect(self):
        self.engine.disconnect()
        self._connected = False

    def place(self, symbol: str, direction: str, qty: int,
              price: float = 0.0, signal_id: str = "") -> Optional[str]:
        if not self._connected:
            if not self.connect():
                logger.error("Cannot place order — IBKR not connected")
                return None
        return self.engine.place(symbol, direction, qty, price,
                                 signal_id=signal_id)

    def portfolio_value(self) -> float:
        if not self._connected:
            return 0.0
        summary = self.engine.account_summary()
        return summary.get("NetLiquidation", 0.0)

    def positions(self) -> list[dict]:
        if not self._connected:
            return []
        return self.engine.positions()

    def account_summary(self) -> dict:
        if not self._connected:
            return {}
        return self.engine.account_summary()
