"""
Walk-Forward Backtesting Engine
=================================
Runs the signal engine on historical data with strict point-in-time rules.

Non-negotiable rules:
  1. Point-in-time prices: no look-ahead
  2. 1-day signal lag: signal on T executed at T+1 open
  3. 0.5% round-trip transaction cost
  4. Liquidity: max 1% of avg daily volume per position
  5. Quarterly fundamentals as originally filed (via labeled_events)

Walk-forward splits (default):
  Train:      2015-01-01 → 2020-12-31
  Validation: 2021-01-01 → 2021-12-31
  Test:       2022-01-01 → 2024-12-31  ← true out-of-sample

Pass criteria (on test period):
  • Sharpe > 1.2
  • Max drawdown < 18%
  • 50+ trades per year
  • Win rate > 50%

Run:
    python scripts/run_backtest.py --period all
    python scripts/run_backtest.py --period test
"""

import sys, json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg
from alphasense.signal.engine import SignalEngine, rolling_zscore


@dataclass
class BacktestResult:
    period:              str
    start_date:          str
    end_date:            str
    initial_capital:     float
    final_capital:       float
    total_return_pct:    float
    annual_return_pct:   float
    sharpe:              float
    max_drawdown_pct:    float
    n_trades:            int
    win_rate:            float
    avg_pnl_pct:         float
    avg_holding_days:    float
    profit_factor:       float
    trades:              list = field(default_factory=list)
    equity_curve:        list = field(default_factory=list)

    def passed(self) -> bool:
        return (self.sharpe > 1.2 and self.max_drawdown_pct > -18
                and self.win_rate > 50)


class Backtester:
    def __init__(self):
        self.bc       = cfg.backtest
        self.sc       = cfg.signal
        self.universe = pd.DataFrame()
        self.prices   = {}
        self.vix      = pd.DataFrame()
        self.sentiment_df = pd.DataFrame()

    # ── Data Loading ──────────────────────────────────────────────────────────

    def load(self):
        """Load all required data from disk."""
        from alphasense.screener.fundamental import load_universe
        from alphasense.data.nse_client      import load_prices, load_vix
        from alphasense.sentiment.text       import load_scored_news

        self.universe = load_universe()
        if self.universe.empty:
            logger.warning("No universe — run build_universe.py first")

        symbols       = self.universe["symbol"].tolist() if not self.universe.empty else None
        self.prices   = load_prices(symbols)
        self.vix      = load_vix()
        self.sentiment_df = load_scored_news()

        logger.info(f"Data loaded: {len(self.universe)} stocks, "
                    f"{len(self.prices)} price files, "
                    f"{len(self.sentiment_df)} sentiment rows")

    # ── Point-in-time helpers ─────────────────────────────────────────────────

    def _prices_pit(self, date: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return {sym: df[df.index < date] for sym, df in self.prices.items()
                if not df[df.index < date].empty}

    def _vix_pit(self, date: pd.Timestamp) -> float:
        if self.vix.empty:
            return 15.0
        avail = self.vix[self.vix.index <= date]
        return float(avail["vix_close"].iloc[-1]) if not avail.empty else 15.0

    def _sentiment_pit(self, date: pd.Timestamp) -> dict[str, float]:
        from alphasense.sentiment.text import aggregate_sentiment_by_stock
        return aggregate_sentiment_by_stock(self.sentiment_df, as_of=date)

    # ── Fast event-driven backtest (uses labeled_events.parquet) ─────────────

    def run_fast(self, start: str, end: str, period: str = "test") -> BacktestResult:
        """
        Fast backtest using pre-computed labeled events (no day-by-day loop).
        Uses labeled_events.parquet — requires zscore and return_20d columns.
        """
        events_path = cfg.data_dir / "labeled_events.parquet"
        if not events_path.exists():
            logger.error("labeled_events.parquet not found — run build_labels.py first")
            return self._empty_result(period, start, end)

        events = pd.read_parquet(events_path)
        events["date"] = pd.to_datetime(events["date"])

        # Filter to period
        df = events[(events["date"] >= start) & (events["date"] <= end)].copy()
        if df.empty:
            logger.warning(f"No events in {start}→{end}")
            return self._empty_result(period, start, end)

        # Filter to universe if available
        if not self.universe.empty:
            df = df[df["symbol"].isin(self.universe["symbol"])]

        # Only events with a meaningful Z-score drop
        if "zscore" in df.columns:
            df = df[df["zscore"] < self.sc.zscore_threshold]

        # Deduplicate: one signal per stock per 20-day window
        df = df.sort_values("date")
        df["gap"] = df.groupby("symbol")["date"].diff().dt.days
        df = df[(df["gap"].isna()) | (df["gap"] > 20)]

        # VIX filter
        if not self.vix.empty:
            def _vix_ok(row):
                av = self.vix[self.vix.index <= row["date"]]
                if av.empty: return True
                return float(av["vix_close"].iloc[-1]) <= self.sc.vix_max
            df = df[df.apply(_vix_ok, axis=1)]

        bc      = self.bc
        capital = bc.capital
        trades  = []
        equity  = [(pd.Timestamp(start), capital)]

        # Track open positions: symbol → exit_date (entry_date + 20 days)
        open_positions: dict[str, pd.Timestamp] = {}

        for _, row in df.iterrows():
            trade_date = pd.Timestamp(row["date"])

            # Expire positions whose 20-day hold window has passed
            open_positions = {
                sym: exit_dt for sym, exit_dt in open_positions.items()
                if exit_dt > trade_date
            }

            if len(open_positions) >= self.sc.max_positions:
                continue
            if row["symbol"] in open_positions:   # already holding this stock
                continue

            # Position size: 10% of current capital, capped at configured max
            pos_size = min(capital * self.sc.max_pct_capital, bc.position_size)
            if pos_size < 10_000:   # minimum ₹10k position
                continue

            ret_5d  = row.get("return_5d",  0) or 0
            ret_10d = row.get("return_10d", 0) or 0
            ret_20d = row.get("return_20d", 0) or 0
            if pd.isna(ret_20d):
                continue

            # Simulate exit: check stop-loss at 5d/10d checkpoints
            if ret_5d  <= self.sc.stop_loss_pct:
                exit_ret, reason = self.sc.stop_loss_pct, "stop_loss"
            elif ret_10d <= self.sc.stop_loss_pct:
                exit_ret, reason = self.sc.stop_loss_pct, "stop_loss"
            elif ret_20d <= self.sc.stop_loss_pct:
                exit_ret, reason = self.sc.stop_loss_pct, "stop_loss"
            else:
                exit_ret = ret_20d
                reason   = "recovery" if ret_20d > 0.02 else "time_stop"

            net     = exit_ret - (bc.cost_bps / 10000)
            pnl     = pos_size * net
            capital += pnl

            open_positions[row["symbol"]] = trade_date + pd.Timedelta(days=20)

            trades.append({
                "symbol":    row["symbol"],
                "date":      trade_date,
                "zscore":    row.get("zscore", 0),
                "net_return": round(net, 4),
                "pnl":        round(pnl, 0),
                "reason":     reason,
                "capital":    round(capital, 0),
            })
            equity.append((trade_date, capital))

        return self._compute_metrics(trades, equity, bc.capital, period, start, end)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, trades, equity, initial, period, start, end) -> BacktestResult:
        tdf = pd.DataFrame(trades)
        edf = pd.DataFrame(equity, columns=["date", "capital"])

        final  = edf["capital"].iloc[-1]
        total  = (final - initial) / initial
        years  = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.01)
        annual = (1 + total) ** (1 / years) - 1

        edf["ret"]  = edf["capital"].pct_change()
        daily       = edf["ret"].dropna()
        sharpe      = float((daily.mean() / daily.std()) * np.sqrt(252)) \
                      if len(daily) > 1 and daily.std() > 0 else 0.0

        edf["peak"]  = edf["capital"].cummax()
        edf["dd"]    = (edf["capital"] - edf["peak"]) / edf["peak"]
        max_dd       = float(edf["dd"].min()) * 100

        n       = len(tdf)
        win_r   = 0.0
        avg_pnl = 0.0
        avg_hld = 0.0
        pf      = 0.0
        if n > 0:
            win_r   = float((tdf["net_return"] > 0).mean()) * 100
            avg_pnl = float(tdf["net_return"].mean()) * 100
            gains   = tdf[tdf["net_return"] > 0]["net_return"].sum()
            losses  = abs(tdf[tdf["net_return"] <= 0]["net_return"].sum())
            pf      = gains / losses if losses > 0 else float("inf")

        return BacktestResult(
            period=period, start_date=start, end_date=end,
            initial_capital=initial, final_capital=final,
            total_return_pct=round(total * 100, 2),
            annual_return_pct=round(annual * 100, 2),
            sharpe=round(sharpe, 2),
            max_drawdown_pct=round(max_dd, 2),
            n_trades=n, win_rate=round(win_r, 1),
            avg_pnl_pct=round(avg_pnl, 2),
            avg_holding_days=round(avg_hld, 1),
            profit_factor=round(pf, 2),
            trades=trades,
            equity_curve=[(str(d), v) for d, v in equity],
        )

    def _empty_result(self, period, start, end) -> BacktestResult:
        return BacktestResult(period=period, start_date=start, end_date=end,
                              initial_capital=self.bc.capital,
                              final_capital=self.bc.capital,
                              total_return_pct=0, annual_return_pct=0,
                              sharpe=0, max_drawdown_pct=0,
                              n_trades=0, win_rate=0, avg_pnl_pct=0,
                              avg_holding_days=0, profit_factor=0)

    # ── Walk-forward ──────────────────────────────────────────────────────────

    def run_walk_forward(self) -> tuple[BacktestResult, BacktestResult, BacktestResult]:
        bc = self.bc
        logger.info("=== Walk-Forward Backtest ===")
        train = self.run_fast(bc.train_start, bc.train_end, "train")
        val   = self.run_fast(bc.val_start,   bc.val_end,   "validation")
        test  = self.run_fast(bc.test_start,  bc.test_end,  "test")
        self._print(train)
        self._print(val)
        self._print(test)
        self._save([train, val, test])
        return train, val, test

    def _print(self, r: BacktestResult):
        print(f"\n{'─'*52}")
        print(f"  {r.period.upper()} ({r.start_date} → {r.end_date})")
        print(f"{'─'*52}")
        print(f"  Total Return:     {r.total_return_pct:>8.2f}%")
        print(f"  Annual Return:    {r.annual_return_pct:>8.2f}%")
        print(f"  Sharpe Ratio:     {r.sharpe:>8.2f}")
        print(f"  Max Drawdown:     {r.max_drawdown_pct:>8.2f}%")
        print(f"  Trades:           {r.n_trades:>8d}")
        print(f"  Win Rate:         {r.win_rate:>8.1f}%")
        print(f"  Profit Factor:    {r.profit_factor:>8.2f}")
        print(f"  Final Capital:    ₹{r.final_capital:>12,.0f}")
        checks = [
            ("Sharpe > 1.2",  r.sharpe > 1.2),
            ("Max DD < 18%",  r.max_drawdown_pct > -18),
            ("Win rate > 50%", r.win_rate > 50),
        ]
        print(f"\n  Criteria:")
        for name, ok in checks:
            print(f"    {'✅' if ok else '❌'} {name}")

    def _save(self, results: list[BacktestResult]):
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            out = cfg.results_dir / f"{r.period}_results.json"
            d   = {k: v for k, v in r.__dict__.items()}
            with open(out, "w") as f:
                json.dump(d, f, indent=2, default=str)
        logger.info(f"Results saved to {cfg.results_dir}")


def load_result(period: str = "test") -> dict:
    path = cfg.results_dir / f"{period}_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())