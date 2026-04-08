"""Run walk-forward backtest on labeled events."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphasense.backtest.engine import Backtester, load_result
from config.settings import cfg

p = argparse.ArgumentParser(description="Walk-forward backtest")
p.add_argument("--period",  choices=["train", "val", "test", "all"], default="all",
               help="Which period to run (default: all)")
p.add_argument("--results", action="store_true",
               help="Print saved results without re-running")
p.add_argument("--equity-csv", action="store_true",
               help="Export equity curve to CSV after run")
args = p.parse_args()

if args.results:
    periods = ["train", "validation", "test"] if args.period == "all" else [args.period]
    for period in periods:
        r = load_result(period)
        if not r:
            print(f"No saved results for '{period}' — run without --results first")
            continue
        print(f"\n{'─'*52}")
        print(f"  {period.upper()}  ({r.get('start_date')} → {r.get('end_date')})")
        print(f"{'─'*52}")
        print(f"  Total Return:  {r.get('total_return_pct', 0):>8.2f}%")
        print(f"  Annual Return: {r.get('annual_return_pct', 0):>8.2f}%")
        print(f"  Sharpe:        {r.get('sharpe', 0):>8.2f}")
        print(f"  Max Drawdown:  {r.get('max_drawdown_pct', 0):>8.2f}%")
        print(f"  Trades:        {r.get('n_trades', 0):>8d}")
        print(f"  Win Rate:      {r.get('win_rate', 0):>8.1f}%")
        print(f"  Profit Factor: {r.get('profit_factor', 0):>8.2f}")

else:
    bt = Backtester()
    bt.load()

    if args.period == "all":
        train, val, test = bt.run_walk_forward()
        results = [train, val, test]
    else:
        period_map = {
            "train": (cfg.backtest.train_start, cfg.backtest.train_end),
            "val":   (cfg.backtest.val_start,   cfg.backtest.val_end),
            "test":  (cfg.backtest.test_start,  cfg.backtest.test_end),
        }
        start, end = period_map[args.period]
        result = bt.run_fast(start, end, args.period)
        bt._print(result)
        bt._save([result])
        results = [result]

    if args.equity_csv:
        import pandas as pd
        for r in results:
            if r.equity_curve:
                df = pd.DataFrame(r.equity_curve, columns=["date", "capital"])
                out = cfg.results_dir / f"{r.period}_equity.csv"
                df.to_csv(out, index=False)
                print(f"Equity curve → {out}")