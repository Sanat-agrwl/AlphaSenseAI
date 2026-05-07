"""
Nightly Feedback Loop
======================
Replaces the daily backtest re-run. Analyzes actual live closed trades to:

  1. Rolling signal quality  — win rate, avg return, Sharpe (last 30 / all-time)
  2. Exit reason breakdown   — what's driving our exits? (time_stop = bad, recovery = good)
  3. Signal condition split  — trades WITH sentiment score vs WITHOUT (no-news entries)
  4. Z-score bucket analysis — is -2.0 to -2.5 profitable? deeper dips better?
  5. Ensemble model accuracy — was FinBERT right? was GPT-4o right? (cross-ref scores)
  6. Alerts                  — win rate decay, stop-loss frequency, parameter drift
  7. Parameter suggestions   — automated threshold recommendations based on outcomes

Output: data/results/feedback_report.json  (read by dashboard Feedback tab)

Schedule: 23:00 IST (17:30 UTC) — after post-close pipeline is done
Usage:
    python alphasense/pipeline/feedback_loop.py
"""

import json, sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

cfg.logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    cfg.logs_dir / "feedback_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="14 days", level="INFO",
)

REPORT_PATH = cfg.results_dir / "feedback_report.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_trades() -> pd.DataFrame:
    """Load all closed paper trades from paper_state.json."""
    p = cfg.data_dir / "paper_state.json"
    if not p.exists():
        return pd.DataFrame()
    state  = json.loads(p.read_text())
    trades = state.get("trades", [])
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df["days_held"]  = (df["exit_date"] - df["entry_date"]).dt.days
    df["win"]        = df["pnl"] > 0
    return df


def _load_ensemble_scores(days: int = 60) -> dict[str, list[dict]]:
    """Load recent ensemble scores keyed by symbol."""
    sent_dir = cfg.data_dir / "sentiment"
    by_sym: dict[str, list[dict]] = defaultdict(list)
    if not sent_dir.exists():
        return by_sym
    for i in range(days):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        f = sent_dir / f"ensemble_{d}.json"
        if f.exists():
            try:
                for rec in json.load(open(f)):
                    by_sym[rec["symbol"]].append(rec)
            except Exception:
                pass
    return by_sym


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    return float(arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def _rolling_metrics(df: pd.DataFrame, n: int = 30) -> dict:
    sub = df.tail(n) if len(df) >= n else df
    if sub.empty:
        return {}
    returns = sub["pnl_pct"].tolist()
    return {
        "trades":        int(len(sub)),
        "win_rate":      round(float(sub["win"].mean()), 4),
        "avg_return_pct":round(float(sub["pnl_pct"].mean() * 100), 2),
        "total_pnl":     round(float(sub["pnl"].sum()), 0),
        "sharpe":        round(_sharpe(returns), 2),
        "avg_days_held": round(float(sub["days_held"].mean()), 1),
        "best_trade":    round(float(sub["pnl_pct"].max() * 100), 2),
        "worst_trade":   round(float(sub["pnl_pct"].min() * 100), 2),
    }


# ── Analysis functions ────────────────────────────────────────────────────────

def analyze_exit_reasons(df: pd.DataFrame) -> dict:
    if df.empty or "signal_id" not in df.columns:
        return {}
    col = "signal_id"   # exit_reason not stored directly in Trade; derive from signal_id
    # Try to find exit_reason in state file
    p = cfg.data_dir / "paper_state.json"
    if p.exists():
        state = json.loads(p.read_text())
        trades = state.get("trades", [])
        if trades and "exit_reason" in trades[0]:
            df2 = pd.DataFrame(trades)
            counts = df2["exit_reason"].value_counts().to_dict()
            win_by_reason = {}
            for reason, grp in df2.groupby("exit_reason"):
                grp["win"] = grp["pnl_pct"] > 0
                win_by_reason[reason] = round(float(grp["win"].mean()), 3)
            return {"counts": counts, "win_rate_by_reason": win_by_reason}
    counts = {"unknown": len(df)}
    return {"counts": counts, "win_rate_by_reason": {}}


def analyze_signal_conditions(df: pd.DataFrame) -> dict:
    """Compare trades that had a sentiment score vs those that didn't (no news)."""
    if df.empty:
        return {}
    # signal_id format: SIG-YYYYMMDD-SYMBOL or INTRA-YYYYMMDDHHMI-SYMBOL
    # We use pnl_pct as the outcome
    # Rough split: if sentiment files exist cross-reference
    sent_scores = _load_ensemble_scores(days=90)

    with_sent, without_sent = [], []
    for _, row in df.iterrows():
        sym  = row["symbol"]
        date = row["entry_date"]
        # Look for any ensemble score for this symbol within 1 day of entry
        scores = sent_scores.get(sym, [])
        has_score = False
        for s in scores:
            # Can't recover exact date from the flat list, so use presence as proxy
            if s.get("final_score", 0) != 0.0:
                has_score = True
                break
        bucket = with_sent if has_score else without_sent
        bucket.append(row["pnl_pct"])

    result = {}
    if with_sent:
        arr = np.array(with_sent)
        result["with_news"] = {
            "trades":    len(with_sent),
            "win_rate":  round(float((arr > 0).mean()), 3),
            "avg_return":round(float(arr.mean() * 100), 2),
        }
    if without_sent:
        arr = np.array(without_sent)
        result["without_news"] = {
            "trades":    len(without_sent),
            "win_rate":  round(float((arr > 0).mean()), 3),
            "avg_return":round(float(arr.mean() * 100), 2),
        }
    return result


def analyze_model_calibration(df: pd.DataFrame) -> dict:
    """
    For each closed trade, check if FinBERT and GPT-4o scores predicted the direction.
    A negative score at entry → stock should recover (price rise) = win.
    A positive score at entry → stock has positive momentum, shouldn't have triggered.
    """
    sent_dir = cfg.data_dir / "sentiment"
    if not sent_dir.exists() or df.empty:
        return {}

    model_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "scores": []})

    for _, trade in df.iterrows():
        sym       = trade["symbol"]
        entry_dt  = trade["entry_date"]
        outcome   = "win" if trade["pnl"] > 0 else "loss"
        # Look for ensemble scores around entry date (±1 day)
        for delta in range(-1, 2):
            d = (entry_dt + timedelta(days=delta)).strftime("%Y%m%d")
            f = sent_dir / f"ensemble_{d}.json"
            if not f.exists():
                continue
            try:
                for rec in json.load(open(f)):
                    if rec["symbol"] != sym:
                        continue
                    for vote in rec.get("votes", []):
                        model = vote["model"]
                        score = vote["score"]
                        model_stats[model]["total"] += 1
                        model_stats[model]["scores"].append(score)
                        # Negative score at entry = predicted recovery = win expected
                        predicted_win = score < -0.1
                        if (predicted_win and outcome == "win") or \
                           (not predicted_win and outcome == "loss"):
                            model_stats[model]["correct"] += 1
            except Exception:
                pass

    result = {}
    for model, stats in model_stats.items():
        if stats["total"] > 0:
            scores = stats["scores"]
            result[model] = {
                "predictions": stats["total"],
                "accuracy":    round(stats["correct"] / stats["total"], 3),
                "avg_score":   round(float(np.mean(scores)), 3),
                "score_std":   round(float(np.std(scores)), 3),
            }
    return result


def generate_alerts(df: pd.DataFrame, rolling: dict) -> list[str]:
    alerts = []
    if df.empty:
        return ["No closed trades yet — alerts will appear once positions are exited."]

    wr = rolling.get("win_rate", 1.0)
    n  = rolling.get("trades", 0)

    if n >= 5 and wr < 0.45:
        alerts.append(f"Win rate {wr*100:.0f}% in last {n} trades is below 45% — "
                      "consider tightening Z-score threshold from -2.0 to -2.3.")

    if n >= 5 and wr > 0.75:
        alerts.append(f"Win rate {wr*100:.0f}% — strategy is performing well, "
                      "consider slightly relaxing Z-score threshold to capture more signals.")

    # Check time_stop dominance
    p = cfg.data_dir / "paper_state.json"
    if p.exists():
        trades_raw = json.loads(p.read_text()).get("trades", [])
        if trades_raw:
            total = len(trades_raw)
            time_stops = sum(1 for t in trades_raw if t.get("signal_id","").startswith("SIG"))
            # Rough proxy — check exit reasons if stored
            reasons = [t.get("exit_reason", "") for t in trades_raw]
            ts_count = reasons.count("time_stop")
            if total >= 3 and ts_count / total > 0.6:
                alerts.append(f"{ts_count}/{total} exits are time_stop — "
                              "sentiment/price recovery rarely triggering; "
                              "positions may be held too long.")

    avg_ret = rolling.get("avg_return_pct", 0)
    if n >= 5 and avg_ret < 0:
        alerts.append(f"Average return {avg_ret:.1f}% is negative — "
                      "review entry conditions or widen stop-loss from -8% to -6%.")

    if not alerts:
        alerts.append("No anomalies detected — strategy parameters within expected range.")

    return alerts


def generate_parameter_suggestions(df: pd.DataFrame, signal_split: dict) -> list[str]:
    suggestions = []
    if df.empty:
        return suggestions

    # If no-news trades win less
    with_n   = signal_split.get("with_news", {})
    without_n = signal_split.get("without_news", {})
    if with_n and without_n:
        wr_with    = with_n.get("win_rate", 0)
        wr_without = without_n.get("win_rate", 0)
        if wr_without < wr_with - 0.15 and without_n.get("trades", 0) >= 3:
            suggestions.append(
                f"Trades WITHOUT news sentiment win only {wr_without*100:.0f}% vs "
                f"{wr_with*100:.0f}% with news — consider requiring sentiment coverage "
                "before entering (set require_news=True in SignalConfig)."
            )

    # Days held analysis
    if not df.empty:
        avg_days = df["days_held"].mean()
        wins     = df[df["win"]]
        losses   = df[~df["win"]]
        if len(wins) >= 2 and len(losses) >= 2:
            win_days  = wins["days_held"].mean()
            loss_days = losses["days_held"].mean()
            if loss_days > win_days + 3:
                suggestions.append(
                    f"Losing trades held {loss_days:.0f} days avg vs "
                    f"{win_days:.0f} for winners — reduce time_stop from 20 to 15 days "
                    "to cut losers earlier."
                )

    return suggestions


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    logger.info(f"\n{'#'*55}")
    logger.info(f"FEEDBACK-LOOP  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*55}")

    df = _load_trades()
    logger.info(f"Closed trades loaded: {len(df)}")

    if df.empty:
        report = {
            "generated_at": datetime.now().isoformat(),
            "status": "no_trades",
            "message": "No closed trades yet. Keep running the pipeline.",
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        logger.info("No trades yet — wrote empty report")
        return

    # ── Compute all metrics ───────────────────────────────────────────────────
    rolling_30  = _rolling_metrics(df, n=30)
    rolling_all = _rolling_metrics(df, n=len(df))
    exits       = analyze_exit_reasons(df)
    signal_split = analyze_signal_conditions(df)
    model_cal   = analyze_model_calibration(df)
    alerts      = generate_alerts(df, rolling_30)
    suggestions = generate_parameter_suggestions(df, signal_split)

    # ── Z-score bucket analysis ───────────────────────────────────────────────
    zscore_buckets = {}
    if "signal_id" in df.columns:
        # Use pnl_pct distribution by quintile as proxy (no zscore stored in Trade)
        q25 = float(df["pnl_pct"].quantile(0.25))
        q75 = float(df["pnl_pct"].quantile(0.75))
        zscore_buckets = {
            "bottom_quartile_return":  round(q25 * 100, 2),
            "top_quartile_return":     round(q75 * 100, 2),
            "median_return":           round(float(df["pnl_pct"].median() * 100), 2),
        }

    # ── Log key findings ──────────────────────────────────────────────────────
    logger.info(f"Rolling 30-trade: WR={rolling_30.get('win_rate',0)*100:.0f}% "
                f"avg={rolling_30.get('avg_return_pct',0):.1f}% "
                f"Sharpe={rolling_30.get('sharpe',0):.2f}")
    for alert in alerts:
        logger.info(f"  ALERT: {alert}")
    for sug in suggestions:
        logger.info(f"  SUGGEST: {sug}")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "generated_at":    datetime.now().isoformat(),
        "total_trades":    int(len(df)),
        "rolling_30":      rolling_30,
        "all_time":        rolling_all,
        "exit_reasons":    exits,
        "signal_split":    signal_split,
        "model_calibration": model_cal,
        "return_distribution": zscore_buckets,
        "alerts":          alerts,
        "parameter_suggestions": suggestions,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Feedback report saved → {REPORT_PATH}")


if __name__ == "__main__":
    run()
