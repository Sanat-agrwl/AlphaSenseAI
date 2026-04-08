"""
Ground Truth Labeler
=====================
Pairs news/announcement events with forward stock price returns.
Output: labeled_events.parquet — the training dataset for FinBERT fine-tuning
and the primary input for backtesting.

For each event:
  - return_5d, return_10d, return_20d  (actual price moves after the event)
  - zscore                              (how extreme the 5d move was historically)
  - label_5d                            (negative / neutral / positive)

Run:
    python scripts/build_labels.py
"""

import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg
from alphasense.data.nse_client import load_prices


def compute_zscore_series(prices: pd.Series, window: int = 252,
                           period: int = 5) -> pd.Series:
    ret    = prices.pct_change(period)
    mu     = ret.rolling(window).mean()
    sigma  = ret.rolling(window).std().replace(0, np.nan)
    return (ret - mu) / sigma


def label_events(events_df: pd.DataFrame,
                 prices: dict[str, pd.DataFrame],
                 forward_days: list[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Compute forward returns and Z-score for each event row.
    events_df must have columns: symbol, date.
    """
    df = events_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for n in forward_days:
        df[f"return_{n}d"] = np.nan
    df["zscore"] = np.nan

    for idx, row in df.iterrows():
        sym  = row["symbol"]
        date = row["date"]
        if sym not in prices:
            continue

        price_df = prices[sym]["close"] if "close" in prices[sym].columns else prices[sym].iloc[:, 0]
        future   = price_df[price_df.index >= date]
        if len(future) < max(forward_days) + 1:
            continue

        entry = future.iloc[0]
        for n in forward_days:
            if len(future) > n:
                df.at[idx, f"return_{n}d"] = (future.iloc[n] - entry) / entry

        # Z-score of the 5d move relative to rolling history
        z_series = compute_zscore_series(price_df)
        near = z_series[z_series.index <= date]
        if not near.empty:
            df.at[idx, "zscore"] = near.iloc[-1]

    # Labels
    df["label_5d"] = df["return_5d"].apply(
        lambda x: "negative" if x < -0.02
        else "positive" if x > 0.02
        else "neutral" if pd.notna(x) else None
    )

    labeled = df.dropna(subset=["return_5d"])
    neg = (labeled["label_5d"] == "negative").sum()
    pos = (labeled["label_5d"] == "positive").sum()
    neu = (labeled["label_5d"] == "neutral").sum()
    logger.info(f"Labeled {len(labeled)}/{len(df)} events | neg={neg} neu={neu} pos={pos}")
    return df


def build_and_save():
    """
    Build labeled_events.parquet from BSE announcements + news.
    Requires price data already downloaded.
    """
    from alphasense.data.bse_client  import load_announcements
    from alphasense.data.news_client import load_news

    # ── Load events ──────────────────────────────────────────────────────────
    all_events = []

    anns = load_announcements(relevant_only=True)
    if not anns.empty:
        ann_rows = anns[["scrip_code", "company_name", "subject", "category", "date", "body"]].copy()
        ann_rows = ann_rows.rename(columns={"scrip_code": "symbol", "subject": "headline"})
        ann_rows["source"] = "BSE"
        all_events.append(ann_rows)

    news = load_news()
    if not news.empty and "matched_symbols" in news.columns:
        news_exp = news.explode("matched_symbols").dropna(subset=["matched_symbols"])
        news_exp = news_exp.rename(columns={
            "matched_symbols": "symbol",
            "published_at":    "date",
        })
        news_exp["category"] = "news"
        all_events.append(news_exp[["symbol", "date", "headline", "body", "source", "category"]])

    if not all_events:
        logger.error("No events found. Run fetch_bse and fetch_news first.")
        return

    events_df = pd.concat(all_events, ignore_index=True)
    events_df["date"] = pd.to_datetime(events_df["date"])
    logger.info(f"Total events before labeling: {len(events_df)}")

    # ── Load prices ───────────────────────────────────────────────────────────
    symbols = events_df["symbol"].unique().tolist()
    prices  = load_prices(symbols)
    logger.info(f"Prices loaded for {len(prices)} stocks")

    # ── Label ─────────────────────────────────────────────────────────────────
    labeled = label_events(events_df, prices)

    out = cfg.data_dir / "labeled_events.parquet"
    labeled.to_parquet(out, index=False)
    logger.success(f"Saved: {out} ({len(labeled)} rows)")
    return labeled