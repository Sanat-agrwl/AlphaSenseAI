"""
Text Sentiment Pipeline
========================
Two-stage NLP pipeline:

Stage 1 — Relevance Filter
  Fast keyword heuristic: skip CSR notices, AGM notices, book closure etc.
  Only business-material news reaches Stage 2.

Stage 2 — Sentiment Scorer
  Base: ProsusAI/finbert   →  scores -1 (very negative) to +1 (very positive)
  Fine-tuned: finbert_indian (trained on Indian market events with price labels)

Fine-tuner: SentimentFineTuner
  Uses price-as-label ground truth from labeled_events.parquet.
  Saves model to data/models/finbert_indian/.

Run:
    python scripts/score_news.py --today
    python scripts/score_news.py --finetune
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

# ─── Stage 1: Relevance Classifier ───────────────────────────────────────────

IRRELEVANT_KW = [
    "csr activity", "corporate social responsibility",
    "board meeting notice", "record date", "trading window closure",
    "book closure", "loss of share certificate", "duplicate share",
    "annual report", "notice of agm", "newspaper advertisement",
]

RELEVANT_KW = [
    "financial results", "quarterly results", "revenue", "profit", "loss",
    "earnings", "guidance", "acquisition", "merger", "demerger",
    "default", "fraud", "regulatory action", "credit rating", "downgrade",
    "upgrade", "resignation of", "appointment of", "ceo", "cfo", "md",
    "buyback", "dividend", "bonus", "order win", "contract", "deal",
    "capex", "expansion", "capacity", "debt", "fundraise", "qip",
]


class RelevanceFilter:
    def classify(self, headline: str, body: str = "") -> tuple[bool, float]:
        text = f"{headline} {body}".lower()
        for kw in IRRELEVANT_KW:
            if kw in text:
                return False, 0.9
        for kw in RELEVANT_KW:
            if kw in text:
                return True, 0.9
        return True, 0.5   # default: pass through (better to over-include)


# ─── Stage 2: Sentiment Scorer ───────────────────────────────────────────────

class SentimentScorer:
    """Wraps FinBERT (base or fine-tuned) for sentiment scoring."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or cfg.sentiment.base_model
        self.pipe = None

    def load(self):
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if __import__("torch").cuda.is_available() else -1
        logger.info(f"Loading sentiment model: {self.model_path}")
        self.pipe = hf_pipeline(
            "sentiment-analysis",
            model=self.model_path,
            device=device,
            max_length=cfg.sentiment.max_length,
            truncation=True,
        )
        logger.info("Model ready")

    def _map(self, result: dict) -> tuple[float, str]:
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":  return  score, "positive"
        if label == "negative":  return -score, "negative"
        return 0.0, "neutral"

    def score(self, text: str) -> tuple[float, str]:
        if self.pipe is None:
            self.load()
        return self._map(self.pipe(text[:cfg.sentiment.max_length])[0])

    def score_batch(self, texts: list[str]) -> list[tuple[float, str]]:
        if self.pipe is None:
            self.load()
        results = self.pipe(
            [t[:cfg.sentiment.max_length] for t in texts],
            batch_size=32,
        )
        return [self._map(r) for r in results]


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class TextSentimentPipeline:
    """Relevance filter → sentiment scorer → enriched DataFrame."""

    def __init__(self, use_finetuned: bool = True):
        model_path = cfg.sentiment.tuned_dir \
            if use_finetuned and Path(cfg.sentiment.tuned_dir).exists() \
            else cfg.sentiment.base_model
        self.filter  = RelevanceFilter()
        self.scorer  = SentimentScorer(model_path)
        self._loaded = False

    def load(self):
        self.scorer.load()
        self._loaded = True

    def process(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Input: DataFrame with columns headline, body (optional).
        Output: adds is_relevant, sentiment_score, sentiment_label.
        """
        if not self._loaded:
            self.load()

        df = news_df.copy()

        # Stage 1
        rel_results = df.apply(
            lambda r: self.filter.classify(r.get("headline", ""), r.get("body", "")),
            axis=1
        )
        df["is_relevant"]         = [r[0] for r in rel_results]
        df["relevance_confidence"] = [r[1] for r in rel_results]

        # Stage 2
        mask  = df["is_relevant"]
        texts = df.loc[mask].apply(
            lambda r: f"{r.get('headline','')} {r.get('body','')[:500]}", axis=1
        ).tolist()

        df["sentiment_score"] = np.nan
        df["sentiment_label"] = ""

        if texts:
            scored = self.scorer.score_batch(texts)
            df.loc[mask, "sentiment_score"] = [s[0] for s in scored]
            df.loc[mask, "sentiment_label"] = [s[1] for s in scored]

        df.loc[~mask, "sentiment_label"] = "irrelevant"
        logger.info(f"Processed {len(df)} articles: {mask.sum()} relevant")
        return df

    def score_today(self) -> pd.DataFrame:
        """Load today's news, score, save scored_news.parquet."""
        from alphasense.data.news_client import load_news
        date_str = __import__("datetime").datetime.now().strftime("%Y%m%d")
        news_df  = load_news(date_str)
        if news_df.empty:
            logger.warning("No news for today")
            return news_df
        scored = self.process(news_df)
        out    = cfg.data_dir / "scored_news.parquet"
        # Append to existing file
        if out.exists():
            old    = pd.read_parquet(out)
            scored = pd.concat([old, scored]).drop_duplicates(
                subset=["url"] if "url" in scored.columns else None
            ).reset_index(drop=True)
        scored.to_parquet(out, index=False)
        logger.info(f"Scored news saved: {out}")
        return scored


# ─── Fine-Tuner ───────────────────────────────────────────────────────────────

class SentimentFineTuner:
    """Fine-tune FinBERT on labeled_events.parquet using price-as-label."""

    def prepare_dataset(self, events_path: Path):
        from datasets import Dataset
        df = pd.read_parquet(events_path).dropna(subset=["return_5d", "headline"])
        sc = cfg.sentiment

        def _label(ret):
            if ret < sc.neg_threshold: return 0
            if ret > sc.pos_threshold: return 2
            return 1

        df["label"] = df["return_5d"].apply(_label)
        if "body" in df.columns:
            df["text"] = df["headline"] + " " + df["body"].fillna("")
        else:
            df["text"] = df["headline"]
        df["text"] = df["text"].str[:1000]

        neg, neu, pos = (df["label"] == 0).sum(), (df["label"] == 1).sum(), (df["label"] == 2).sum()
        logger.info(f"Dataset: {len(df)} samples | neg={neg} neu={neu} pos={pos}")
        return Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def train(self, events_path: Path = None):
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   TrainingArguments, Trainer)
        import torch

        events_path = events_path or (cfg.data_dir / "labeled_events.parquet")
        ds          = self.prepare_dataset(events_path)
        split       = ds.train_test_split(test_size=0.2, seed=42)
        sc          = cfg.sentiment

        tokenizer = AutoTokenizer.from_pretrained(sc.base_model)
        model     = AutoModelForSequenceClassification.from_pretrained(sc.base_model, num_labels=3)

        def _tok(batch):
            return tokenizer(batch["text"], padding="max_length",
                             truncation=True, max_length=sc.max_length)

        train_ds = split["train"].map(_tok, batched=True)
        eval_ds  = split["test"].map(_tok,  batched=True)

        args = TrainingArguments(
            output_dir=sc.tuned_dir,
            num_train_epochs=sc.num_epochs,
            per_device_train_batch_size=sc.batch_size,
            per_device_eval_batch_size=sc.batch_size,
            learning_rate=sc.learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
        )
        Trainer(model=model, args=args, tokenizer=tokenizer,
                train_dataset=train_ds, eval_dataset=eval_ds).train()

        model.save_pretrained(sc.tuned_dir)
        tokenizer.save_pretrained(sc.tuned_dir)
        logger.success(f"Fine-tuned model saved: {sc.tuned_dir}")


def load_scored_news() -> pd.DataFrame:
    path = cfg.data_dir / "scored_news.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def aggregate_sentiment_by_stock(scored_df: pd.DataFrame,
                                  as_of: pd.Timestamp = None,
                                  lookback_days: int = 5) -> dict[str, float]:
    """
    Return {symbol: mean_sentiment_score} for the lookback window.
    Only uses data strictly before as_of (point-in-time safe).
    """
    if scored_df.empty:
        return {}
    df = scored_df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"])
    if as_of is not None:
        df = df[df["published_at"] < as_of]
    cutoff = (as_of or pd.Timestamp.now()) - pd.Timedelta(days=lookback_days)
    df = df[df["published_at"] >= cutoff]
    df = df[df["is_relevant"] == True].dropna(subset=["sentiment_score"])

    if "matched_symbols" not in df.columns:
        return {}

    df_exp = df.explode("matched_symbols").dropna(subset=["matched_symbols"])
    return df_exp.groupby("matched_symbols")["sentiment_score"].mean().to_dict()