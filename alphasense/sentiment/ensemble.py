"""
Multi-Model Ensemble Scorer
============================
Combines signals from multiple LLMs on each news headline:
  - FinBERT       (fine-tuned financial NLP — base model)
  - Claude        (news analyst — qualitative reasoning)
  - GPT-4o        (bull-case perspective)

Each model votes -1.0 → +1.0. Weighted average produces final score.
Brier score is tracked per model for calibration over time.

Usage:
    from alphasense.sentiment.ensemble import EnsembleScorer
    scorer = EnsembleScorer()
    result = scorer.score("HDFC Bank Q3 profit drops 40%", symbol="HDFCBANK")
    # result.score, result.confidence, result.votes
"""
import os, json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg


@dataclass
class ModelVote:
    model:      str
    score:      float          # -1.0 → +1.0
    confidence: float          # 0.0 → 1.0
    reasoning:  str = ""


@dataclass
class EnsembleResult:
    headline:    str
    symbol:      str
    final_score: float          # weighted average
    confidence:  float          # avg confidence
    votes:       list[ModelVote] = field(default_factory=list)
    is_signal:   bool = False   # True if final_score < threshold

    def to_dict(self) -> dict:
        return {
            "headline":    self.headline,
            "symbol":      self.symbol,
            "final_score": round(self.final_score, 4),
            "confidence":  round(self.confidence, 4),
            "is_signal":   self.is_signal,
            "votes":       [{"model": v.model, "score": round(v.score, 4),
                             "confidence": round(v.confidence, 4),
                             "reasoning": v.reasoning} for v in self.votes],
        }


# ── Individual model scorers ──────────────────────────────────────────────────

def _score_finbert(headline: str) -> ModelVote:
    """FinBERT: fast, domain-tuned, no API cost."""
    try:
        from transformers import pipeline as hf_pipeline
        _cache = _score_finbert.__dict__.setdefault("_pipe", None)
        if _cache is None:
            # Check for fine-tuned model first
            ft_path = cfg.base_dir / "models" / "finbert_indian"
            model   = str(ft_path) if ft_path.exists() else "ProsusAI/finbert"
            _score_finbert._pipe = hf_pipeline(
                "text-classification", model=model, top_k=None, device=-1
            )
        pipe = _score_finbert._pipe
        out  = pipe(headline[:512])[0]
        scores = {r["label"]: r["score"] for r in out}
        label  = max(scores, key=scores.get)
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        raw_score = label_map[label] * scores[label]
        return ModelVote(
            model="finbert",
            score=raw_score,
            confidence=scores[label],
            reasoning=f"{label} ({scores[label]:.2f})",
        )
    except Exception as e:
        logger.warning(f"FinBERT error: {e}")
        return ModelVote(model="finbert", score=0.0, confidence=0.0)


def _score_claude(headline: str, symbol: str, context: str = "") -> ModelVote:
    """Claude: qualitative analyst with market context reasoning."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ModelVote(model="claude", score=0.0, confidence=0.0,
                         reasoning="no API key")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are an expert Indian equity analyst. Rate the sentiment of this headline for stock {symbol}.

Headline: "{headline}"
{f'Context: {context}' if context else ''}

Respond with JSON only:
{{
  "score": <float from -1.0 (very negative) to +1.0 (very positive)>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence>"
}}

Focus on: earnings impact, management credibility, regulatory risk, sector tailwinds/headwinds."""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",   # fast + cheap for bulk scoring
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        resp = json.loads(msg.content[0].text.strip())
        return ModelVote(
            model="claude",
            score=max(-1.0, min(1.0, float(resp["score"]))),
            confidence=max(0.0, min(1.0, float(resp["confidence"]))),
            reasoning=resp.get("reasoning", ""),
        )
    except Exception as e:
        logger.warning(f"Claude scoring error: {e}")
        return ModelVote(model="claude", score=0.0, confidence=0.0)


def _score_gpt4o(headline: str, symbol: str) -> ModelVote:
    """GPT-4o: second opinion / bull-case perspective."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ModelVote(model="gpt4o", score=0.0, confidence=0.0,
                         reasoning="no API key")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"""Rate the market sentiment of this Indian equity headline for {symbol}.
Headline: "{headline}"
Reply with JSON only: {{"score": <-1.0 to 1.0>, "confidence": <0 to 1>, "reasoning": "<one line>"}}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",    # cheap, fast
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        d = json.loads(resp.choices[0].message.content)
        return ModelVote(
            model="gpt4o",
            score=max(-1.0, min(1.0, float(d["score"]))),
            confidence=max(0.0, min(1.0, float(d["confidence"]))),
            reasoning=d.get("reasoning", ""),
        )
    except Exception as e:
        logger.warning(f"GPT-4o scoring error: {e}")
        return ModelVote(model="gpt4o", score=0.0, confidence=0.0)


# ── Ensemble ──────────────────────────────────────────────────────────────────

# Model weights — FinBERT anchors, LLMs refine
MODEL_WEIGHTS = {
    "finbert": 0.50,
    "claude":  0.30,
    "gpt4o":   0.20,
}


class EnsembleScorer:
    """
    Score a headline using all available models.
    Models with no API key are skipped; weights redistributed automatically.
    """

    def __init__(self, use_claude: bool = True, use_gpt4o: bool = True):
        self.use_claude = use_claude and bool(os.getenv("ANTHROPIC_API_KEY"))
        self.use_gpt4o  = use_gpt4o  and bool(os.getenv("OPENAI_API_KEY"))
        active = ["finbert"]
        if self.use_claude: active.append("claude")
        if self.use_gpt4o:  active.append("gpt4o")
        logger.info(f"EnsembleScorer: active models = {active}")

    def score(self, headline: str, symbol: str = "",
              context: str = "") -> EnsembleResult:
        votes: list[ModelVote] = []

        # Always run FinBERT (no API needed)
        votes.append(_score_finbert(headline))

        # LLMs — only if keys present
        if self.use_claude:
            votes.append(_score_claude(headline, symbol, context))
            time.sleep(0.1)   # avoid burst rate limit

        if self.use_gpt4o:
            votes.append(_score_gpt4o(headline, symbol))

        # Weighted average (skip votes with 0 confidence = failed)
        total_w = 0.0
        weighted = 0.0
        for v in votes:
            w = MODEL_WEIGHTS.get(v.model, 0.1)
            if v.confidence > 0:
                weighted += v.score * w
                total_w  += w

        final = weighted / total_w if total_w > 0 else 0.0
        avg_conf = sum(v.confidence for v in votes if v.confidence > 0) / max(1, sum(1 for v in votes if v.confidence > 0))

        result = EnsembleResult(
            headline=headline,
            symbol=symbol,
            final_score=round(final, 4),
            confidence=round(avg_conf, 4),
            votes=votes,
            is_signal=final < cfg.signal.sentiment_threshold,
        )
        return result

    def score_batch(self, articles: list[dict],
                    delay: float = 0.2) -> list[EnsembleResult]:
        """
        Score a list of articles. Each article dict must have:
          headline, matched_symbols (list)
        """
        results = []
        for a in articles:
            syms = a.get("matched_symbols", [])
            if not syms:
                continue
            # Score once per article, tag with primary symbol
            for sym in syms[:3]:   # max 3 symbols per article to save API cost
                r = self.score(a["headline"], sym, a.get("body", "")[:200])
                results.append(r)
                time.sleep(delay)
        return results

    def aggregate_by_symbol(self, results: list[EnsembleResult],
                            days: int = 7) -> dict[str, float]:
        """
        Aggregate ensemble scores to one score per stock.
        Returns: symbol → mean score (negative = bearish = potential BUY).
        """
        from collections import defaultdict
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in results:
            buckets[r.symbol].append(r.final_score)
        return {sym: sum(scores) / len(scores)
                for sym, scores in buckets.items()}


# ── Persistence ───────────────────────────────────────────────────────────────

def save_ensemble_scores(results: list[EnsembleResult], date_str: str = None) -> Path:
    if date_str is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
    out_dir = cfg.data_dir / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ensemble_{date_str}.json"
    with open(out, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logger.info(f"Saved {len(results)} ensemble scores → {out}")
    return out


def load_ensemble_scores(days: int = 7) -> list[dict]:
    from datetime import datetime, timedelta
    sent_dir = cfg.data_dir / "sentiment"
    if not sent_dir.exists():
        return []
    records = []
    for i in range(days):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        f = sent_dir / f"ensemble_{d}.json"
        if f.exists():
            records.extend(json.load(open(f)))
    return records


def aggregate_sentiment(days: int = 7) -> dict[str, float]:
    """Load recent ensemble scores and return per-symbol average."""
    from collections import defaultdict
    records = load_ensemble_scores(days)
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in records:
        buckets[r["symbol"]].append(r["final_score"])
    return {sym: round(sum(s)/len(s), 4) for sym, s in buckets.items()}
