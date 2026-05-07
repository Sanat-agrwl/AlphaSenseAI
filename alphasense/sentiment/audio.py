"""
Audio Sentiment Pipeline
=========================
Turns earnings call audio into a Management Confidence Score (0–100).

Steps:
  1. Transcribe with Whisper large-v3 (handles Hindi+English)
  2. Speaker diarization with pyannote.audio (optional)
  3. Dual scoring:
       Lexical score  — forward-guidance keyword frequencies
       Evasion score  — hedge phrases, repetition, disfluency
  4. Combine → Management Confidence Score
  5. Quarter-over-quarter delta detection (leading indicator)

Run:
    python scripts/score_transcripts.py --audio path/to/call.mp3 --symbol CDSL --quarter 2024Q4
    python scripts/score_transcripts.py --batch data/transcripts/manifest.csv
"""

import sys, re, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg

# ─── Keyword dictionaries ────────────────────────────────────────────────────

POSITIVE_KW = {
    "strong": 1.0, "confident": 1.5, "optimistic": 1.5, "growth": 0.8,
    "expansion": 1.0, "capacity": 0.8, "pipeline": 0.8, "order book": 1.2,
    "robust": 1.0, "outperform": 1.5, "record": 1.2, "milestone": 1.0,
    "upgrade": 1.0, "accelerat": 1.0, "momentum": 0.8,
    "ahead of expectations": 2.0, "beat": 1.0, "margin expansion": 1.5,
    "market share gain": 1.5, "debt reduction": 1.0,
    # Hindi (transliterated)
    "achha": 0.8, "badhiya": 1.0, "mazboot": 1.0, "vikas": 0.8,
}

NEGATIVE_KW = {
    "headwind": -1.0, "challenge": -0.8, "uncertain": -1.2, "delay": -1.0,
    "deferral": -1.2, "slowdown": -1.5, "decline": -1.0, "pressure": -0.8,
    "difficult": -0.8, "miss": -1.5, "below expectation": -2.0,
    "shortfall": -1.5, "restructur": -1.0, "impair": -1.5, "write-off": -1.5,
    "investigation": -2.0, "attrition": -0.8, "one-time": -0.5,
    # Hindi
    "mushkil": -1.0, "chunauti": -0.8, "kamzor": -1.0,
}

HEDGE_PHRASES = [
    "we believe", "we expect", "we think", "we hope", "it could be",
    "it might be", "going forward", "in due course", "subject to",
    "depending on", "as and when", "we are exploring", "we are evaluating",
    "to the best of our knowledge",
]

DISFLUENCY = ["um", "uh", "you know", "basically", "actually", "sort of", "kind of"]

CUSTOM_VOCAB = [
    "SEBI", "NIFTY", "SENSEX", "NSE", "BSE", "crore", "lakh", "rupee",
    "demat", "CDSL", "NSDL", "promoter", "FII", "DII", "QIP",
    "EBITDA", "PAT", "PBT", "EPS", "capex", "opex", "ROCE", "ROE",
]


# ─── Deepgram Transcriber (primary) ──────────────────────────────────────────

class DeepgramTranscriber:
    """
    Transcribes earnings call audio via Deepgram Nova-2.
    Advantages over local Whisper:
      - Handles Hindi-English code-switching natively
      - Built-in speaker diarization (no pyannote needed)
      - ~30× faster than real-time (vs Whisper large-v3 at ~3× on CPU)
      - Supports Gujarati, Tamil, Marathi for future expansion

    Requires: DEEPGRAM_API_KEY in .env
              pip install deepgram-sdk
    """

    LANGUAGE_HINTS = ["hi", "en-IN"]   # Hindi + Indian English

    def __init__(self):
        import os
        self.api_key = os.getenv("DEEPGRAM_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, audio_path: Path, output_path: Path = None) -> dict:
        if not self.available:
            raise RuntimeError("DEEPGRAM_API_KEY not set")

        from deepgram import DeepgramClient, PrerecordedOptions
        logger.info(f"Deepgram transcribing {audio_path.name}...")

        dg      = DeepgramClient(self.api_key)
        options = PrerecordedOptions(
            model            = "nova-2",
            language         = "hi",        # Hindi — Deepgram auto-detects English within it
            detect_language  = True,
            diarize          = True,        # free speaker separation
            punctuate        = True,
            utterances       = True,        # sentence-level with speaker labels
            keywords         = CUSTOM_VOCAB,
        )

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        resp = dg.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_data, "mimetype": "audio/mpeg"},
            options,
        )

        result   = resp.results
        channel  = result.channels[0].alternatives[0]
        full_text = channel.transcript

        # Build segment list compatible with existing EarningsCallScorer
        segments = []
        if result.utterances:
            for u in result.utterances:
                segments.append({
                    "id":       u.id,
                    "start":    u.start,
                    "end":      u.end,
                    "text":     u.transcript,
                    "speaker":  f"SPEAKER_{u.speaker:02d}",
                    "confidence": u.confidence,
                })
        else:
            # Fallback: use words grouped into ~30s chunks
            words = channel.words or []
            chunk, t0 = [], 0.0
            for w in words:
                chunk.append(w.word)
                if w.end - t0 > 30 or w == words[-1]:
                    segments.append({"id": len(segments), "start": t0,
                                     "end": w.end, "text": " ".join(chunk),
                                     "speaker": "SPEAKER_00"})
                    chunk, t0 = [], w.end

        duration = segments[-1]["end"] if segments else 0.0
        transcript = {
            "text":             full_text,
            "language":         result.channels[0].detected_language or "hi",
            "segments":         segments,
            "duration_seconds": duration,
            "engine":           "deepgram-nova-2",
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))

        logger.info(f"  {duration/60:.1f} min | {transcript['language']} | "
                    f"{len(segments)} utterances")
        return transcript


# ─── Whisper Transcriber (fallback when no Deepgram key) ─────────────────────

class Transcriber:
    """Local Whisper large-v3. Slower but free. Used when DEEPGRAM_API_KEY absent."""

    def __init__(self):
        self.model = None

    def load(self):
        import whisper
        logger.info(f"Loading Whisper {cfg.audio.whisper_model}...")
        self.model = whisper.load_model(cfg.audio.whisper_model,
                                        device=cfg.audio.device)

    def transcribe(self, audio_path: Path, output_path: Path = None) -> dict:
        if self.model is None:
            self.load()
        logger.info(f"Whisper transcribing {audio_path.name}...")
        result = self.model.transcribe(
            str(audio_path),
            verbose=False,
            word_timestamps=True,
            initial_prompt=(
                "Indian company earnings call. Mix of Hindi and English. "
                "Terms: " + ", ".join(CUSTOM_VOCAB[:15])
            ),
        )
        transcript = {
            "text":     result["text"],
            "language": result.get("language", "unknown"),
            "segments": [
                {"id": s["id"], "start": s["start"], "end": s["end"],
                 "text": s["text"], "speaker": "SPEAKER_00"}
                for s in result["segments"]
            ],
            "duration_seconds": result["segments"][-1]["end"] if result["segments"] else 0,
            "engine": "whisper",
        }
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
        logger.info(f"  {transcript['duration_seconds']/60:.1f} min | {transcript['language']}")
        return transcript


def get_transcriber():
    """Returns Deepgram if key is set, otherwise falls back to Whisper."""
    dg = DeepgramTranscriber()
    if dg.available:
        logger.info("Transcriber: Deepgram Nova-2")
        return dg
    logger.info("Transcriber: Whisper (no DEEPGRAM_API_KEY)")
    return Transcriber()


# ─── Diarizer ────────────────────────────────────────────────────────────────

class Diarizer:
    def __init__(self):
        self.pipeline = None

    def load(self):
        if not cfg.audio.hf_token:
            logger.warning("No HF_TOKEN — diarization disabled")
            return
        try:
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=cfg.audio.hf_token,
            )
        except ImportError:
            logger.warning("pyannote.audio not installed — diarization disabled")

    def diarize(self, audio_path: Path) -> list[dict]:
        if self.pipeline is None:
            return []
        diar = self.pipeline(str(audio_path),
                             min_speakers=cfg.audio.min_speakers,
                             max_speakers=cfg.audio.max_speakers)
        return [{"speaker": spk, "start": t.start, "end": t.end,
                 "duration": t.end - t.start}
                for t, _, spk in diar.itertracks(yield_label=True)]

    @staticmethod
    def identify_management(diar_segs: list[dict]) -> dict[str, str]:
        """
        Heuristic: speaker with most air-time in the first third = management.
        """
        if not diar_segs:
            return {}
        total       = max(s["end"] for s in diar_segs)
        first_third = total / 3
        by_speaker_first = Counter()
        by_speaker_total = Counter()
        for s in diar_segs:
            by_speaker_total[s["speaker"]] += s["duration"]
            if s["start"] < first_third:
                overlap = min(s["end"], first_third) - s["start"]
                by_speaker_first[s["speaker"]] += overlap
        mgmt = {spk for spk, _ in by_speaker_first.most_common(2)
                if by_speaker_total[spk] > total * 0.1}
        return {spk: ("management" if spk in mgmt else "analyst")
                for spk in by_speaker_total}


# ─── Scorer ──────────────────────────────────────────────────────────────────

class EarningsCallScorer:
    def score_lexical(self, text: str) -> float:
        tl    = text.lower()
        words = tl.split()
        total = len(words) or 1
        pos   = sum(text.lower().count(kw) * w for kw, w in POSITIVE_KW.items())
        neg   = sum(text.lower().count(kw) * abs(w) for kw, w in NEGATIVE_KW.items())
        raw   = (pos - neg) / (total / 100)
        return float(max(-1.0, min(1.0, raw / 5.0)))

    def score_evasion(self, segments: list[dict],
                      speaker_roles: dict[str, str] = None) -> dict:
        full  = " ".join(s["text"] for s in segments)
        tl    = full.lower()
        words = tl.split()
        total = len(words) or 1

        hedge_count    = sum(tl.count(p) for p in HEDGE_PHRASES)
        hedge_density  = hedge_count / (total / 1000)

        sents       = [s.strip().lower() for s in re.split(r'[.!?]+', full) if len(s.strip()) > 20]
        seen, reps  = set(), 0
        for s in sents:
            fp = " ".join(s.split()[:8])
            if fp in seen: reps += 1
            seen.add(fp)
        rep_ratio = reps / max(len(sents), 1)

        disfluency_count = sum(tl.count(m) for m in DISFLUENCY)
        disfluency_rate  = disfluency_count / (total / 1000)

        avg_resp = 0.0
        if speaker_roles:
            mgmt_segs = [s for s in segments
                         if speaker_roles.get(s.get("speaker"), "") == "management"]
            if mgmt_segs:
                avg_resp = float(np.mean([len(s["text"].split()) for s in mgmt_segs]))

        hedge_score    = min(100.0, hedge_density * 15)
        rep_score      = min(100.0, rep_ratio * 500)
        disfluency_score = min(100.0, disfluency_rate * 10)
        evasion        = 0.40 * hedge_score + 0.30 * rep_score + 0.30 * disfluency_score

        return {
            "evasion_score":     round(evasion, 1),
            "hedge_density":     round(hedge_density, 2),
            "repetition_ratio":  round(rep_ratio, 3),
            "disfluency_rate":   round(disfluency_rate, 2),
            "avg_response_words": round(avg_resp, 1),
        }

    def confidence_score(self, lexical: float, evasion: dict) -> float:
        lex_comp  = (lexical + 1) * 50
        eva_comp  = 100 - evasion["evasion_score"]
        score     = 0.45 * lex_comp + 0.55 * eva_comp
        return round(max(0.0, min(100.0, score)), 1)

    def score(self, transcript: dict, speaker_roles: dict = None) -> dict:
        text     = transcript.get("text", "")
        segments = transcript.get("segments", [])
        lexical  = self.score_lexical(text)
        evasion  = self.score_evasion(segments, speaker_roles)
        conf     = self.confidence_score(lexical, evasion)
        logger.info(f"  Confidence: {conf}/100 | Lexical: {lexical:.2f} | Evasion: {evasion['evasion_score']:.1f}")
        return {
            "lexical_score":   round(lexical, 4),
            "evasion_scores":  evasion,
            "confidence_score": conf,
            "duration_min":    round(transcript.get("duration_seconds", 0) / 60, 1),
            "total_words":     len(text.split()),
        }


# ─── Delta Detector ──────────────────────────────────────────────────────────

class ConfidenceDeltaDetector:
    """Flags significant QoQ confidence drops — leading earnings indicator."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Input: symbol, quarter, confidence_score."""
        df = df.sort_values(["symbol", "quarter"]).copy()
        n  = cfg.audio.lookback_qtrs
        df["rolling_avg"] = (
            df.groupby("symbol")["confidence_score"]
              .transform(lambda x: x.rolling(n, min_periods=2).mean())
        )
        df["delta"]     = df["confidence_score"] - df["rolling_avg"]
        df["delta_std"] = (
            df.groupby("symbol")["delta"]
              .transform(lambda x: x.rolling(n, min_periods=2).std())
        )
        df["is_anomaly"] = (
            df["delta"].lt(0)
            & (df["delta"].abs() > 1.5 * df["delta_std"].fillna(1e9))
        )
        anomalies = df[df["is_anomaly"]]
        for _, r in anomalies.iterrows():
            logger.warning(f"Confidence drop: {r['symbol']} {r['quarter']} "
                           f"{r['confidence_score']:.1f} (avg {r['rolling_avg']:.1f}, Δ={r['delta']:.1f})")
        return df


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class AudioPipeline:
    def __init__(self):
        self.transcriber = get_transcriber()   # Deepgram if key set, else Whisper
        self.diarizer    = Diarizer()          # fallback pyannote (Whisper path only)
        self.scorer      = EarningsCallScorer()

    def process(self, audio_path: Path, symbol: str, quarter: str,
                output_dir: Path = None) -> dict:
        output_dir = output_dir or cfg.transcripts_dir
        t_path = output_dir / "text" / f"{symbol}_{quarter}_transcript.json"

        # Transcribe (or load cached)
        if t_path.exists():
            transcript = json.loads(t_path.read_text())
            logger.info(f"Loaded cached transcript: {t_path.name}")
        else:
            transcript = self.transcriber.transcribe(audio_path, t_path)

        # Speaker roles — Deepgram provides labels directly in segments;
        # Whisper path needs pyannote diarization separately.
        speaker_roles = {}
        if transcript.get("engine") == "deepgram-nova-2":
            # Identify management from Deepgram's diarized utterances
            diar_segs = [
                {"speaker": s["speaker"],
                 "start":   s["start"],
                 "end":     s["end"],
                 "duration": s["end"] - s["start"]}
                for s in transcript.get("segments", [])
                if "speaker" in s
            ]
            speaker_roles = Diarizer.identify_management(diar_segs)
        else:
            try:
                self.diarizer.load()
                segs          = self.diarizer.diarize(audio_path)
                speaker_roles = Diarizer.identify_management(segs)
            except Exception as e:
                logger.warning(f"Diarization skipped: {e}")

        # Score
        scores = self.scorer.score(transcript, speaker_roles)
        result = {
            "symbol":  symbol,
            "quarter": quarter,
            "engine":  transcript.get("engine", "unknown"),
            **scores,
        }

        out = output_dir / "text" / f"{symbol}_{quarter}_scores.json"
        out.write_text(json.dumps(result, indent=2))
        logger.info(f"Scores saved → {out.name}")
        return result

    def load_all_scores(self) -> pd.DataFrame:
        records = []
        for f in cfg.transcripts_dir.glob("text/*_scores.json"):
            records.append(json.loads(f.read_text()))
        df = pd.DataFrame(records)
        if not df.empty and "quarter" in df.columns:
            df = df.sort_values(["symbol", "quarter"])
        return df