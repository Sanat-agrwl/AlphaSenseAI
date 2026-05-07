"""
Fundamental Signal Modifier
============================
Combines XBRL quarterly financials + audio confidence scores into
per-symbol modifiers that adjust (or block) the main signal engine.

Output: FundamentalModifier per symbol
  • sentiment_delta  — added to sentiment_threshold (positive = relax, negative = tighten)
  • zscore_delta     — added to zscore_threshold     (positive = relax, negative = tighten)
  • blocked          — True = skip BUY entirely
  • reason           — human-readable explanation

Logic:
  Block BUY if:
    • audio confidence < 40  AND  XBRL score == -1   (management evasive + deteriorating)
    • XBRL score == -1 (debt rising + revenue declining + poor FCF)

  Relax thresholds if:
    • audio confidence >= 70  AND  XBRL score == +1  (confident + strong fundamentals)
    → sentiment_delta = +0.15, zscore_delta = +0.3

  Tighten thresholds if:
    • audio confidence < 50  OR  XBRL score == -1
    → sentiment_delta = -0.1, zscore_delta = -0.2

These modifiers stack with BSE event signals already in engine.py.
"""

import json, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import cfg


@dataclass
class FundamentalModifier:
    symbol:         str
    sentiment_delta: float = 0.0
    zscore_delta:    float = 0.0
    blocked:         bool  = False
    reason:          str   = ""


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_audio_confidence(symbols: list[str]) -> dict[str, float]:
    """
    Returns {symbol: confidence_score (0–100)} from the latest scores JSON
    in data/transcripts/text/.  Symbols with no scores return None.
    """
    scores: dict[str, float] = {}
    text_dir = cfg.transcripts_dir / "text"
    if not text_dir.exists():
        return scores

    # Index existing score files  {symbol: most-recent path}
    latest: dict[str, Path] = {}
    for f in text_dir.glob("*_scores.json"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            sym = parts[0].upper()
            if sym in symbols:
                prev = latest.get(sym)
                if prev is None or f.stat().st_mtime > prev.stat().st_mtime:
                    latest[sym] = f

    for sym, path in latest.items():
        try:
            data = json.loads(path.read_text())
            scores[sym] = float(data.get("confidence_score", 50.0))
        except Exception:
            pass

    return scores


def load_xbrl_signals(symbols: list[str]) -> dict[str, dict]:
    """Returns {symbol: xbrl_signals_dict} from data/xbrl/ (or universe summary)."""
    from alphasense.data.xbrl_client import load_universe_summary, load_signals

    # Try universe summary first (single file, fast)
    summary = load_universe_summary()
    result  = {}
    for sym in symbols:
        if sym in summary:
            result[sym] = summary[sym]
        else:
            # Try individual file
            sig = load_signals(sym)
            if sig:
                result[sym] = sig
    return result


# ── Modifier computation ──────────────────────────────────────────────────────

def compute_modifier(symbol: str,
                     xbrl: Optional[dict],
                     audio_conf: Optional[float]) -> FundamentalModifier:
    mod = FundamentalModifier(symbol=symbol)

    xbrl_score = xbrl.get("overall_score", 0) if xbrl else 0
    has_audio  = audio_conf is not None

    # ── Hard block ────────────────────────────────────────────────────────────
    if xbrl_score == -1:
        # Fundamentals are clearly deteriorating
        if has_audio and audio_conf < 40:
            mod.blocked = True
            mod.reason  = (f"XBRL score=-1 (fundamental deterioration) "
                           f"+ audio confidence {audio_conf:.0f}/100 (evasive mgmt)")
            logger.debug(f"  {symbol}: BLOCKED — {mod.reason}")
            return mod
        else:
            # Still tighten aggressively even without audio
            mod.sentiment_delta = -0.15
            mod.zscore_delta    = -0.3
            mod.reason          = "XBRL score=-1 (tighten thresholds)"

    # ── Positive boost ────────────────────────────────────────────────────────
    elif xbrl_score == 1 and has_audio and audio_conf >= 70:
        mod.sentiment_delta = 0.15
        mod.zscore_delta    = 0.3
        mod.reason          = (f"XBRL score=+1 (strong fundamentals) "
                               f"+ audio confidence {audio_conf:.0f}/100")
        logger.debug(f"  {symbol}: BOOST — {mod.reason}")

    # ── Mild tighten if audio is evasive ─────────────────────────────────────
    elif has_audio and audio_conf < 50:
        mod.sentiment_delta = -0.08
        mod.zscore_delta    = -0.15
        mod.reason          = f"Low audio confidence ({audio_conf:.0f}/100)"

    # ── Mild relax if XBRL positive but audio neutral ─────────────────────────
    elif xbrl_score == 1 and not has_audio:
        mod.sentiment_delta = 0.08
        mod.zscore_delta    = 0.15
        mod.reason          = "XBRL score=+1 (no audio data)"

    return mod


def get_modifiers(symbols: list[str]) -> dict[str, FundamentalModifier]:
    """
    Load XBRL + audio data and compute per-symbol modifiers.
    Returns {symbol: FundamentalModifier}.
    """
    xbrl_data  = load_xbrl_signals(symbols)
    audio_data = load_audio_confidence(symbols)

    modifiers = {}
    for sym in symbols:
        mod = compute_modifier(
            sym,
            xbrl     = xbrl_data.get(sym),
            audio_conf = audio_data.get(sym),
        )
        modifiers[sym] = mod

    n_boost  = sum(1 for m in modifiers.values() if m.sentiment_delta > 0 and not m.blocked)
    n_tight  = sum(1 for m in modifiers.values() if m.sentiment_delta < 0 and not m.blocked)
    n_block  = sum(1 for m in modifiers.values() if m.blocked)
    logger.info(f"Fundamental modifiers: {n_boost} boost | {n_tight} tighten | {n_block} block "
                f"(of {len(symbols)} symbols)")
    return modifiers
