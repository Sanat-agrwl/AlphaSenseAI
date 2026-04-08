"""Score earnings call audio for management confidence."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import cfg

p = argparse.ArgumentParser(description="Score earnings call audio with Whisper + confidence model")
p.add_argument("--file",    type=str, help="Path to a single audio file (.mp3 / .wav / .m4a)")
p.add_argument("--symbol",  type=str, help="Stock symbol for tagging this call")
p.add_argument("--quarter", type=str, default="", help="Quarter label e.g. Q3FY25")
p.add_argument("--list",    action="store_true", help="List all scored transcripts")
p.add_argument("--anomalies", action="store_true",
               help="Show confidence-delta anomalies (1.5σ drops)")
args = p.parse_args()

if args.file:
    from alphasense.sentiment.audio import AudioPipeline
    pipe   = AudioPipeline()
    result = pipe.process(
        audio_path=args.file,
        symbol=args.symbol or Path(args.file).stem,
        quarter=args.quarter,
    )
    print(f"\nSymbol:           {result['symbol']}")
    print(f"Quarter:          {result['quarter']}")
    print(f"Confidence score: {result['confidence_score']:.1f} / 100")
    print(f"Evasion score:    {result['evasion_score']:.1f} / 100")
    print(f"Positive signals: {result.get('positive_count', 0)}")
    print(f"Negative signals: {result.get('negative_count', 0)}")
    print(f"Hedge phrases:    {result.get('hedge_count', 0)}")
    if result.get("transcript_path"):
        print(f"Transcript saved: {result['transcript_path']}")

elif args.list:
    import pandas as pd
    scored_dir = cfg.data_dir / "transcripts" / "scored"
    if not scored_dir.exists() or not list(scored_dir.glob("*.json")):
        print("No scored transcripts found — run --file first")
    else:
        import json
        rows = []
        for f in sorted(scored_dir.glob("*.json")):
            d = json.loads(f.read_text())
            rows.append(d)
        df = pd.DataFrame(rows)
        cols = [c for c in ["symbol", "quarter", "confidence_score", "evasion_score"] if c in df.columns]
        print(df[cols].sort_values("confidence_score").to_string())

elif args.anomalies:
    from alphasense.sentiment.audio import ConfidenceDeltaDetector
    detector = ConfidenceDeltaDetector()
    anomalies = detector.compute()
    if not anomalies:
        print("No confidence-delta anomalies detected")
    else:
        print(f"Anomalies detected ({len(anomalies)}):")
        for a in anomalies:
            print(f"  {a['symbol']:20s}  delta={a['delta']:.1f}  "
                  f"(prev={a['prev']:.1f} → curr={a['current']:.1f})")

else:
    p.print_help()