"""
Copy existing trial data into the new AlphaSense project data directory.

Usage:
    python scripts/migrate_data.py [--dry-run]
"""
import sys, shutil, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg

TRIAL_DIR = Path(__file__).parent.parent.parent / "trial" / "data"
DEST_DIR  = cfg.data_dir

COPY_MAP = {
    # trial source (relative to TRIAL_DIR)  → dest (relative to DEST_DIR)
    "nse":                                     "nse",
    "bse":                                     "bse",
    "news":                                    "news",
    "labeled_events.parquet":                  "labeled_events.parquet",
    "universe.parquet":                        "universe.parquet",
    "backtest_results":                        "backtest_results",
}

p = argparse.ArgumentParser(description="Migrate trial data to AlphaSense/data/")
p.add_argument("--dry-run", action="store_true", help="Print what would be copied without doing it")
args = p.parse_args()

if not TRIAL_DIR.exists():
    print(f"Trial data dir not found: {TRIAL_DIR}")
    sys.exit(1)

total = 0
for src_rel, dst_rel in COPY_MAP.items():
    src = TRIAL_DIR / src_rel
    dst = DEST_DIR / dst_rel
    if not src.exists():
        print(f"  SKIP (not found): {src}")
        continue

    if src.is_dir():
        files = list(src.rglob("*"))
        size_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
        print(f"  {'WOULD COPY' if args.dry_run else 'COPY'} dir  {src_rel:40s} → {dst_rel}  ({len(files)} files, {size_mb:.1f} MB)")
        if not args.dry_run:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        total += len(files)
    else:
        size_mb = src.stat().st_size / 1e6
        print(f"  {'WOULD COPY' if args.dry_run else 'COPY'} file {src_rel:40s} → {dst_rel}  ({size_mb:.1f} MB)")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        total += 1

print(f"\n{'DRY RUN — nothing copied.' if args.dry_run else f'Done. {total} items migrated to {DEST_DIR}'}")
