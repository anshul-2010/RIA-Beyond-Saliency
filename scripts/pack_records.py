"""Bundle the per-image records behind every number in the paper into one archive.

    python scripts/pack_records.py                 # writes ria_records.zip
    python scripts/pack_records.py --out x.zip

Takes the latest run of each name the analysis scripts read (results/<stamp>_<name>/) and
stores its config.json, results.jsonl and any summary files. Unzip the archive in the
repository root to get results/ back, then run `bash reproduce.sh analyse`.
Winoground results are never included (its licence forbids redistribution).
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    [f"e2_whatsup_{m}" for m in ("clip", "siglip", "qwen")]
    + [f"e2_vsr_{m}" for m in ("clip", "siglip", "qwen2.5")]
    + [f"e3_{m}" for m in ("clip", "siglip", "qwen")]
    + [f"e3_{m}_contrast" for m in ("clip", "siglip", "qwen")]
    + [f"t05_ctx_{m}" for m in ("clip", "siglip", "qwen2.5")]
    + ["e1_clip", "e1_siglip", "e2_fixlip_clip", "e2_fixlip_siglip",
       "e4_paraphrase_clip", "e5_sanity_clip", "e5_task_conditioning_clip"]
)
KEEP = ("config.json", "results.jsonl", "summary.json", "summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "ria_records.zip")
    args = ap.parse_args()
    missing, n = [], 0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in NAMES:
            runs = sorted((ROOT / "results").glob(f"*_{name}"))
            if not runs:
                missing.append(name)
                continue
            run = runs[-1]
            if "winoground" in run.name:
                continue
            for f in KEEP:
                p = run / f
                if p.exists():
                    z.write(p, Path("results") / run.name / f)
                    n += 1
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({n} files, {size:.1f} MB)")
    if missing:
        print("not found (skipped):", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
