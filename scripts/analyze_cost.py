"""T27: computational cost per item, from the timings logged with every run.

    py -3 scripts/analyze_cost.py

All timings are wall-clock seconds on a Kaggle T4 GPU unless marked CPU, read from the
`seconds` fields of the run records (median over items). RIA rows include every
operator x substitute x triple configuration the run computed.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.runlog import REPO_ROOT, read_results  # noqa: E402


def latest(suffix):
    found = sorted((REPO_ROOT / "results").glob(f"*_{suffix}"))
    return found[-1] if found else None


def med(recs, key):
    vals = [r["seconds"][key] for r in recs if key in r.get("seconds", {})]
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)


def main():
    rows = []
    for model, e2, e3 in (("CLIP B/16", "e2_whatsup_clip", "e3_clip"), ("SigLIP base", "e2_whatsup_siglip", "e3_siglip"),
                          ("Qwen2.5-VL-3B", "e2_whatsup_qwen", "e3_qwen")):
        for exp, suffix in (("E2 What's Up", e2), ("E3 hard negatives", e3)):
            run = latest(suffix)
            if run is None:
                continue
            recs = read_results(run)
            n_cfg = len(recs[0].get("ria", []))
            for key, label in (("ria", f"RIA ({n_cfg} triple configurations)"), ("first_order", "Grad-CAM x2 + Chefer"),
                               ("margins", "model margins only")):
                s, n = med(recs, key)
                if s is not None:
                    rows.append((model, exp, label, s, n))
    for model, suffix in (("CLIP B/16", "e2_fixlip_clip"), ("SigLIP base", "e2_fixlip_siglip")):
        run = latest(suffix)
        if run:
            s, n = med(read_results(run), "fixlip")
            rows.append((model, "E2 What's Up", "FIxLIP, budget 512x32 (CPU kernel)", s, n))
    print("| model | experiment | method | median s / item | items |\n|---|---|---|---|---|")
    for model, exp, label, s, n in rows:
        print(f"| {model} | {exp} | {label} | {s:.2f} | {n} |")


if __name__ == "__main__":
    main()
