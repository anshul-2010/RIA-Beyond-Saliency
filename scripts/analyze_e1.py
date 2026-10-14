"""Analyse E1 (Flickr30k Entities phrase localisation) runs, with baselines (T22).

    py -3 scripts/analyze_e1.py

Per method: pointing game (15 px tolerance and strict) and energy in box, plus two
baselines every pointing-game result must be read against: always pointing at the image
centre, and a uniformly random point. Flickr30k objects cluster near the centre, so the
centre baseline is high (~0.67).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e1_flickr as E  # noqa: E402
from ria.clip_scorer import box_to_frame  # noqa: E402
from ria.datasets import flickr30k_entities as F  # noqa: E402
from ria.runlog import REPO_ROOT, read_results  # noqa: E402


class Grid:
    size, patch = 224, 16
    grid = 14


def baselines(recs, phrases, tol=15, draws=20, seed=0):
    rng = np.random.default_rng(seed)
    g = Grid
    centres = [(r * g.patch + g.patch // 2, c * g.patch + g.patch // 2) for r in range(g.grid) for c in range(g.grid)]
    mid = (g.grid // 2) * g.grid + g.grid // 2
    centre, rand = [], []
    for r in recs:
        p = phrases[r["key"]]
        mask = E.dilate_union(E.box_mask([box_to_frame(b, p.size, g.size) for b in p.boxes], g.size), tol)
        hits = np.array([mask[y, x] for y, x in centres])
        centre.append(hits[mid])
        rand.append(hits[rng.integers(0, len(hits), draws)].mean())
    return float(np.mean(centre)), float(np.mean(rand))


def main():
    phrases = {p.key: p for p in F.load("test")}
    for m in ("clip", "siglip"):
        found = sorted((REPO_ROOT / "results").glob(f"*_e1_{m}"))
        if not found:
            continue
        recs = read_results(found[-1])
        c, rnd = baselines(recs, phrases)
        print(f"\n### {m.upper()} ({len(recs)} phrases)\n")
        print("| method | pointing (15 px) | pointing (strict) | energy in box |\n|---|---|---|---|")
        for meth in recs[0]["metrics"]:
            vals = [r["metrics"][meth] for r in recs]
            print(f"| {meth} | {np.mean([v['hit'] for v in vals]):.3f} | {np.mean([v['hit_strict'] for v in vals]):.3f} "
                  f"| {np.mean([v['energy'] for v in vals]):.3f} |")
        share = np.mean([r["metrics"][next(iter(r["metrics"]))]["box_share"] for r in recs])
        print(f"| centre-point baseline | {c:.3f} | | |\n| random-point baseline | {rnd:.3f} | | {share:.3f} (box share) |")


if __name__ == "__main__":
    main()
