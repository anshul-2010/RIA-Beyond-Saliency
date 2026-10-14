"""Compare RIA with the official FIxLIP on the images FIxLIP could afford (T20).

    py -3 scripts/compare_fixlip.py

Uses the held-out counterfactual test from analyze_e2.py: among images the model gets
right, does a method's evidence predict that the model is also right on the paired
image? For FIxLIP, "evidence" is the sign of (a) the region x phrase interactions summed
over subject and object, (b) the subject x object image-image interaction. FIxLIP costs
~11 min/image on CPU and OOMs on GPU at full budget for ViT-B/16, so only 60-70 images
per model exist; the comparison is underpowered and reported as such.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_e2 import evidence, operator_validity  # noqa: E402
from ria.runlog import REPO_ROOT, read_results  # noqa: E402

RUNS = {"CLIP": ("e2_whatsup_clip", "e2_fixlip_clip"), "SigLIP": ("e2_whatsup_siglip", "e2_fixlip_siglip")}


def latest(suffix):
    found = sorted((REPO_ROOT / "results").glob(f"*_{suffix}"))
    if not found:
        raise FileNotFoundError(f"no results/*_{suffix}")
    return found[-1]


def methods(ops, fx):
    def ria(r):
        e = evidence(r, ops, 2.0)
        return 1 if e["true"] else (-1 if e["wrong"] else 0)

    return {
        "RIA G3 (sign agreement)": ria,
        "FIxLIP region x phrase": lambda r: np.sign(fx[r["key"]]["fixlip"]["subj"]["sum"] + fx[r["key"]]["fixlip"]["obj"]["sum"]),
        "FIxLIP region x region": lambda r: np.sign(fx[r["key"]]["fixlip"]["pair:subj|obj"]["sum"]),
    }


def table(sel, by, f):
    t = np.zeros((2, 2), int)
    for r in sel:
        if not r["margins"][r["opposite"]] > 0:
            continue
        s = f(r)
        if s == 0:
            continue
        mate = by[(r["group"], r["opposite"])]
        t[0 if s > 0 else 1, 0 if mate["margins"][mate["opposite"]] > 0 else 1] += 1
    return t


def fmt(t):
    if t.sum(axis=1).min() == 0:
        return f"+{t[0, 0]}/{t[0].sum()}  -{t[1, 0]}/{t[1].sum()}   (degenerate)"
    odds, p = fisher_exact(t, alternative="greater")
    return f"+{t[0, 0]}/{t[0].sum()}  -{t[1, 0]}/{t[1].sum()}   OR {odds:.2f}  one-sided p {p:.2f}"


def main():
    pooled = {}
    for name, (main_run, fx_run) in RUNS.items():
        recs = read_results(latest(main_run))
        _, ops = operator_validity(recs, 0.25)
        by = {(r["group"], r["relation"]): r for r in recs}
        fx = {r["key"]: r for r in read_results(latest(fx_run))}
        sel = [r for r in recs if r["key"] in fx]
        secs = np.mean([r["seconds"]["fixlip"] for r in fx.values()])
        print(f"\n{name}: {len(sel)} images with FIxLIP ({secs:.0f} s/image)")
        for m, f in methods(ops, fx).items():
            t = table(sel, by, f)
            pooled[m] = pooled.get(m, np.zeros((2, 2), int)) + t
            print(f"   {m:26s} {fmt(t)}")
    print("\npooled over models:")
    for m, t in pooled.items():
        print(f"   {m:26s} {fmt(t)}")


if __name__ == "__main__":
    main()
