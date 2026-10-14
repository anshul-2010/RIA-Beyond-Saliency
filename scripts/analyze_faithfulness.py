"""Deletion faithfulness on held-out operators (T20).

    py -3 scripts/analyze_faithfulness.py

Question: does a method's importance score for a region predict what actually happens to
the model when that region is deleted? The target is the observed change in the margin
(true vs opposite relation) when the subject (or object) region is deleted with operator X:
    actual_X(region) = m - m(without region)  =  local region x phrase term under X.
Predictors:
  RIA (first order)   the same quantity measured with a DIFFERENT operator Y != X, so the
                      prediction and the target never share a removal
  Grad-CAM (signed), Grad-CAM, Chefer   the region's sum for the same true-vs-opposite target
Spearman correlation across images, per region, averaged over target operators.

The full-image margin m appears in every deletion effect and is identical across operators,
so RIA's held-out prediction and the target share it. All methods are therefore scored by
PARTIAL Spearman correlation controlling for m (ranks residualised on m's rank), which
removes that shared term for everyone.
Joint deletion: RIA's G3 under Y predicts the joint (both-objects) effect under X; first-
order methods can only offer the sum of the two region scores.
"""

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.runlog import REPO_ROOT, read_results  # noqa: E402

RUNS = {"CLIP": "e2_whatsup_clip", "SigLIP": "e2_whatsup_siglip", "Qwen": "e2_whatsup_qwen"}
VALID = {"Qwen": ("blur", "mean")}


def latest(suffix):
    found = sorted((REPO_ROOT / "results").glob(f"*_{suffix}"))
    return found[-1] if found else None


def row(rec, op):
    return next(x for x in rec["ria"] if x["a"] == "subj" and x["b"] == "obj" and x["sub"] == rec["opposite"]
                and x["op"] == op)


MARGIN: list = []          # full-image margin per record, set per model in main()


def rho(a, b):
    """Partial Spearman correlation of a and b controlling for the full-image margin."""
    from scipy.stats import rankdata
    ra, rb, rm = rankdata(a), rankdata(b), rankdata(MARGIN)
    res = lambda v: v - np.polyval(np.polyfit(rm, v, 1), rm)
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def main():
    print("Partial Spearman rho with the observed deletion effect (controlling for the full-image margin):\n")
    print("| model | region | RIA, held-out operator | Grad-CAM signed | Grad-CAM | Chefer |")
    print("|---|---|---|---|---|---|")
    joint = {}
    for name, suffix in RUNS.items():
        run = latest(suffix)
        if run is None:
            continue
        recs = read_results(run)
        MARGIN[:] = [r["margins"][r["opposite"]] for r in recs]
        ops = VALID.get(name) or sorted({x["op"] for x in recs[0]["ria"]})
        for region in ("subj", "obj"):
            actual = {o: [row(r, o)["pair_local"][f"{region}|rel"] for r in recs] for o in ops}
            ria = np.mean([rho(actual[y], actual[x]) for x, y in permutations(ops, 2)])
            cells = [f"{ria:.2f}"]
            for m in ("gradcam_signed", "gradcam", "chefer"):
                if m in recs[0]:
                    pred = [r[m][r["opposite"]][region]["sum"] for r in recs]
                    cells.append(f"{np.mean([rho(pred, actual[x]) for x in ops]):.2f}")
                else:
                    cells.append("–")
            print(f"| {name} | {region} | " + " | ".join(cells) + " |")
        # joint deletion: effect of removing both objects on the margin = m - m(no subj, no obj)
        def both(r, o):
            x = row(r, o)
            return x["pair_local"]["subj|rel"] + x["pair_local"]["obj|rel"] - x["G3"]
        actual_both = {o: [both(r, o) for r in recs] for o in ops}
        g3 = np.mean([rho([row(r, y)["G3"] for r in recs], actual_both[x]) for x, y in permutations(ops, 2)])
        sum_pred = np.mean([rho([row(r, y)["pair_local"]["subj|rel"] + row(r, y)["pair_local"]["obj|rel"] for r in recs],
                                actual_both[x]) for x, y in permutations(ops, 2)])
        joint[name] = (g3, sum_pred)
    print("\nJoint deletion (both objects) predicted from a held-out operator:")
    print("| model | RIA first-order sum | RIA G3 alone |\n|---|---|---|")
    for name, (g3, s) in joint.items():
        print(f"| {name} | {s:.2f} | {g3:.2f} |")


if __name__ == "__main__":
    main()
