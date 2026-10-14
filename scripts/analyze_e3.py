"""Analyse E3 records from scripts/e3_hardneg.py (T21).

    py -3 scripts/analyze_e3.py results/<stamp>_e3_clip [more runs ...]

Target: the model prefers the true caption over the benchmark foil (margin > 0).
Features come only from the true caption (RIA uses neutral substitutes, never the foil),
so they cannot contain the foil's score. They do share s(I, T) with the target, so every
feature set includes s(I, T) and we report the AUROC GAIN of adding each method.

Nested feature sets (cross-validated logistic regression, 5 folds x 10 repeats):
  confidence      s(I, T)
  + first order   removal effects of subj, obj and the relation phrase
  + second order  local region x phrase interactions (subj|rel, obj|rel)
  + RIA           G3(subj, obj, rel): mean, and sign agreement across valid operators
  gradient        confidence + Grad-CAM (signed) and Chefer subj+obj sums, as a separate baseline

Operators are validated as in analyze_e2.py (removal effect >= min_validity x strongest).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.runlog import read_results  # noqa: E402


def finite(rec) -> bool:
    """False if any score or term is NaN/inf (rare fp16 overflow in the generative VLM)."""
    import math
    ok = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
    return ok(rec["margin"]) and ok(rec["s_true"]) and all(
        ok(x["G3"]) and all(ok(v) for v in x["effect"].values()) for x in rec["ria"])


def rows(rec, a="subj", b="obj"):
    return [r for r in rec["ria"] if r["a"] == a and r["b"] == b]


def valid_operators(recs, min_validity):
    ops = sorted({r["op"] for r in recs[0]["ria"]})
    eff = {o: float(np.median([abs(r["effect"][n]) for rec in recs for r in rows(rec) if r["op"] == o
                               for n in ("subj", "obj")])) for o in ops}
    top = max(eff.values())
    return eff, [o for o in ops if eff[o] >= min_validity * top]


def features(rec, ops):
    rs = [r for r in rows(rec) if r["op"] in ops]
    mean = lambda f: float(np.mean([f(r) for r in rs]))
    g3_by_op = [np.mean([r["G3"] for r in rs if r["op"] == o]) for o in ops]
    f = {
        "s_true": rec["s_true"],
        "d_subj": mean(lambda r: r["effect"]["subj"]), "d_obj": mean(lambda r: r["effect"]["obj"]),
        "d_rel": mean(lambda r: r["effect"]["rel"]),
        "i2_subj": mean(lambda r: r["pair_local"]["subj|rel"]), "i2_obj": mean(lambda r: r["pair_local"]["obj|rel"]),
        "g3": mean(lambda r: r["G3"]),
        "g3_agree": float(np.sign(np.sign(g3_by_op).sum()) if (np.all(np.array(g3_by_op) > 0) or np.all(np.array(g3_by_op) < 0)) else 0.0),
    }
    for m in ("gradcam_signed", "chefer"):
        if m in rec:
            f[m] = float(np.mean([v["subj"]["sum"] + v["obj"]["sum"] for v in rec[m].values()]))
    return f


SETS = {
    "confidence": ["s_true"],
    "+ first order": ["s_true", "d_subj", "d_obj", "d_rel"],
    "+ second order": ["s_true", "d_subj", "d_obj", "d_rel", "i2_subj", "i2_obj"],
    "+ RIA (G3)": ["s_true", "d_subj", "d_obj", "d_rel", "i2_subj", "i2_obj", "g3", "g3_agree"],
    "gradient baselines": ["s_true", "gradcam_signed", "chefer"],
}


def cv_auroc(X, y, seed=0):
    if len(set(y)) < 2 or min(np.bincount(y)) < 5:
        return None, None
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores)))


def analyse(recs, ops):
    feats = [features(r, ops) for r in recs]
    y = np.array([r["margin"] > 0 for r in recs], dtype=int)
    out = {"n": len(recs), "accuracy": float(y.mean())}
    for name, cols in SETS.items():
        if not all(c in feats[0] for c in cols):
            continue
        X = np.array([[f[c] for c in cols] for f in feats])
        out[name] = cv_auroc(X, y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--min-validity", type=float, default=0.25)
    args = ap.parse_args()
    for run in args.runs:
        recs = read_results(run)
        dropped = sum(not finite(r) for r in recs)
        recs = [r for r in recs if finite(r)]
        model = json.loads((run / "config.json").read_text(encoding="utf-8"))["args"]["model"]
        eff, ops = valid_operators(recs, args.min_validity)
        slices = {"all": recs, "unflagged": [r for r in recs if not r["flags"]]}
        for ds in sorted({r["dataset"] for r in recs}):
            slices[ds] = [r for r in recs if r["dataset"] == ds]
        print(f"### {model}  (operators used: {', '.join(ops)})\n")
        header = ["slice", "n", "model acc"] + list(SETS)
        print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
        summary = {"operator_effect": eff, "operators": ops, "dropped_nonfinite": dropped}
        for name, sel in slices.items():
            res = analyse(sel, ops)
            summary[name] = res
            cells = [f"{res[k][0]:.3f} ± {res[k][1]:.3f}" if res.get(k) and res[k][0] is not None else "–" for k in SETS]
            print(f"| {name} | {res['n']} | {res['accuracy']:.2f} | " + " | ".join(cells) + " |")
        (run / "summary_e3.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print()


if __name__ == "__main__":
    main()
