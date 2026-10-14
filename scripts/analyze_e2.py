"""Analyse E2 records from scripts/e2_whatsup.py (T20).

    py -3 scripts/analyze_e2.py results/<stamp>_e2_whatsup [more run dirs ...] [--min-validity 0.25]

Every image asks: does the evidence favour the true relation over its opposite
(on/under, left/right, in front/behind)? G3 below is G3(subj, obj, rel) computed
with the opposite relation as the substitute.

  validity   a removal operator counts only if it actually removes the objects:
             median |score change| when removing subj or obj must be at least
             `min_validity` x that of the strongest operator. (Qwen's visual-token
             replacement fails this: its vision encoder has already spread object
             information to other tokens.)
  relational G3 has the same sign under every valid operator; "true" = positive
             (favours the true relation), "wrong" = negative. Symmetric true/wrong
             rates mean noise; asymmetry means directional relational evidence.
  floor      the older criterion, |mean| > k * sd over valid operators, for reference.
  pairs      rates split by whether the model is right on BOTH images of the pair.
  word bias  per opposite pair r / r': P(true | r) + P(true | r') over relational
             images; near 2 = image-driven, near 1 = favours one word whatever the scene.

  types      for images with sign-agreeing G3, the four margins (true vs opposite) with
             both objects, without subj, without obj, without both are rebuilt from the
             stored terms and classified: AND (synergy, G3 > 0: both objects needed),
             OR (redundancy: margin survives removing either object, not both; G3 < 0
             with a correct answer), model wrong, or other. A negative G3 is therefore
             not "evidence for the opposite relation" by itself.
  counterfactual  held-out test that avoids that circularity: among images the model
             gets right, does correct-direction G3 predict that the model is ALSO
             right on the paired image (same objects, opposite relation)? The paired
             image's margin is not contained in this image's G3. One-sided Fisher test.
             Exploratory (the stratification was chosen after seeing CLIP/SigLIP);
             confirm on a second dataset.

Removal-based scores contain the full-image margin algebraically, so "does the
evidence predict the model's own margin" would be circular and is not reported.
Writes summary.json and summary.md next to the records.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.runlog import read_results  # noqa: E402

PAIRS = (("behind", "in front of"), ("to the right of", "to the left of"), ("on", "under"),
         # VSR relation pairs (scripts/e2_vsr.py)
         ("on top of", "beneath"), ("at the right side of", "at the left side of"), ("above", "below"),
         ("left of", "right of"), ("under", "over"), ("facing", "facing away from"),
         ("far away from", "close to"), ("far from", "near"))


def finite(rec) -> bool:
    """False if a margin or term is NaN/inf (rare fp16 overflow in the generative VLM)."""
    import math
    ok = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
    return all(ok(v) for v in rec["margins"].values()) and all(
        ok(x["G3"]) and all(ok(v) for v in x["effect"].values()) for x in rec["ria"])


def opposite_rows(rec, a="subj", b="obj"):
    return [r for r in rec["ria"] if r["a"] == a and r["b"] == b and r["sub"] == rec["opposite"]]


def operator_validity(recs, min_validity):
    """Median |score change| from removing each object, per operator; which operators pass."""
    ops = sorted({r["op"] for r in recs[0]["ria"]})
    eff = {}
    for o in ops:
        vals = [abs(r["effect"][n]) for rec in recs for r in opposite_rows(rec) if r["op"] == o for n in ("subj", "obj")]
        eff[o] = float(np.median(vals))
    top = max(eff.values())
    valid = [o for o in ops if eff[o] >= min_validity * top]
    return eff, valid


def evidence(rec, ops, k):
    rows = {r["op"]: r for r in opposite_rows(rec) if r["op"] in ops}
    g = np.array([rows[o]["G3"] for o in ops])
    out = {"G3": float(g.mean()),
           "true": bool((g > 0).all()), "wrong": bool((g < 0).all()),
           "floor": bool(len(g) > 1 and abs(g.mean()) > k * g.std(ddof=1))}
    d_ap = np.mean([rows[o]["pair"]["subj|rel"] for o in ops])
    d_bp = np.mean([rows[o]["pair"]["obj|rel"] for o in ops])
    denom = abs(d_ap) + abs(d_bp) + abs(out["G3"])
    out["rho"] = abs(out["G3"]) / denom if denom > 0 else 0.0
    ctx = []
    for a in ("subj", "obj"):
        v = np.array([r["G3"] for r in opposite_rows(rec, a, "context") if r["op"] in ops])
        ctx.append((v > 0).all() or (v < 0).all())
    out["ctx_relational"] = bool(any(ctx))
    return out


def summarise(recs, ops, k):
    by = {(r["group"], r["relation"]): r for r in recs}
    rows = []
    for r in recs:
        e = evidence(r, ops, k)
        mate = by.get((r["group"], r["opposite"]))
        right = r["margins"][r["opposite"]] > 0
        both = mate is not None and right and mate["margins"][mate["opposite"]] > 0
        e.update(subset=r["subset"], relation=r["relation"], flagged=bool(r["flags"]), prefers_true=right,
                 pair="right on both" if both else ("right here only" if right else "wrong here"))
        rows.append(e)

    def block(sel):
        if not sel:
            return {}
        rel = [x for x in sel if x["true"] or x["wrong"]]
        return {"n": len(sel), "model_acc": float(np.mean([x["prefers_true"] for x in sel])),
                "true": float(np.mean([x["true"] for x in sel])), "wrong": float(np.mean([x["wrong"] for x in sel])),
                "floor": float(np.mean([x["floor"] for x in sel])),
                "p_true_given_relational": float(np.mean([x["true"] for x in rel])) if rel else None,
                "rho_median": float(np.median([x["rho"] for x in rel])) if rel else None,
                "ctx_relational": float(np.mean([x["ctx_relational"] for x in sel]))}

    out = {"all": block(rows), "unflagged": block([x for x in rows if not x["flagged"]])}
    for pc in ("right on both", "right here only", "wrong here"):
        out[f"model {pc}"] = block([x for x in rows if x["pair"] == pc])
    for s in sorted({x["subset"] for x in rows}):
        out[f"subset {s}"] = block([x for x in rows if x["subset"] == s])
    for rel in sorted({x["relation"] for x in rows}):
        out[f"rel {rel}"] = block([x for x in rows if x["relation"] == rel])
    return out


def margins(row):
    """Margins (true vs opposite) with both, -subj, -obj, -both, rebuilt from one triple record."""
    m = row["effect"]["rel"]
    ma, mb = m - row["pair_local"]["subj|rel"], m - row["pair_local"]["obj|rel"]
    return m, ma, mb, row["G3"] - m + ma + mb


def interaction_types(recs, ops):
    from collections import Counter
    counts = Counter()
    for rec in recs:
        rows = [r for r in opposite_rows(rec) if r["op"] in ops]
        g = np.array([r["G3"] for r in rows])
        if not ((g > 0).all() or (g < 0).all()):
            continue
        m, ma, mb, mab = np.mean([margins(r) for r in rows], axis=0)
        if g.mean() > 0:
            counts["AND (synergy)"] += 1
        elif m > 0 and ma > 0 and mb > 0 and mab < min(ma, mb):
            counts["OR (redundancy)"] += 1
        elif m <= 0:
            counts["model wrong"] += 1
        else:
            counts["other negative"] += 1
    return dict(counts)


def counterfactual(recs, ops, k):
    """2x2 over images the model gets right: G3 true/wrong vs paired image right/wrong."""
    from scipy.stats import fisher_exact
    by = {(r["group"], r["relation"]): r for r in recs}
    t = np.zeros((2, 2), int)
    for r in recs:
        mate = by.get((r["group"], r["opposite"]))
        if mate is None or not r["margins"][r["opposite"]] > 0:
            continue
        e = evidence(r, ops, k)
        if e["true"] or e["wrong"]:
            t[0 if e["true"] else 1, 0 if mate["margins"][mate["opposite"]] > 0 else 1] += 1
    if t.sum(axis=1).min() == 0:
        return {"table": t.tolist(), "odds_ratio": None, "p_one_sided": None}
    odds, p = fisher_exact(t, alternative="greater")
    return {"table": t.tolist(), "odds_ratio": float(odds), "p_one_sided": float(p)}


def word_bias(summary):
    out = {}
    for r1, r2 in PAIRS:
        a, b = summary.get(f"rel {r1}"), summary.get(f"rel {r2}")
        if a and b and a["p_true_given_relational"] is not None and b["p_true_given_relational"] is not None:
            out[f"{r1} / {r2}"] = (a["p_true_given_relational"], b["p_true_given_relational"])
    return out


def to_markdown(model, eff, valid, summary, cf, types=None):
    def f(v):
        return "–" if v is None else f"{v:.2f}"
    lines = [f"### {model}", "", "Operator validity (median |score change| removing one object):", "",
             "| operator | effect | used |", "|---|---|---|"]
    lines += [f"| {o} | {e:.4g} | {'yes' if o in valid else 'no'} |" for o, e in eff.items()]
    lines += ["", "| slice | n | model prefers true | G3 true direction | G3 wrong direction | P(true given relational) | "
              "±k·sd floor | ρ median | context relational |", "|---|---|---|---|---|---|---|---|---|"]
    for name, b in summary.items():
        if b:
            lines.append(f"| {name} | {b['n']} | {f(b['model_acc'])} | {f(b['true'])} | {f(b['wrong'])} | "
                         f"{f(b['p_true_given_relational'])} | {f(b['floor'])} | {f(b['rho_median'])} | {f(b['ctx_relational'])} |")
    lines += ["", "Word-bias test (sum near 1 = word-driven, near 2 = image-driven):", "",
              "| opposite pair | P(true given relational) each | sum |", "|---|---|---|"]
    for pair, (x, y) in word_bias(summary).items():
        lines.append(f"| {pair} | {x:.2f} + {y:.2f} | {x + y:.2f} |")
    if types:
        n = sum(types.values())
        lines += ["", f"Interaction types among {n} sign-agreeing images: " +
                  ", ".join(f"{k} {v / n:.0%}" for k, v in sorted(types.items(), key=lambda kv: -kv[1]))]
    t = cf["table"]
    lines += ["", "Counterfactual test (images the model gets right; is the paired image also right?):", "",
              "| G3 on this image | paired right | paired wrong |", "|---|---|---|",
              f"| true direction | {t[0][0]} | {t[0][1]} |", f"| wrong direction | {t[1][0]} | {t[1][1]} |", "",
              f"odds ratio {f(cf['odds_ratio'])}, one-sided Fisher p {f(cf['p_one_sided'])} (exploratory)"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--min-validity", type=float, default=0.25)
    ap.add_argument("--operators", nargs="*", default=None, help="override the validity check")
    ap.add_argument("--k", type=float, default=2.0)
    args = ap.parse_args()
    for run in args.runs:
        recs = read_results(run)
        n_all = len(recs)
        recs = [r for r in recs if finite(r)]
        if len(recs) < n_all:
            print(f"dropped {n_all - len(recs)} records with non-finite scores")
        model = json.loads((run / "config.json").read_text(encoding="utf-8"))["args"]["model"]
        eff, valid = operator_validity(recs, args.min_validity)
        ops = args.operators or valid
        summary = summarise(recs, ops, args.k)
        cf = counterfactual(recs, ops, args.k)
        types = interaction_types(recs, ops)
        (run / "summary.json").write_text(json.dumps({"operator_effect": eff, "operators": ops, "counterfactual": cf, "interaction_types": types,
                                                      **summary}, indent=1), encoding="utf-8")
        md = to_markdown(model, eff, ops, summary, cf, types)
        (run / "summary.md").write_text(md, encoding="utf-8")
        print(md + "\n")


if __name__ == "__main__":
    main()
