"""E4 paraphrase stability (T24): do RIA's conclusions survive meaning-preserving rewording?

    py -3 scripts/e4_paraphrase.py --n 100

What's Up captions ("A cup to the left of a table") are rewritten with templates that keep
the relation phrase verbatim, so the same opposite-relation substitution applies:
    P1  "There is a {subj} {rel} a {obj}."
    P2  "In this picture, a {subj} is {rel} a {obj}."
    P3  "This photo shows a {subj} {rel} a {obj}."
For each caption form, G3(subj, obj, rel) with the opposite relation as substitute under
every operator. Reported per paraphrase vs the original caption:
  rank rho     Spearman correlation of mean G3 across images
  verdict      agreement of the per-image verdict (true / wrong / no sign agreement)
  true, wrong  the directional rates under that wording
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.datasets import whatsup  # noqa: E402
from ria.interactions import relational_triple  # noqa: E402
from ria.players import PhrasePlayer  # noqa: E402
from ria.runlog import Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402

TEMPLATES = {
    "original": None,
    "P1": "There is a {subj} {rel} a {obj}.",
    "P2": "In this picture, a {subj} is {rel} a {obj}.",
    "P3": "This photo shows a {subj} {rel} a {obj}.",
}


def caption(ex, form):
    t = TEMPLATES[form]
    return ex.true.text if t is None else t.format(subj=ex.true.subj, rel=ex.true.rel, obj=ex.true.obj)


def g3s(scorer, mi, text, rel, opposite, ops):
    sub = PhrasePlayer("rel", rel, [opposite])
    return [relational_triple(scorer.with_operator(o).game(mi.image, text, mi.regions, [sub]), "subj", "obj", "rel").triple
            for o in ops]


def verdict(g):
    g = np.asarray(g)
    return "true" if (g > 0).all() else ("wrong" if (g < 0).all() else "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--operators", nargs="+", default=["drop", "blur", "mean"])
    cfg = load_config(ap)
    exs = [e for s in ("A", "B") for e in whatsup.load(s, require_boxes=True)]
    exs = random.Random(cfg["seed"]).sample(exs, cfg["n"])
    sc = make_scorer(cfg["model"], device="cpu")

    res = {f: {} for f in TEMPLATES}
    with Run(cfg["name"] or "e4_paraphrase", cfg) as log:
        for i, ex in enumerate(exs, 1):
            mi = whatsup.to_model_input(ex, sc.size)
            for form in TEMPLATES:
                text = caption(ex, form)
                g = g3s(sc, mi, text, ex.true.rel, whatsup.OPPOSITE[ex.true.rel], cfg["operators"])
                res[form][ex.key] = g
                log.log({"key": ex.key, "form": form, "caption": text, "G3": g})
            if i % 10 == 0:
                print(f"  {i}/{len(exs)}", flush=True)
        keys = sorted(res["original"])
        base = [np.mean(res["original"][k]) for k in keys]
        print("\n| wording | rank rho vs original | verdict agreement | true | wrong |\n|---|---|---|---|---|")
        for form in TEMPLATES:
            vals = [np.mean(res[form][k]) for k in keys]
            agree = np.mean([verdict(res[form][k]) == verdict(res["original"][k]) for k in keys])
            v = [verdict(res[form][k]) for k in keys]
            r = spearmanr(base, vals).statistic
            print(f"| {form} | {r:.2f} | {agree:.2f} | {v.count('true') / len(v):.2f} | {v.count('wrong') / len(v):.2f} |")
            log.log({"summary": True, "form": form, "rank_rho": float(r), "verdict_agreement": float(agree)})


if __name__ == "__main__":
    main()
