"""E2 on What's Up: relational attribution under relation changes with objects fixed (T20).

    py -3 scripts/e2_whatsup.py --config configs/e2_whatsup_clip.json [--limit 4] [--no-fixlip]

One JSONL record per image with everything the analysis needs:
  margins      s(I, T) - s(I, T_foil) for each foil relation (model behaviour)
  ria          per (operator, substitute, triple): removal effects, pair dividends and
               local pairs, G3; triples (subj,obj), (subj,context), (obj,context)
  gradcam / gradcam_signed / chefer
               per substitute: region scores for the phrase-contrastive target
  gradcam_caption  region scores for the whole-caption target
  fixlip       region x phrase sums and the subj x obj image-image sum (true caption)
The analysis (opposite-relation sign tests, relational share, faithfulness) runs
offline on these records.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria import baselines as B  # noqa: E402
from ria.datasets import whatsup  # noqa: E402
from ria.interactions import memoize, powerset, relational_triple  # noqa: E402
from ria.players import PhrasePlayer  # noqa: E402
from ria.runlog import Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402

TRIPLES = (("subj", "obj"), ("subj", "context"), ("obj", "context"))


def ria_records(scorer, mi, operators):
    out = []
    players = [r.name for r in mi.regions] + ["rel"]
    for op in operators:
        sc = scorer.with_operator(op)
        for sub in mi.phrase.substitutes:
            game = memoize(sc.game(mi.image, mi.caption, mi.regions, [PhrasePlayer("rel", mi.phrase.span, [sub])]))
            game(powerset(players))                       # one batched evaluation of all 16 subsets
            for a, b in TRIPLES:
                r = relational_triple(game, a, b, "rel")
                out.append({"op": op, "sub": sub, "a": a, "b": b, "score": r.score, "effect": r.effect,
                            "pair": r.pair, "pair_local": r.pair_local, "G3": r.triple})
    return out


def first_order(scorer, mi, sets, rec):
    for name, signed in (("gradcam", False), ("gradcam_signed", True)):
        rec[name] = {sub: B.aggregate(B.gradcam(scorer, mi.image, mi.caption,
                                                PhrasePlayer("rel", mi.phrase.span, [sub]), signed=signed), sets)
                     for sub in mi.phrase.substitutes}
    rec["chefer"] = {sub: B.aggregate(B.chefer(scorer, mi.image, mi.caption,
                                               PhrasePlayer("rel", mi.phrase.span, [sub])), sets)
                     for sub in mi.phrase.substitutes}
    rec["gradcam_caption"] = B.aggregate(B.gradcam(scorer, mi.image, mi.caption, None), sets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--subsets", nargs="+", default=["A", "B"])
    ap.add_argument("--limit", type=int, default=0, help="first N examples per subset (0 = all)")
    ap.add_argument("--operators", nargs="+", default=["drop", "blur", "mean"])
    ap.add_argument("--no-fixlip", action="store_true")
    ap.add_argument("--only-fixlip", action="store_true", help="skip RIA and first-order baselines")
    ap.add_argument("--no-first-order", action="store_true", help="skip gradient baselines (generative VLMs)")
    ap.add_argument("--margins-only", action="store_true", help="only the model's margins (accuracy check)")
    ap.add_argument("--groups", type=int, default=0,
                    help="sample this many whole minimal-pair groups per subset (0 = all images)")
    ap.add_argument("--fixlip-budget-image", type=int, default=2 ** 11)
    ap.add_argument("--fixlip-budget-text", type=int, default=2 ** 6)
    cfg = load_config(ap)

    scorer = make_scorer(cfg["model"])
    fixlip = None if cfg["no_fixlip"] else B.FixlipBaseline(
        cfg["model"], budget_image=cfg["fixlip_budget_image"], budget_text=cfg["fixlip_budget_text"])
    exs = []
    for s in cfg["subsets"]:
        sub = whatsup.load(s, require_boxes=True)
        if cfg["groups"]:
            groups = sorted(whatsup.group_pairs(sub).items())
            random.Random(cfg["seed"]).shuffle(groups)
            sub = [e for _, g in groups[: cfg["groups"]] for e in g]
        exs += sub[: cfg["limit"] or None]
    print(f"{len(exs)} examples on {scorer.device}, model {cfg['model']}, fixlip {'off' if fixlip is None else 'on'}",
          flush=True)

    with Run(cfg["name"] or "e2_whatsup", cfg) as log:
        t_start = time.perf_counter()
        for n, ex in enumerate(exs, 1):
            mi = whatsup.to_model_input(ex, scorer.size)
            sets = B.region_patch_sets(scorer, mi.regions)
            rec = {"key": ex.key, "subset": ex.subset, "group": "/".join(ex.pair_key), "relation": ex.true.rel,
                   "opposite": whatsup.OPPOSITE[ex.true.rel], "caption": mi.caption, "flags": ex.flags,
                   "box_scores": ex.box_scores, "patches": {k: int(v.sum()) for k, v in sets.items()}, "seconds": {}}

            t = time.perf_counter()
            rec["margins"] = {f.rel: scorer.margin(mi.image, mi.caption, [f.text]) for f in ex.foils}
            rec["seconds"]["margins"] = time.perf_counter() - t
            if not (cfg["only_fixlip"] or cfg["margins_only"]):
                t = time.perf_counter()
                rec["ria"] = ria_records(scorer, mi, cfg["operators"])
                rec["seconds"]["ria"] = time.perf_counter() - t
                if not cfg["no_first_order"]:
                    t = time.perf_counter()
                    first_order(scorer, mi, sets, rec)
                    rec["seconds"]["first_order"] = time.perf_counter() - t

            if fixlip is not None:
                t = time.perf_counter()
                rec["fixlip"] = fixlip.region_scores(mi.image, mi.caption, mi.phrase, sets)
                rec["seconds"]["fixlip"] = time.perf_counter() - t

            log.log(rec)
            if n % 20 == 0 or n == len(exs):
                el = time.perf_counter() - t_start
                print(f"  {n}/{len(exs)}  {el / n:.1f}s/example  eta {el / n * (len(exs) - n) / 60:.0f} min", flush=True)
        print(f"logged {log.n} records to {log.dir}", flush=True)


if __name__ == "__main__":
    main()
