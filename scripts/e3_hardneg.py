"""E3: does relational attribution predict failures on compositional hard negatives? (T21)

    py -3 scripts/e3_hardneg.py --config configs/e3_clip.json [--limit 20]

Items: SugarCrepe replace_rel (relation span of 1-4 words that occurs once, both nouns
found by ria.text.relation_args) and ARO VG-Relation (stratified: at most
`aro_per_relation` items per relation).

Non-circular design. The target is whether the model prefers the true caption over the
benchmark's foil. RIA is computed from the TRUE caption only, with the relation replaced
by neutral substitutes ("near", "and", "beside") and never by the foil's wording, so the
foil's score never enters the features. analyze_e3.py then asks whether RIA adds
predictive value beyond the true caption's own score s(I, T), which RIA does share.

Per item: target margin, s(I, T), RIA terms for (subj, obj), (subj, context),
(obj, context) under every operator x substitute, and phrase-conditioned first-order
baselines (encoders only). Missing OWLv2 boxes are detected first and cached in
data/boxes/e3_owlv2.json.
"""

import argparse
import gc
import shutil
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria import baselines as B  # noqa: E402
from ria.datasets.common import item_input  # noqa: E402
from ria.datasets.compositional import load_aro_relation, load_sugarcrepe  # noqa: E402
from ria.detect import BoxCache, OwlDetector, detect_all  # noqa: E402
from ria.interactions import memoize, powerset, relational_triple  # noqa: E402
from ria.players import PhrasePlayer  # noqa: E402
from ria.runlog import REPO_ROOT, Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402

TRIPLES = (("subj", "obj"), ("subj", "context"), ("obj", "context"))
NEUTRAL = ("near", "and", "beside")
# T07 lexicon sensitivity: contrasting relation words (still never the foil's wording)
CONTRAST = ("under", "behind", "above", "in front of", "holding", "riding")
BOX_CACHE = REPO_ROOT / "data" / "boxes" / "e3_owlv2.json"


def neutral_substitutes(span: str, foil_span: str | None, n: int = 2, lexicon=NEUTRAL) -> list[str]:
    taken = {span.lower(), (foil_span or "").lower()}
    return [s for s in lexicon if s not in taken][:n]


def select(cfg):
    from ria.text import relation_args
    items = []
    if cfg["sugarcrepe"]:
        for x in load_sugarcrepe("replace_rel"):
            if not 1 <= len(x.span.split()) <= 4:
                continue
            if len(re.findall(r"(?<!\w)" + re.escape(x.span) + r"(?!\w)", x.caption)) != 1:
                continue
            args = relation_args(x.caption, x.span)
            if args:
                x.subj, x.obj = args
                items.append(x)
    if cfg["aro_per_relation"]:
        by_rel = defaultdict(list)
        for x in load_aro_relation():
            by_rel[x.relation].append(x)
        rng = random.Random(cfg["seed"])
        for rel in sorted(by_rel):
            xs = by_rel[rel]
            rng.shuffle(xs)
            for x in xs[: cfg["aro_per_relation"]]:
                x.span, x.foil_span = rel, None           # the relation word is the phrase player
                items.append(x)
    if cfg["limit"]:
        items = items[: cfg["limit"]]
    return items


def ensure_boxes(items, detector_name):
    cache = BoxCache(BOX_CACHE)
    # re-detect when the cached boxes were found for different nouns (e.g. after a parser change)
    todo = [x for x in items if x.key not in cache or {x.subj, x.obj} - set(cache[x.key]["boxes"])]
    for x in todo:
        cache.data.pop(x.key, None)
    if todo:
        print(f"detecting boxes for {len(todo)} items", flush=True)
        det = OwlDetector(detector_name)
        detect_all(det, [(x.key, x.image, [x.subj, x.obj]) for x in todo], cache)
        del det
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return cache


def ria_records(scorer, it, operators):
    out = []
    players = [r.name for r in it.regions] + ["rel"]
    for op in operators:
        sc = scorer.with_operator(op)
        for sub in it.phrase.substitutes:
            game = memoize(sc.game(it.image, it.caption, it.regions, [PhrasePlayer("rel", it.phrase.span, [sub])]))
            game(powerset(players))
            for a, b in TRIPLES:
                r = relational_triple(game, a, b, "rel")
                out.append({"op": op, "sub": sub, "a": a, "b": b, "score": r.score, "effect": r.effect,
                            "pair": r.pair, "pair_local": r.pair_local, "G3": r.triple})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--sugarcrepe", action="store_true", default=True)
    ap.add_argument("--no-sugarcrepe", dest="sugarcrepe", action="store_false")
    ap.add_argument("--aro-per-relation", type=int, default=40)
    ap.add_argument("--operators", nargs="+", default=["drop", "blur", "mean"])
    ap.add_argument("--no-first-order", action="store_true")
    ap.add_argument("--substitutes", choices=["neutral", "contrast"], default="neutral",
                    help="contrast = T07 lexicon-sensitivity run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run-dir", default=None, help="fixed output dir; resume there, skipping finished items")
    cfg = load_config(ap)

    items = select(cfg)
    print(f"{len(items)} items ({sum(x.dataset.startswith('sugarcrepe') for x in items)} SugarCrepe, "
          f"{sum(x.dataset.startswith('aro') for x in items)} ARO)", flush=True)
    boxes = ensure_boxes(items, cfg["detector"])
    scorer = make_scorer(cfg["model"])

    with Run(cfg["name"] or "e3_hardneg", cfg, run_dir=cfg["run_dir"]) as log:
        done = log.done_keys()
        if done:
            print(f"resuming: {len(done)} items already done", flush=True)
        if BOX_CACHE.exists():
            shutil.copy2(BOX_CACHE, log.dir / BOX_CACHE.name)      # results/ is saved live on Kaggle
        t_start = time.perf_counter()
        for n, x in enumerate(items, 1):
            if x.key in done or x.key not in boxes or {x.subj, x.obj} - set(boxes[x.key]["boxes"]):
                continue
            rec_box = boxes[x.key]
            try:
                img = x.image()
            except FileNotFoundError:
                continue
            subs = neutral_substitutes(x.span, x.foil_span,
                                       lexicon=CONTRAST if cfg["substitutes"] == "contrast" else NEUTRAL)
            it = item_input(img, x.caption, x.span, subs, rec_box["boxes"][x.subj]["box"],
                            rec_box["boxes"][x.obj]["box"], scorer.size)
            rec = {"key": x.key, "dataset": x.dataset, "relation": x.relation or x.span, "caption": x.caption,
                   "foil": x.foil, "span": x.span, "substitutes": subs, "subj": x.subj, "obj": x.obj,
                   "flags": rec_box["flags"], "seconds": {}}
            t = time.perf_counter()
            rec["margin"] = scorer.margin(it.image, x.caption, [x.foil])          # target
            rec["s_true"] = scorer.game(it.image, x.caption, [], [])([frozenset()])[frozenset()]
            rec["ria"] = ria_records(scorer, it, cfg["operators"])
            rec["seconds"]["ria"] = time.perf_counter() - t
            if not cfg["no_first_order"]:
                t = time.perf_counter()
                sets = B.region_patch_sets(scorer, it.regions)
                for name, signed in (("gradcam", False), ("gradcam_signed", True)):
                    rec[name] = {s: B.aggregate(B.gradcam(scorer, it.image, x.caption,
                                                          PhrasePlayer("rel", x.span, [s]), signed=signed), sets)
                                 for s in subs}
                rec["chefer"] = {s: B.aggregate(B.chefer(scorer, it.image, x.caption, PhrasePlayer("rel", x.span, [s])),
                                                sets) for s in subs}
                rec["seconds"]["first_order"] = time.perf_counter() - t
            log.log(rec)
            if n % 50 == 0 or n == len(items):
                el = time.perf_counter() - t_start
                print(f"  {n}/{len(items)}  {el / n:.1f}s/item  eta {el / n * (len(items) - n) / 60:.0f} min", flush=True)
        print(f"logged {log.n} records to {log.dir}", flush=True)


if __name__ == "__main__":
    main()
