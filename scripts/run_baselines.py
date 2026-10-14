"""Run RIA and every baseline on the same What's Up examples (T19).

    py -3 scripts/run_baselines.py --config configs/baselines_whatsup.json [--n 8] [--methods gradcam chefer]

Writes one JSONL record per (example, method) with per-region scores, plus RIA's
robust triple (subj, obj, rel). FIxLIP needs scripts/setup_fixlip.py first.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria import baselines as B  # noqa: E402
from ria.datasets import whatsup  # noqa: E402
from ria.interactions import robust_triple  # noqa: E402
from ria.runlog import Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402

ALL = ["gradcam", "gradcam_signed", "gradcam_caption", "chefer", "interaction2", "fixlip", "ria"]


def select(cfg):
    exs = [e for s in cfg["subsets"] for e in whatsup.load(s, require_boxes=True)]
    if cfg["keys"]:
        by = {e.key: e for e in exs}
        return [by[k] for k in cfg["keys"]]
    if cfg["exclude_flagged"]:
        exs = [e for e in exs if not e.flags]
    groups = list(whatsup.group_pairs(exs).values())          # keep whole minimal-pair groups
    random.Random(cfg["seed"]).shuffle(groups)
    out = []
    for g in groups:
        if cfg["n"] and len(out) >= cfg["n"]:
            break
        out += g
    return out[: cfg["n"]] if cfg["n"] else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--subsets", nargs="+", default=["A", "B"])
    ap.add_argument("--n", type=int, default=8, help="number of examples (0 = all), taken as whole groups")
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--exclude-flagged", action="store_true")
    ap.add_argument("--methods", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--operators", nargs="+", default=["drop", "blur", "mean"])
    ap.add_argument("--fixlip-budget-image", type=int, default=2 ** 11)
    ap.add_argument("--fixlip-budget-text", type=int, default=2 ** 6)
    ap.add_argument("--k", type=float, default=2.0)
    cfg = load_config(ap)

    scorer = make_scorer(cfg["model"])
    fixlip = (B.FixlipBaseline(cfg["model"], budget_image=cfg["fixlip_budget_image"],
                               budget_text=cfg["fixlip_budget_text"]) if "fixlip" in cfg["methods"] else None)
    exs = select(cfg)
    print(f"{len(exs)} examples, methods {cfg['methods']}")

    with Run(cfg["name"] or "baselines_whatsup", cfg) as log:
        for n, ex in enumerate(exs, 1):
            mi = whatsup.to_model_input(ex, scorer.size)
            sets = B.region_patch_sets(scorer, mi.regions)
            base = {"key": ex.key, "group": "/".join(ex.pair_key), "relation": ex.true.rel, "caption": mi.caption,
                    "flags": ex.flags, "patches": {k: int(v.sum()) for k, v in sets.items()}}
            for m in cfg["methods"]:
                t = time.perf_counter()
                if m == "gradcam":
                    scores = B.aggregate(B.gradcam(scorer, mi.image, mi.caption, mi.phrase), sets)
                elif m == "gradcam_signed":
                    scores = B.aggregate(B.gradcam(scorer, mi.image, mi.caption, mi.phrase, signed=True), sets)
                elif m == "gradcam_caption":
                    scores = B.aggregate(B.gradcam(scorer, mi.image, mi.caption, None), sets)
                elif m == "chefer":
                    scores = B.aggregate(B.chefer(scorer, mi.image, mi.caption, mi.phrase), sets)
                elif m == "interaction2":
                    scores = B.interaction2(scorer, mi.image, mi.caption, mi.regions, mi.phrase, cfg["operators"])
                elif m == "fixlip":
                    scores = fixlip.region_scores(mi.image, mi.caption, mi.phrase, sets)
                else:  # ria
                    games = scorer.method_games(mi.image, mi.caption, mi.regions, mi.phrase, cfg["operators"])
                    scores = robust_triple(games, "subj", "obj", mi.phrase.name, k=cfg["k"]).as_row()
                log.log({**base, "method": m, "scores": scores, "seconds": time.perf_counter() - t})
            print(f"  {n}/{len(exs)} {ex.key}", flush=True)
        print(f"logged {log.n} records to {log.dir}")


if __name__ == "__main__":
    main()
