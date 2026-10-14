"""T05: is G3 robust to the background? Local vs background-averaged (Banzhaf interaction index).

    python scripts/t05_context_average.py run --models openai/clip-vit-base-patch16 ...
    py -3 scripts/t05_context_average.py analyse

The local third-order term G3(subj, obj, rel) keeps every other player (the background
region) present. The Banzhaf interaction index averages over the other players' presence;
with one other player c (the background):
    G3_avg = (G3 | c present + G3 | c removed) / 2 = G3 - Gamma4(subj, obj, rel, c) / 2
The run stores the model's score for all 16 subsets of {subj, obj, context, rel} (opposite
relation as substitute) per valid operator; the analysis compares local and averaged G3:
rank correlation, directional rates (sign agreement across operators), interaction types.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.datasets import whatsup  # noqa: E402
from ria.interactions import memoize, mixed_difference, powerset  # noqa: E402
from ria.players import PhrasePlayer  # noqa: E402
from ria.runlog import REPO_ROOT, Run, load_config, read_results  # noqa: E402

PLAYERS = ["subj", "obj", "context", "rel"]
OPS = {"qwen": ["blur", "mean"]}


def key(s):
    return "|".join(sorted(s)) or "-"


def run(cfg):
    from ria.scorers import make_scorer
    exs = [e for s in ("A", "B") for e in whatsup.load(s, require_boxes=True)][: cfg["limit"] or None]
    for model in cfg["models"]:
        scorer = make_scorer(model)
        ops = OPS.get("qwen" if "qwen" in model.lower() else "", ["drop", "blur", "mean"])
        name = f"t05_ctx_{model.split('/')[-1].split('-')[0].lower()}"
        with Run(name, {**cfg, "model": model, "operators": ops}) as log:
            t0 = time.perf_counter()
            for n, ex in enumerate(exs, 1):
                mi = whatsup.to_model_input(ex, scorer.size)
                opp = whatsup.OPPOSITE[ex.true.rel]
                values = {}
                for op in ops:
                    g = memoize(scorer.with_operator(op).game(mi.image, mi.caption, mi.regions,
                                                              [PhrasePlayer("rel", mi.phrase.span, [opp])]))
                    values[op] = {key(s): v for s, v in g(powerset(PLAYERS)).items()}
                log.log({"key": ex.key, "relation": ex.true.rel, "values": values})
                if n % 100 == 0 or n == len(exs):
                    print(f"  {model}: {n}/{len(exs)}  {(time.perf_counter() - t0) / n:.1f}s/image", flush=True)
        del scorer
        gc.collect()


def terms(vals):
    v = {frozenset(k.split("|")) if k != "-" else frozenset(): x for k, x in vals.items()}
    g3 = mixed_difference(v, ["subj", "obj", "rel"])
    g4 = mixed_difference(v, PLAYERS)
    return g3, g3 - g4 / 2


def analyse():
    from scipy.stats import spearmanr
    print("| model | rank rho local vs averaged | true / wrong (local) | true / wrong (averaged) | verdict agreement |")
    print("|---|---|---|---|---|")
    for m in ("clip", "siglip", "qwen2.5"):
        found = sorted((REPO_ROOT / "results").glob(f"*_t05_ctx_{m}"))
        if not found:
            continue
        recs = read_results(found[-1])
        # an operator must be defined on every subset: SigLIP's patch drop with all three regions
        # removed leaves its attention-pooling head no tokens (NaN); CLIP keeps its CLS token
        ops = [o for o in recs[0]["values"]
               if all(np.isfinite(list(r["values"][o].values())).all() for r in recs)]
        print(f"{m}: operators {', '.join(ops)}", file=sys.stderr)
        loc = np.array([[terms(r["values"][o])[0] for o in ops] for r in recs])
        avg = np.array([[terms(r["values"][o])[1] for o in ops] for r in recs])
        verdict = lambda a: np.where((a > 0).all(1), 1, np.where((a < 0).all(1), -1, 0))
        vl, va = verdict(loc), verdict(avg)
        rho = spearmanr(loc.mean(1), avg.mean(1)).statistic
        print(f"| {m} | {rho:.2f} | {np.mean(vl == 1):.2f} / {np.mean(vl == -1):.2f} | "
              f"{np.mean(va == 1):.2f} / {np.mean(va == -1):.2f} | {np.mean(vl == va):.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyse"])
    ap.add_argument("--models", nargs="+", default=["openai/clip-vit-base-patch16"])
    ap.add_argument("--limit", type=int, default=0)
    cfg = load_config(ap)
    run(cfg) if cfg["mode"] == "run" else analyse()


if __name__ == "__main__":
    main()
