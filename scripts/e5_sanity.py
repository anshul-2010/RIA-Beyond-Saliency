"""E5: model-randomisation sanity check (Adebayo et al., 2018) (T25).

    py -3 scripts/e5_sanity.py --n 100

An explanation that reflects the model must change when the model's weights are
destroyed. Vision-encoder blocks of CLIP are re-initialised in a cascade from the top
(last 1, last 4, all blocks + embeddings); for each stage we recompute, on the same
What's Up images, RIA's terms (G3 and the subject removal effect, opposite relation as
substitute, patch dropping) and signed Grad-CAM (subj + obj sum), and report the
Spearman rank correlation with the unrandomised model. Passing = correlation collapses.
"""

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria import baselines as B  # noqa: E402
from ria.datasets import whatsup  # noqa: E402
from ria.interactions import relational_triple  # noqa: E402
from ria.players import PhrasePlayer  # noqa: E402
from ria.runlog import Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402


def randomise(scorer, stage, seed=0):
    """Return a scorer whose vision blocks are re-initialised per `stage`."""
    sc = copy.copy(scorer)
    sc.model = copy.deepcopy(scorer.model)
    layers = sc.model.vision_model.encoder.layers
    k = {"top1": 1, "top4": 4, "all": len(layers)}[stage]
    g = torch.Generator().manual_seed(seed)
    mods = list(layers[-k:])
    if stage == "all":
        mods.append(sc.model.vision_model.embeddings)
    # Re-initialise as at training start: LayerNorm scale 1 / bias 0, other weights N(0, 0.02),
    # biases 0. (Zeroing LayerNorm scales would make every block output a constant.)
    with torch.no_grad():
        for m in mods:
            for sub in m.modules():
                if isinstance(sub, torch.nn.LayerNorm):
                    sub.weight.fill_(1.0)
                    sub.bias.zero_()
                    continue
                for name, p in sub.named_parameters(recurse=False):
                    if name == "bias":
                        p.zero_()
                    else:
                        p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return sc


def explain(scorer, mi):
    sub = PhrasePlayer("rel", mi.phrase.span, [whatsup.OPPOSITE[mi.example.true.rel]])
    r = relational_triple(scorer.with_operator("drop").game(mi.image, mi.caption, mi.regions, [sub]), "subj", "obj", "rel")
    sets = B.region_patch_sets(scorer, mi.regions)
    gc = B.aggregate(B.gradcam(scorer, mi.image, mi.caption, sub, signed=True), sets)
    return {"G3": r.triple, "effect_subj": r.effect["subj"], "gradcam_signed": gc["subj"]["sum"] + gc["obj"]["sum"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--n", type=int, default=100)
    cfg = load_config(ap)

    exs = [e for s in ("A", "B") for e in whatsup.load(s, require_boxes=True)]
    exs = random.Random(cfg["seed"]).sample(exs, cfg["n"])
    base = make_scorer(cfg["model"])
    stages = {"original": base, **{s: randomise(base, s, cfg["seed"]) for s in ("top1", "top4", "all")}}

    vals = {s: {} for s in stages}
    with Run(cfg["name"] or "e5_sanity", cfg) as log:
        for i, ex in enumerate(exs, 1):
            mi = whatsup.to_model_input(ex, base.size)
            for s, sc in stages.items():
                v = explain(sc, mi)
                vals[s][ex.key] = v
                log.log({"key": ex.key, "stage": s, **v})
            if i % 20 == 0:
                print(f"  {i}/{len(exs)}", flush=True)
        keys = sorted(vals["original"])
        print("\nSpearman rho with the unrandomised model (lower = method depends on the weights):")
        print("| quantity | top 1 block | top 4 blocks | all blocks + embeddings |\n|---|---|---|---|")
        def rho(q, s):
            a = [vals["original"][k][q] for k in keys]
            b = [vals[s][k][q] for k in keys]
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                return float("nan")                    # constant: no ranking to compare
            return spearmanr(a, b).statistic

        for q in ("G3", "effect_subj", "gradcam_signed"):
            row = [rho(q, s) for s in ("top1", "top4", "all")]
            print(f"| {q} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
            log.log({"summary": True, "quantity": q, "rho": dict(zip(("top1", "top4", "all"), row))})


if __name__ == "__main__":
    main()
