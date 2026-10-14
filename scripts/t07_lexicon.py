"""T07: are the E3 conclusions sensitive to the substitute lexicon?

    py -3 scripts/t07_lexicon.py

Compares each model's neutral-substitute E3 run ("near" / "and" / "beside") with a run
using contrasting relation words (scripts/e3_hardneg.py --substitutes contrast; still
never the foil's wording). Per model, on the items present in both runs:
  - item-level Spearman of d_rel and G3 between the two lexicons;
  - the E3 AUROC table (confidence, + first order, + second order, + RIA) under each.
If the "first-order reliance predicts failures, G3 adds nothing" pattern holds under both
lexicons, the E3 conclusion does not hinge on the neutral word choice.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_e3 import analyse, features, finite, valid_operators  # noqa: E402
from ria.runlog import REPO_ROOT, read_results  # noqa: E402

RUNS = {"CLIP B/16": ("e3_clip", "e3_clip_contrast"), "SigLIP base": ("e3_siglip", "e3_siglip_contrast"),
        "Qwen2.5-VL-3B": ("e3_qwen", "e3_qwen_contrast")}


def records(suffix):
    found = sorted((REPO_ROOT / "results").glob(f"*_{suffix}"))          # latest complete run
    return {r["key"]: r for r in read_results(found[-1]) if finite(r)} if found else {}


def main():
    fmt = lambda v: "–" if v is None or v[0] is None else f"{v[0]:.3f} ± {v[1]:.3f}"
    for model, (neutral, contrast) in RUNS.items():
        a, b = records(neutral), records(contrast)
        keys = sorted(set(a) & set(b))
        if not keys:
            print(f"{model}: no paired runs yet ({len(a)} neutral, {len(b)} contrast records)")
            continue
        ra, rb = [a[k] for k in keys], [b[k] for k in keys]
        ops = sorted(set(valid_operators(ra, 0.25)[1]) & set(valid_operators(rb, 0.25)[1]))
        fa, fb = [features(r, ops) for r in ra], [features(r, ops) for r in rb]
        print(f"### {model}: {len(keys)} paired items, operators {', '.join(ops)}")
        for f in ("d_rel", "g3"):
            print(f"  item-level Spearman {f}: {spearmanr([x[f] for x in fa], [x[f] for x in fb]).statistic:.2f}")
        res = {"neutral": analyse(ra, ops), "contrast": analyse(rb, ops)}
        print("| feature set | neutral | contrast |\n|---|---|---|")
        for name in ("confidence", "+ first order", "+ second order", "+ RIA (G3)"):
            print(f"| {name} | {fmt(res['neutral'].get(name))} | {fmt(res['contrast'].get(name))} |")
        print()


if __name__ == "__main__":
    main()
