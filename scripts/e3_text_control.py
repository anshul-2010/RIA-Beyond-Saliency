"""E3 control: is relation reliance just text-encoder sensitivity? (T21)

    py -3 scripts/e3_text_control.py results/<stamp>_e3_clip [--model openai/clip-vit-base-patch16]

Relation reliance d_rel = s(I, T) - mean s(I, T_neutral) predicts hard-negative failures.
If it were only the text encoder ignoring the relation word, text-only similarity
cos(T, T_neutral) (or even cos(T, T_foil)) should predict failures as well. This script
compares cross-validated AUROC (all sets include s(I, T)). Dual encoders only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_e3 as A  # noqa: E402
from ria.players import PhrasePlayer  # noqa: E402
from ria.runlog import read_results  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--model", default=None, help="defaults to the run's model")
    args = ap.parse_args()
    import json
    recs = read_results(args.run)
    model = args.model or json.loads((args.run / "config.json").read_text(encoding="utf-8"))["args"]["model"]
    sc = make_scorer(model)
    _, ops = A.valid_operators(recs, 0.25)

    sim_neutral, sim_foil = [], []
    with torch.no_grad():
        for i in range(0, len(recs), 64):
            batch = recs[i:i + 64]
            caps = [r["caption"] for r in batch]
            neut = [PhrasePlayer("rel", r["span"], r["substitutes"]).apply(r["caption"], r["substitutes"][0]) for r in batch]
            e_c, e_n, e_f = sc.encode_text(caps), sc.encode_text(neut), sc.encode_text([r["foil"] for r in batch])
            sim_neutral += (e_c * e_n).sum(-1).tolist()
            sim_foil += (e_c * e_f).sum(-1).tolist()

    feats = [A.features(r, ops) for r in recs]
    y = np.array([r["margin"] > 0 for r in recs], int)
    sets = {
        "relation reliance d_rel": lambda f, i: [f["s_true"], f["d_rel"]],
        "text-only cos(T, T_neutral)": lambda f, i: [f["s_true"], sim_neutral[i]],
        "text-only cos(T, T_foil) [reference]": lambda f, i: [f["s_true"], sim_foil[i]],
        "d_rel + cos(T, T_neutral)": lambda f, i: [f["s_true"], sim_neutral[i], f["d_rel"]],
    }
    print(f"{model}: AUROC for predicting the model's preference (all sets include s(I, T))")
    for name, fn in sets.items():
        m, se = A.cv_auroc(np.array([fn(f, i) for i, f in enumerate(feats)]), y)
        print(f"   {name:38s} {m:.3f} ± {se:.3f}")
    r = np.corrcoef([f["d_rel"] for f in feats], 1 - np.array(sim_neutral))[0, 1]
    print(f"   corr(d_rel, 1 - cos(T, T_neutral)) = {r:.3f}")


if __name__ == "__main__":
    main()
