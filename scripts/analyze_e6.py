"""E6: results by linguistic category of the relation phrase (T26).

    py -3 scripts/analyze_e6.py

E3 items (SugarCrepe replace_rel + ARO VG-Relation) are typed by ria.text.phrase_type
(preposition / verb / attribute / noun / other). Per model and type:
  model acc       how often the model prefers the true caption over the foil
  reliance        median relation reliance d_rel (score drop when the relation is neutralised)
  AUROC           cross-validated AUROC of s(I, T) + d_rel for predicting the model's preference
  G3 agree        share of items whose G3 agrees in sign across valid operators
E2 (What's Up) is all prepositions; its per-relation results are in analyze_e2.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_e3 as A  # noqa: E402
from ria.runlog import REPO_ROOT, read_results  # noqa: E402
from ria.text import phrase_type  # noqa: E402

RUNS = {"CLIP": "e3_clip", "SigLIP": "e3_siglip", "Qwen": "e3_qwen"}


def main():
    print("| model | type | n | model acc | median reliance | AUROC s+d_rel | G3 sign-agreeing |")
    print("|---|---|---|---|---|---|---|")
    types_cache = {}
    for name, suffix in RUNS.items():
        found = sorted((REPO_ROOT / "results").glob(f"*_{suffix}"))
        if not found:
            continue
        recs = [r for r in read_results(found[-1]) if A.finite(r)]
        _, ops = A.valid_operators(recs, 0.25)
        for r in recs:
            if r["key"] not in types_cache:
                types_cache[r["key"]] = phrase_type(r["caption"], r["span"])
        for t in ("preposition", "verb", "attribute", "noun", "other"):
            sel = [r for r in recs if types_cache[r["key"]] == t]
            if len(sel) < 20:
                continue
            feats = [A.features(r, ops) for r in sel]
            y = np.array([r["margin"] > 0 for r in sel], int)
            auc = A.cv_auroc(np.array([[f["s_true"], f["d_rel"]] for f in feats]), y)[0]
            agree = np.mean([f["g3_agree"] != 0 for f in feats])
            print(f"| {name} | {t} | {len(sel)} | {y.mean():.2f} | {np.median([f['d_rel'] for f in feats]):.4f} | "
                  f"{'–' if auc is None else f'{auc:.3f}'} | {agree:.2f} |")


if __name__ == "__main__":
    main()
