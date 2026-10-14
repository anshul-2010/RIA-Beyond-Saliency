"""E4: are RIA's quantities stable across arbitrary method choices? (T24)

    py -3 scripts/analyze_e4.py

For each E2 (What's Up) and E3 (SugarCrepe + ARO) run, Spearman rank correlation across
items between configurations that should agree:
  operators    the same quantity under two removal operators (drop / blur / mean / token)
  substitutes  the same quantity with two different substitutes for the relation
Quantities: E2 -> G3(subj, obj, rel); E3 -> relation reliance d_rel and G3.
Relation reliance removes no image region, so it is operator-independent by construction
(rho = 1 across operators is trivial); its real sensitivity is to the substitute.
Paraphrase stability (the other half of E4) needs new runs and is not covered here.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.runlog import REPO_ROOT, read_results  # noqa: E402

RUNS = {"E2 CLIP": "e2_whatsup_clip", "E2 SigLIP": "e2_whatsup_siglip", "E2 Qwen": "e2_whatsup_qwen",
        "E3 CLIP": "e3_clip", "E3 SigLIP": "e3_siglip"}


def latest(suffix):
    found = sorted((REPO_ROOT / "results").glob(f"*_{suffix}"))
    return found[-1] if found else None


def table(recs, quantity):
    """{(op, sub): {item_key: value}} for the subj x obj triple."""
    out = {}
    for r in recs:
        for x in r["ria"]:
            if x["a"] == "subj" and x["b"] == "obj":
                out.setdefault((x["op"], x["sub"]), {})[r["key"]] = quantity(x)
    return out


def rho(a, b):
    keys = sorted(set(a) & set(b))
    return spearmanr([a[k] for k in keys], [b[k] for k in keys]).statistic if len(keys) > 10 else np.nan


def summarise(name, recs, quantity, qname):
    t = table(recs, quantity)
    ops = sorted({o for o, _ in t})
    subs = sorted({s for _, s in t})
    op_r = [rho(t[(o1, s)], t[(o2, s)]) for s in subs for o1, o2 in combinations(ops, 2)]
    sub_r = [rho(t[(o, s1)], t[(o, s2)]) for o in ops for s1, s2 in combinations(subs, 2)]
    per_op = {f"{o1}-{o2}": np.nanmean([rho(t[(o1, s)], t[(o2, s)]) for s in subs]) for o1, o2 in combinations(ops, 2)}
    print(f"| {name} | {qname} | {np.nanmean(op_r):.2f} | {np.nanmean(sub_r):.2f} | "
          + ", ".join(f"{k} {v:.2f}" for k, v in per_op.items()) + " |")


def main():
    print("| run | quantity | across operators (mean ρ) | across substitutes (mean ρ) | by operator pair |")
    print("|---|---|---|---|---|")
    for name, suffix in RUNS.items():
        run = latest(suffix)
        if run is None:
            continue
        recs = read_results(run)
        summarise(name, recs, lambda x: x["G3"], "G3")
        if name.startswith("E3"):
            summarise(name, recs, lambda x: x["effect"]["rel"], "relation reliance")


if __name__ == "__main__":
    main()
