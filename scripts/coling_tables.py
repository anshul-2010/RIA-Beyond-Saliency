"""Appendix tables for the COLING version, generated from the stored runs.

    py -3 scripts/coling_tables.py          # writes paper_coling/tables/*.tex

Each file holds one `tabular` (no float), included with \\input in the paper.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_e3 as E3  # noqa: E402
from analyze_e2 import (PAIRS, counterfactual, finite, interaction_types, operator_validity,  # noqa: E402
                        opposite_rows, summarise)
from ria.runlog import REPO_ROOT, read_results  # noqa: E402

OUT = REPO_ROOT / "paper_coling" / "tables"
MODELS = (("clip", "CLIP"), ("siglip", "SigLIP"), ("qwen", "Qwen"))
CATEGORY = {  # linguistic class of each relation (projective / topological / orientation / distance)
    "to the left of": "proj.", "to the right of": "proj.", "left of": "proj.", "right of": "proj.",
    "at the left side of": "proj.", "at the right side of": "proj.", "above": "proj.", "below": "proj.",
    "over": "proj.", "under": "proj.", "on top of": "proj.", "beneath": "proj.", "behind": "proj.",
    "in front of": "proj.", "on": "top.", "facing": "orient.", "facing away from": "orient.",
    "near": "dist.", "close to": "dist.", "far from": "dist.", "far away from": "dist.",
}


def latest(suffix):
    found = sorted((REPO_ROOT / "results").glob(f"*_{suffix}"))
    return found[-1] if found else None


def e2_run(ds, m):
    suffix = f"e2_whatsup_{m}" if ds == "whatsup" else f"e2_vsr_{'qwen2.5' if m == 'qwen' else m}"
    recs = [r for r in read_results(latest(suffix)) if finite(r)]
    eff, ops = operator_validity(recs, 0.25)
    return recs, eff, ops


def f2(x):
    return "--" if x is None else f"{x:.2f}".replace("0.", ".", 1) if abs(x) < 1 else f"{x:.2f}"


def pct(x):
    return "--" if x is None else f"{100 * x:.0f}"


def write(name, lines):
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


def operator_table(data):
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Data & Model & drop & blur & mean & token \\", r"\midrule"]
    for ds, dname in (("whatsup", "What's Up"), ("vsr", "VSR")):
        for k, (m, mname) in enumerate(MODELS):
            _, eff, ops = data[(ds, m)]
            cells = []
            for o in ("drop", "blur", "mean", "token"):
                if o not in eff:
                    cells.append("--")
                else:
                    v = f"{eff[o]:.3g}"
                    cells.append(v if o in ops else r"\textcolor{bad}{" + v + "}")
            lines.append(f"{dname if k == 0 else ''} & {mname} & " + " & ".join(cells) + r" \\")
        if ds == "whatsup":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("operator_validity", lines)


def relation_table(data, ds, name):
    recs = {m: data[(ds, m)] for m, _ in MODELS}
    sums = {m: summarise(recs[m][0], recs[m][2], 2.0) for m, _ in MODELS}
    rels = sorted({k[4:] for k in sums["qwen"] if k.startswith("rel ")},
                  key=lambda r: (["proj.", "top.", "orient.", "dist."].index(CATEGORY.get(r, "proj.")), r))
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for _, n in MODELS)
    rule = "".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(3))
    lines = [r"\begin{tabular}{@{}llr" + "rrr" * 3 + r"@{}}", r"\toprule",
             r" & & & " + head + r" \\", rule,
             r"Relation & Class & $n$ & " + " & ".join(["acc", "true", "wrong"] * 3) + r" \\", r"\midrule"]
    for rel in rels:
        b = {m: sums[m].get(f"rel {rel}") for m, _ in MODELS}
        cells = []
        for m, _ in MODELS:
            x = b[m]
            cells += [pct(x["model_acc"]), pct(x["true"]), pct(x["wrong"])] if x else ["--"] * 3
        lines.append(f"\\emph{{{rel}}} & {CATEGORY.get(rel, '')} & {b['qwen']['n']} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = []
    for m, _ in MODELS:
        a = sums[m]["all"]
        cells += [pct(a["model_acc"]), pct(a["true"]), pct(a["wrong"])]
    lines.append(rf"all & & {sums['qwen']['all']['n']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, lines)


def slices_table(data):
    lines = [r"\begin{tabular}{@{}ll" + "rrr" * 3 + r"@{}}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for _, n in MODELS) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(3)),
             r"Data & Slice & " + " & ".join([r"$n$", "true", "wrong"] * 3) + r" \\", r"\midrule"]
    for ds, dname in (("whatsup", "What's Up"), ("vsr", "VSR")):
        sums = {m: summarise(data[(ds, m)][0], data[(ds, m)][2], 2.0) for m, _ in MODELS}
        right_label = "right here, wrong on pair" if ds == "whatsup" else "model right"
        for k, (key, label) in enumerate((("all", "all images"), ("unflagged", "confident boxes"),
                                          ("model right on both", "right on both images"),
                                          ("model right here only", right_label),
                                          ("model wrong here", "model wrong"))):
            if not sums["qwen"].get(key):
                continue
            cells = []
            for m, _ in MODELS:
                x = sums[m].get(key)
                cells += [str(x["n"]), pct(x["true"]), pct(x["wrong"])] if x else ["--"] * 3
            lines.append(f"{dname if k == 0 else ''} & {label} & " + " & ".join(cells) + r" \\")
        if ds == "whatsup":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("e2_slices", lines)


def wordbias_table(data):
    lines = [r"\begin{tabular}{@{}lll" + "c" * 3 + r"@{}}", r"\toprule",
             r"Data & Pair $r$ / $r'$ & Class & CLIP & SigLIP & Qwen \\", r"\midrule"]
    for ds, dname in (("whatsup", "What's Up"), ("vsr", "VSR")):
        sums = {m: summarise(data[(ds, m)][0], data[(ds, m)][2], 2.0) for m, _ in MODELS}
        first = True
        for r1, r2 in PAIRS:
            if f"rel {r1}" not in sums["qwen"] or f"rel {r2}" not in sums["qwen"]:
                continue
            cells = []
            for m, _ in MODELS:
                a, b = sums[m].get(f"rel {r1}"), sums[m].get(f"rel {r2}")
                pa, pb = (a or {}).get("p_true_given_relational"), (b or {}).get("p_true_given_relational")
                cells.append("--" if pa is None or pb is None else f"{f2(pa)}+{f2(pb)}$=${pa + pb:.2f}")
            lines.append(f"{dname if first else ''} & \\emph{{{r1}}} / \\emph{{{r2}}} & {CATEGORY.get(r1, '')} & "
                         + " & ".join(cells) + r" \\")
            first = False
        if ds == "whatsup":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("wordbias_full", lines)


def types_table(data):
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Data & Model / relation & $n$ & AND & OR & wrong & other \\", r"\midrule"]
    keys = ("AND (synergy)", "OR (redundancy)", "model wrong", "other negative")
    for ds, dname in (("whatsup", "What's Up"), ("vsr", "VSR")):
        first = True
        for m, mname in MODELS:
            recs, _, ops = data[(ds, m)]
            t = interaction_types(recs, ops)
            n = sum(t.values())
            lines.append(f"{dname if first else ''} & {mname} & {n} & " + " & ".join(pct(t.get(k, 0) / n) for k in keys) + r" \\")
            first = False
        recs, _, ops = data[(ds, "qwen")]
        for rel in sorted({r["relation"] for r in recs}, key=lambda r: (CATEGORY.get(r, ""), r)):
            sub = [r for r in recs if r["relation"] == rel]
            t = interaction_types(sub, ops)
            n = sum(t.values())
            if n:
                lines.append(rf" & \quad Qwen: \emph{{{rel}}} & {n} & " + " & ".join(pct(t.get(k, 0) / n) for k in keys) + r" \\")
        if ds == "whatsup":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("types_full", lines)


def counterfactual_table(data):
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{$\Gamma^{(3)}$ true: partner} & \multicolumn{2}{c}{$\Gamma^{(3)}$ wrong: partner} & & \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Model & right & wrong & right & wrong & OR & $p$ \\", r"\midrule"]
    for m, mname in MODELS:
        recs, _, ops = data[("whatsup", m)]
        cf = counterfactual(recs, ops, 2.0)
        t = cf["table"]
        lines.append(f"{mname} & {t[0][0]} & {t[0][1]} & {t[1][0]} & {t[1][1]} & {cf['odds_ratio']:.2f} & {cf['p_one_sided']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("counterfactual", lines)


def dataset_table():
    from ria.datasets import vsr, whatsup
    wu = [e for s in ("A", "B") for e in whatsup.load(s, require_boxes=True)]
    v = vsr.load()
    cw, cv = Counter(e.true.rel for e in wu), Counter(x.relation for x in v)
    lines = [r"\begin{tabular}{@{}lllr@{}}", r"\toprule", r"Data & Relation & Class & Images \\", r"\midrule"]
    for k, (rel, n) in enumerate(sorted(cw.items(), key=lambda kv: -kv[1])):
        lines.append(f"{'What' + chr(39) + 's Up' if k == 0 else ''} & \\emph{{{rel}}} & {CATEGORY.get(rel, '')} & {n}" + r" \\")
    lines.append(r"\midrule")
    for k, (rel, n) in enumerate(sorted(cv.items(), key=lambda kv: -kv[1])):
        lines.append(f"{'VSR' if k == 0 else ''} & \\emph{{{rel}}} & {CATEGORY.get(rel, '')} & {n}" + r" \\")
    lines += [r"\midrule", f"total & & & {len(wu)} + {len(v)}" + r" \\", r"\bottomrule", r"\end{tabular}"]
    write("datasets", lines)
    subsets = Counter(e.subset for e in wu)
    print("what's up subsets:", dict(subsets), "flagged:", sum(bool(e.flags) for e in wu))


def e3_slices():
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Model & Slice & $n$ & acc & conf. & +1st & +2nd & +$\Gamma^{(3)}$ \\", r"\midrule"]
    for m, mname in MODELS:
        recs = [r for r in read_results(latest(f"e3_{m}")) if E3.finite(r)]
        _, ops = E3.valid_operators(recs, 0.25)
        slices = (("all", recs), ("confident boxes", [r for r in recs if not r["flags"]]),
                  ("SugarCrepe", [r for r in recs if r["dataset"].startswith("sugarcrepe")]),
                  ("ARO", [r for r in recs if r["dataset"].startswith("aro")]))
        for k, (label, sel) in enumerate(slices):
            a = E3.analyse(sel, ops)
            cells = [f"{a[s][0]:.3f}".lstrip("0") if a.get(s) and a[s][0] is not None else "--"
                     for s in ("confidence", "+ first order", "+ second order", "+ RIA (G3)")]
            lines.append(f"{mname if k == 0 else ''} & {label} & {a['n']} & {a['accuracy']:.2f}".replace(" 0.", " .")
                         + " & " + " & ".join(cells) + r" \\")
        if m != "qwen":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("e3_slices", lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {(ds, m): e2_run(ds, m) for ds in ("whatsup", "vsr") for m, _ in MODELS}
    operator_table(data)
    relation_table(data, "whatsup", "e2_relations_whatsup")
    relation_table(data, "vsr", "e2_relations_vsr")
    slices_table(data)
    wordbias_table(data)
    types_table(data)
    counterfactual_table(data)
    dataset_table()
    e3_slices()


if __name__ == "__main__":
    main()
