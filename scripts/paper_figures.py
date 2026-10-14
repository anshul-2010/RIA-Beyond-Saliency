"""Figures for the paper, drawn from the stored runs.

    py -3 scripts/paper_figures.py            # writes paper/figures/*.pdf

fig_teaser     a What's Up opposite pair: boxes, and the true-vs-opposite margin with both
               objects, without the subject, without the object and without both (Qwen vs CLIP)
fig_direction  share of images whose G3 agrees in sign across valid operators and points
               the true way / the wrong way, per model, What's Up and VSR
fig_wordbias   word-bias sums P(true|r) + P(true|r') per opposite pair, per model
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_e2 import (PAIRS, finite, margins, operator_validity, opposite_rows,  # noqa: E402
                        summarise)
from ria.runlog import REPO_ROOT, read_results  # noqa: E402

OUT = REPO_ROOT / "paper" / "figures"
MODELS = {"clip": "CLIP B/16", "siglip": "SigLIP B/16", "qwen": "Qwen2.5-VL-3B"}
COLOR = {"clip": "#4C72B0", "siglip": "#55A868", "qwen": "#DD8452"}
TRUE, WRONG, NONE = "#3B8E5A", "#C44E52", "#D9D9D9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def run(suffix):
    found = sorted((REPO_ROOT / "results").glob(f"*_{suffix}"))
    recs = [r for r in read_results(found[-1]) if finite(r)]
    return recs, operator_validity(recs, 0.25)[1]


def load_all():
    data = {}
    for m in MODELS:
        data[("whatsup", m)] = run(f"e2_whatsup_{m}")
        data[("vsr", m)] = run(f"e2_vsr_{'qwen2.5' if m == 'qwen' else m}")
    return data


def four_margins(rec, ops):
    rows = [r for r in opposite_rows(rec) if r["op"] in ops]
    return np.mean([margins(r) for r in rows], axis=0), np.mean([r["G3"] for r in rows])


def pick_pair(data):
    """An opposite pair (same objects) where Qwen is right on both with synergy, CLIP is not."""
    q, qops = data[("whatsup", "qwen")]
    c, cops = data[("whatsup", "clip")]
    cby = {r["key"]: r for r in c}
    by = {}
    for r in q:
        by.setdefault((r["group"], frozenset((r["relation"], r["opposite"]))), []).append(r)
    best = None
    for (g, rel), rs in by.items():
        if len(rs) != 2 or rel != frozenset(("on", "under")) or not all(x["key"] in cby for x in rs):
            continue
        qm = [four_margins(x, qops) for x in rs]
        cm = [four_margins(cby[x["key"]], cops) for x in rs]
        if all(m[0] > 0 and g3 > 0 for m, g3 in qm) and any(m[0] <= 0 for m, _ in cm):
            score = min(g3 / m[0] for m, g3 in qm)
            if best is None or score > best[0]:
                best = (score, sorted(rs, key=lambda x: x["relation"] != "on"))
    return best[1], cby


def fig_teaser(data):
    from PIL import Image
    from ria.datasets import whatsup
    pair, cby = pick_pair(data)
    qops, cops = data[("whatsup", "qwen")][1], data[("whatsup", "clip")][1]
    exs = {e.key: e for s in ("A", "B") for e in whatsup.load(s, require_boxes=True)}
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 2.75), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    labels = ["both", "−subj", "−obj", "−both"]
    for i, rec in enumerate(pair):
        ex = exs[rec["key"]]
        img = Image.open(ex.image_path).convert("RGB")
        ax = axes[i, 0]
        ax.imshow(img)
        for noun, col in ((ex.true.subj, "#E8A33D"), (ex.true.obj, "#3D7CE8")):
            x0, y0, x1, y1 = ex.boxes[noun]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.2, ec=col))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"“{rec['caption']}” vs. “{rec['opposite']}”", fontsize=7, pad=2)
        for j, (name, r, ops) in enumerate((("Qwen2.5-VL-3B", rec, qops), ("CLIP B/16", cby[rec["key"]], cops))):
            m, g3 = four_margins(r, ops)
            ax = axes[i, j + 1]
            cols = [TRUE if v > 0 else WRONG for v in m]
            ax.bar(range(4), m, color=cols, width=0.62)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xticks(range(4))
            ax.set_xticklabels(labels if i == 1 else [""] * 4)
            lim = max(abs(m).max() * 1.25, 1e-9)
            ax.set_ylim(-lim, lim)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.text(0.98, 0.95, f"$\\Gamma^{{(3)}}={g3:+.3g}$", transform=ax.transAxes, ha="right", va="top", fontsize=7)
            if i == 0:
                ax.set_title(name, fontsize=8, fontweight="bold", color=COLOR["qwen" if j == 0 else "clip"])
            if j == 0:
                ax.set_ylabel("margin (log-odds)" if i == 0 else "", fontsize=7)
            else:
                ax.set_ylabel("margin (cosine)" if i == 0 else "", fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.4)
    fig.savefig(OUT / "fig_teaser.pdf")
    plt.close(fig)
    print("teaser:", [r["key"] for r in pair])


def fig_direction(data):
    fig, axes = plt.subplots(1, 2, figsize=(3.1, 1.55), sharey=True)
    for ax, ds, title in zip(axes, ("whatsup", "vsr"), ("What's Up (820 images)", "VSR (969 images)")):
        for k, m in enumerate(MODELS):
            recs, ops = data[(ds, m)]
            s = summarise(recs, ops, 2.0)["all"]
            t, w = s["true"], s["wrong"]
            y = len(MODELS) - 1 - k
            ax.barh(y, t, color=TRUE, height=0.62)
            ax.barh(y, 1 - t - w, left=t, color=NONE, height=0.62)
            ax.barh(y, w, left=1 - w, color=WRONG, height=0.62)
            ax.text(t / 2, y, f"{t:.0%}", ha="center", va="center", color="white", fontsize=6.5)
            ax.text(1 - w / 2, y, f"{w:.0%}", ha="center", va="center", color="white", fontsize=6.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50%", "100%" if ds == "vsr" else ""])
        ax.set_title(title, fontsize=7.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels(list(MODELS.values())[::-1])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=TRUE, label="true direction"), Patch(color=NONE, label="operators disagree"),
                        Patch(color=WRONG, label="wrong direction")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.02), handlelength=1, columnspacing=1)
    fig.tight_layout(pad=0.2, rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_direction.pdf")
    plt.close(fig)


def word_bias_sums(recs, ops):
    s = summarise(recs, ops, 2.0)
    out = {}
    for r1, r2 in PAIRS:
        a, b = s.get(f"rel {r1}"), s.get(f"rel {r2}")
        if a and b and a["p_true_given_relational"] is not None and b["p_true_given_relational"] is not None:
            out[f"{r1} / {r2}"] = a["p_true_given_relational"] + b["p_true_given_relational"]
    return out


SHORT = {"behind / in front of": "behind / front", "to the right of / to the left of": "right / left",
         "on / under": "on / under", "on top of / beneath": "on top / beneath",
         "at the right side of / at the left side of": "right side / left side", "above / below": "above / below",
         "left of / right of": "left / right", "under / over": "under / over",
         "facing / facing away from": "facing / away", "far away from / close to": "far away / close",
         "far from / near": "far / near"}


def fig_wordbias(data):
    rows = []
    for ds in ("whatsup", "vsr"):
        sums = {m: word_bias_sums(*data[(ds, m)]) for m in MODELS}
        for pair in sums["qwen"]:
            rows.append((("WU: " if ds == "whatsup" else "VSR: ") + SHORT.get(pair, pair), {m: sums[m].get(pair) for m in MODELS}))
    fig, ax = plt.subplots(figsize=(3.1, 2.55))
    n = len(rows)
    ax.axvspan(0.7, 1.3, color="#F2F2F2", zorder=0)
    ax.axvline(1, color="#999999", lw=0.6, ls="--", zorder=1)
    ax.axvline(2, color="#999999", lw=0.6, ls="--", zorder=1)
    for i, (label, vals) in enumerate(rows):
        y = n - 1 - i
        for m in MODELS:
            if vals[m] is not None:
                ax.plot(vals[m], y, "o", ms=3.6, color=COLOR[m], zorder=3, label=MODELS[m] if i == 0 else None)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.5)
    ax.axhline(n - 3.5, color="black", lw=0.4)
    ax.set_xlim(0.5, 2.05)
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.set_xlabel("P(true | r) + P(true | r′)")
    ax.text(1.0, n - 0.3, "word-driven", ha="center", va="bottom", fontsize=6.5, color="#555555")
    ax.text(2.0, n - 0.3, "image-driven", ha="right", va="bottom", fontsize=6.5, color="#555555")
    ax.set_ylim(-0.6, n + 0.4)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_wordbias.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    fig_direction(data)
    fig_wordbias(data)
    fig_teaser(data)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
