"""Figures for the COLING version, drawn from the stored runs.

    py -3 scripts/coling_figures.py            # writes paper_coling/figures/*.pdf

fig_teaser      What's Up on/under pair: boxes and the four margins (Qwen vs CLIP), verdicts
fig_patterns    idealised margin patterns that define AND, OR, additive and wrong evidence
fig_direction   directional rates of G3 per model and dataset
fig_wordbias    word-bias sums per opposite pair, grouped by linguistic class
fig_relations   Qwen's true/wrong directional rates per VSR relation, by linguistic class
fig_gallery     appendix: redundancy, a word-driven failure and synergy on single images
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paper_figures as P  # noqa: E402  (style, data loading, margins)
from analyze_e2 import interaction_types, summarise  # noqa: E402
from ria.runlog import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "paper_coling" / "figures"
TRUE, WRONG, NONE, NEUTRAL = P.TRUE, P.WRONG, P.NONE, "#9AA5B1"
SUBJ, OBJ = "#E8A33D", "#3D7CE8"

# linguistic classes of opposite pairs, in display order
CLASSES = [
    ("lateral", [("whatsup", "to the right of", "to the left of", "right / left (WU)"),
                 ("vsr", "left of", "right of", "left / right"),
                 ("vsr", "at the right side of", "at the left side of", "right side / left side")]),
    ("vertical", [("vsr", "above", "below", "above / below"),
                  ("vsr", "on top of", "beneath", "on top of / beneath"),
                  ("vsr", "under", "over", "under / over")]),
    ("frontal", [("whatsup", "behind", "in front of", "behind / in front (WU)"),
                 ("vsr", "behind", "in front of", "behind / in front of")]),
    ("support", [("whatsup", "on", "under", "on / under (WU)")]),
    ("orientation", [("vsr", "facing", "facing away from", "facing / facing away")]),
    ("distance", [("vsr", "far away from", "close to", "far away / close to"),
                  ("vsr", "far from", "near", "far from / near")]),
]
REL_CLASS = {}
for cls, pairs in CLASSES:
    for ds, r1, r2, _ in pairs:
        if ds == "vsr":
            REL_CLASS[r1] = REL_CLASS[r2] = cls


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def verdict(m, g3):
    if m[0] <= 0:
        return "model wrong", WRONG
    if abs(g3) < 0.15 * abs(m[0]):
        return "no joint use", NEUTRAL
    if g3 > 0:
        return "AND: synergy", TRUE
    if m[1] > 0 and m[2] > 0 and m[3] < min(m[1], m[2]):
        return "OR: redundancy", "#8E6C3B"
    return "other", NEUTRAL


def margin_panel(ax, m, g3, labels, title=None, title_color="black", ylabel=None, show_x=True):
    cols = [TRUE if v > 0 else WRONG for v in m]
    ax.bar(range(4), m, color=cols, width=0.64, zorder=2)
    ax.axhline(0, color="black", lw=0.6, zorder=3)
    m = np.asarray(m)
    lo, hi = min(m.min(), 0.0), max(m.max(), 0.0)
    span = max(hi - lo, 1e-9)
    ax.set_ylim(lo - 0.08 * span, hi + 0.55 * span)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels if show_x else [""] * 4, fontsize=6.3)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.tick_params(axis="y", labelsize=6)
    text, col = verdict(m, g3)
    ax.text(0.98, 0.96, f"$\\Gamma^{{(3)}}={g3:+.3g}$" if abs(g3) >= 1e-3 * max(abs(m).max(), 1e-9)
            else "$\\Gamma^{(3)}\\approx 0$", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    ax.text(0.98, 0.80, text, transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=col, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=7.5, fontweight="bold", color=title_color, pad=3)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=6.5)


def show_image(ax, img, boxes, title):
    ax.imshow(img)
    for (x0, y0, x1, y1), col in boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.3, ec=col))
    clean(ax)
    ax.set_title(title, fontsize=6.8, pad=2)


def obj_labels(subj, obj):
    def short(s):
        s = s.split()[-1]
        return s if len(s) <= 7 else s[:6] + "."
    return ["both", f"−{short(subj)}", f"−{short(obj)}", "−both"]


# ------------------------------------------------------------------ teaser
def fig_teaser(data):
    from PIL import Image
    from ria.datasets import whatsup
    pair, cby = P.pick_pair({k: (v[0], v[1]) for k, v in data.items()})
    qops, cops = data[("whatsup", "qwen")][1], data[("whatsup", "clip")][1]
    exs = {e.key: e for s in ("A", "B") for e in whatsup.load(s, require_boxes=True)}
    fig = plt.figure(figsize=(6.3, 2.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], wspace=0.42, hspace=0.32)
    for i, rec in enumerate(pair):
        ex = exs[rec["key"]]
        ax = fig.add_subplot(gs[i, 0])
        show_image(ax, Image.open(ex.image_path).convert("RGB"),
                   [(ex.boxes[ex.true.subj], SUBJ), (ex.boxes[ex.true.obj], OBJ)],
                   f"({'ab'[i]}) “{rec['caption']}”  vs.  “{rec['opposite']}”")
        labels = obj_labels(ex.true.subj, ex.true.obj)
        for j, (name, r, ops, col, unit) in enumerate((("Qwen2.5-VL-3B", rec, qops, P.COLOR["qwen"], "log-odds"),
                                                       ("CLIP B/16", cby[rec["key"]], cops, P.COLOR["clip"], "cosine"))):
            m, g3 = P.four_margins(r, ops)
            ax = fig.add_subplot(gs[i, j + 1])
            margin_panel(ax, m, g3, labels, title=name if i == 0 else None, title_color=col,
                         ylabel=f"margin ({unit})", show_x=True)
    save(fig, "fig_teaser")
    return pair


# ------------------------------------------------------------------ idealised patterns
def fig_patterns():
    pats = [("AND: synergy", "the relation needs\nboth objects", [1.0, 0.05, 0.05, 0.0]),
            ("OR: redundancy", "either object\nsuffices", [1.0, 0.9, 0.9, 0.05]),
            ("additive", "objects contribute\nseparately", [1.0, 0.45, 0.55, 0.0]),
            ("model wrong", "opposite relation\npreferred", [-0.8, -0.2, -0.3, 0.0])]
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.45), sharey=True)
    for ax, (title, sub, m) in zip(axes, pats):
        g3 = m[0] - m[1] - m[2] + m[3]
        ax.bar(range(4), m, color=[TRUE if v > 0 else WRONG for v in m], width=0.64, zorder=2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-1.15, 1.35)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["$m$", "$m_{-a}$", "$m_{-b}$", "$m_{-ab}$"], fontsize=6.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=7.5, fontweight="bold", pad=2)
        ax.text(3.45, 1.3, sub, ha="right", va="top", fontsize=6.2, color="#444444", linespacing=1.05)
        sign = "0" if abs(g3) < 1e-9 else (f"+{g3:.2g}" if g3 > 0 else f"{g3:.2g}")
        ax.text(3.45, -1.08, f"$\\Gamma^{{(3)}}={sign}$", ha="right", va="bottom", fontsize=6.8)
    fig.tight_layout(pad=0.25, w_pad=0.9)
    save(fig, "fig_patterns")


# ------------------------------------------------------------------ direction
def fig_direction(data):
    fig, axes = plt.subplots(1, 2, figsize=(3.1, 1.5), sharey=True)
    for ax, ds, title in zip(axes, ("whatsup", "vsr"), ("What's Up (820)", "VSR (969)")):
        for k, m in enumerate(P.MODELS):
            recs, ops = data[(ds, m)]
            s = summarise(recs, ops, 2.0)["all"]
            t, w = s["true"], s["wrong"]
            y = len(P.MODELS) - 1 - k
            ax.barh(y, t, color=TRUE, height=0.62)
            ax.barh(y, 1 - t - w, left=t, color=NONE, height=0.62)
            ax.barh(y, w, left=1 - w, color=WRONG, height=0.62)
            ax.text(t / 2, y, f"{t:.0%}", ha="center", va="center", color="white", fontsize=6.4)
            ax.text(1 - w / 2, y, f"{w:.0%}", ha="center", va="center", color="white", fontsize=6.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50%", "100%" if ds == "vsr" else ""])
        ax.set_title(title, fontsize=7.3)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(P.MODELS)))
    axes[0].set_yticklabels(list(P.MODELS.values())[::-1])
    fig.legend(handles=[Patch(color=TRUE, label="true direction"), Patch(color=NONE, label="operators disagree"),
                        Patch(color=WRONG, label="wrong direction")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.02), handlelength=1, columnspacing=1)
    fig.tight_layout(pad=0.2, rect=(0, 0.1, 1, 1))
    save(fig, "fig_direction")


# ------------------------------------------------------------------ word bias by class
def fig_wordbias(data):
    sums = {(ds, m): summarise(*data[(ds, m)], 2.0) for ds in ("whatsup", "vsr") for m in P.MODELS}

    def wb(ds, m, r1, r2):
        a, b = sums[(ds, m)].get(f"rel {r1}"), sums[(ds, m)].get(f"rel {r2}")
        pa, pb = (a or {}).get("p_true_given_relational"), (b or {}).get("p_true_given_relational")
        return None if pa is None or pb is None else pa + pb

    rows, groups = [], []
    for cls, pairs in CLASSES:
        start = len(rows)
        for ds, r1, r2, label in pairs:
            rows.append((label, {m: wb(ds, m, r1, r2) for m in P.MODELS}))
        groups.append((cls, start, len(rows) - 1))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    ax.axvspan(0.75, 1.25, color="#F2F2F2", zorder=0)
    for x in (1, 2):
        ax.axvline(x, color="#999999", lw=0.6, ls="--", zorder=1)
    for k, (cls, a, b) in enumerate(groups):
        if k % 2 == 1:
            ax.axhspan(n - 1 - b - 0.5, n - 1 - a + 0.5, color="#F7F4EC", zorder=0)
        ax.text(2.08, n - 1 - (a + b) / 2, cls, ha="left", va="center", fontsize=6.5, fontstyle="italic", color="#555555")
    for i, (label, vals) in enumerate(rows):
        y = n - 1 - i
        xs = [v for v in vals.values() if v is not None]
        ax.plot([min(xs), max(xs)], [y, y], color="#CCCCCC", lw=0.8, zorder=2)
        for m in P.MODELS:
            if vals[m] is not None:
                ax.plot(vals[m], y, "o", ms=3.8, color=P.COLOR[m], zorder=3, label=P.MODELS[m] if i == 0 else None,
                        markeredgecolor="white", markeredgewidth=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.4)
    ax.set_xlim(0.5, 2.05)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.set_xlabel("$P(\\mathrm{true}\\mid r)+P(\\mathrm{true}\\mid r')$")
    ax.text(1.0, n - 0.35, "word-driven", ha="center", va="bottom", fontsize=6.3, color="#555555")
    ax.text(1.98, n - 0.35, "image-driven", ha="right", va="bottom", fontsize=6.3, color="#555555")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, frameon=False, handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(right=0.80)
    save(fig, "fig_wordbias")


# ------------------------------------------------------------------ Qwen per relation on VSR
def fig_relations(data):
    recs, ops = data[("vsr", "qwen")]
    s = summarise(recs, ops, 2.0)
    order = [c for c, _ in CLASSES if c not in ("support",)]
    rows = []
    for cls in order:
        rels = [r for r, c in REL_CLASS.items() if c == cls and f"rel {r}" in s]
        rels.sort(key=lambda r: s[f"rel {r}"]["true"] - s[f"rel {r}"]["wrong"], reverse=True)
        rows += [(cls, r, s[f"rel {r}"]) for r in rels]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(3.1, 3.1))
    prev = None
    for i, (cls, rel, b) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, b["true"], color=TRUE, height=0.68)
        ax.barh(y, -b["wrong"], color=WRONG, height=0.68)
        for v, sgn in ((b["true"], 1), (b["wrong"], -1)):
            inside = v >= 0.26
            x = sgn * (v - 0.02) if inside else sgn * (v + 0.02)
            ha = ("right" if sgn > 0 else "left") if inside else ("left" if sgn > 0 else "right")
            ax.text(x, y, f"{v:.0%}", va="center", ha=ha, fontsize=5.7, color="white" if inside else "#333333")
        ax.text(1.24, y, f"{b['model_acc']:.0%}", va="center", ha="right", fontsize=5.8, color="#555555")
        if cls != prev and i > 0:
            ax.axhline(y + 0.5, color="#BBBBBB", lw=0.5)
        prev = cls
    starts = {}
    for i, (cls, _, _) in enumerate(rows):
        starts.setdefault(cls, []).append(n - 1 - i)
    for cls, ys in starts.items():
        ax.annotate(cls, xy=(-0.70, (np.mean(ys) + 0.6) / (n + 0.8)), xycoords="axes fraction",
                    ha="right", va="center", fontsize=6.3, fontstyle="italic", color="#555555")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=6.3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(-0.95, 1.25)
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_xticklabels(["50%", "0", "50%", "100%"], fontsize=6)
    ax.text(1.24, n - 0.3, "acc.", ha="right", va="bottom", fontsize=6, color="#555555")
    ax.text(-0.05, n - 0.3, "← wrong", ha="right", va="bottom", fontsize=6.3, color=WRONG)
    ax.text(0.05, n - 0.3, "true →", ha="left", va="bottom", fontsize=6.3, color=TRUE)
    ax.set_ylim(-0.6, n + 0.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(left=0.52)
    save(fig, "fig_relations")


# ------------------------------------------------------------------ appendix gallery
def fig_gallery(data):
    from PIL import Image
    from ria.datasets import vsr, whatsup
    from ria.detect import BoxCache
    q_wu, qops = data[("whatsup", "qwen")]
    c_wu, cops = data[("whatsup", "clip")]
    q_vsr, _ = data[("vsr", "qwen")]
    c_vsr, _ = data[("vsr", "clip")]
    cby = {r["key"]: r for r in c_wu + c_vsr}

    def best(recs, rel, want, key):
        cands = []
        for r in recs:
            if r["relation"] != rel or r["key"] not in cby:
                continue
            m, g3 = P.four_margins(r, qops)
            if verdict(m, g3)[0].startswith(want):
                cands.append((key(m, g3), r))
        return max(cands, key=lambda c: c[0])[1]

    picks = [
        ("whatsup", best(q_wu, "behind", "OR", lambda m, g: min(m[1], m[2]) - m[3])),
        ("vsr", best(q_vsr, "facing away from", "model wrong", lambda m, g: -m[0])),
        ("vsr", best(q_vsr, "left of", "AND", lambda m, g: g / max(m[0], 1e-6))),
    ]
    wu = {e.key: e for s in ("A", "B") for e in whatsup.load(s, require_boxes=True)}
    vs = {x.key: x for x in vsr.load()}
    vbox = BoxCache(REPO_ROOT / "data" / "boxes" / "vsr_owlv2.json").data
    fig = plt.figure(figsize=(6.3, 5.6))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.1, 1, 1], wspace=0.42, hspace=0.45)
    for i, (ds, rec) in enumerate(picks):
        if ds == "whatsup":
            ex = wu[rec["key"]]
            img, subj, obj = Image.open(ex.image_path).convert("RGB"), ex.true.subj, ex.true.obj
            boxes = [(ex.boxes[subj], SUBJ), (ex.boxes[obj], OBJ)]
        else:
            x = vs[rec["key"]]
            img, subj, obj = x.image(), x.subj, x.obj
            b = vbox[x.key]["boxes"]
            boxes = [(b[subj]["box"], SUBJ), (b[obj]["box"], OBJ)]
        import textwrap
        cap = "\n".join(textwrap.wrap(f"({'abc'[i]}) “{rec['caption'].rstrip('.')}”  vs. “{rec['opposite']}”", 40))
        show_image(fig.add_subplot(gs[i, 0]), img, boxes, cap)
        labels = obj_labels(subj, obj)
        for j, (name, r, ops, col, unit) in enumerate((("Qwen2.5-VL-3B", rec, qops, P.COLOR["qwen"], "log-odds"),
                                                       ("CLIP B/16", cby[rec["key"]], cops, P.COLOR["clip"], "cosine"))):
            m, g3 = P.four_margins(r, ops)
            margin_panel(fig.add_subplot(gs[i, j + 1]), m, g3, labels, title=name if i == 0 else None,
                         title_color=col, ylabel=f"margin ({unit})")
    save(fig, "fig_gallery")
    print("gallery:", [p[1]["key"] for p in picks])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for m in P.MODELS:
        data[("whatsup", m)] = P.run(f"e2_whatsup_{m}")
        data[("vsr", m)] = P.run(f"e2_vsr_{'qwen2.5' if m == 'qwen' else m}")
    fig_patterns()
    fig_direction(data)
    fig_wordbias(data)
    fig_relations(data)
    fig_teaser(data)
    fig_gallery(data)


if __name__ == "__main__":
    main()
