"""RIA in a few lines: does a model need BOTH objects to prefer the right relation?

    python examples/quickstart.py                                   # synthetic image, CLIP, CPU is fine
    python examples/quickstart.py --image photo.jpg --caption "A cup on a table" \
        --relation on --opposite under --subj-box 210 80 300 160 --obj-box 60 150 460 420

Boxes are [x0, y0, x1, y1] in original image pixels. For each removal operator the script
prints the four margins (true caption vs the opposite relation) with both objects, without
the figure, without the ground, without both, and the relational term
G3 = m - m_-a - m_-b + m_-ab. G3 is "directional" when its sign agrees across operators.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.datasets.common import item_input  # noqa: E402
from ria.interactions import memoize, powerset, relational_triple  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402


def synthetic():
    """A red square above a blue circle on a grey background."""
    img = Image.new("RGB", (448, 448), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([174, 60, 274, 160], fill=(210, 40, 40))
    d.ellipse([164, 250, 284, 370], fill=(40, 70, 210))
    return img, "A red square above a blue circle", "above", "below", [174, 60, 274, 160], [164, 250, 284, 370]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16",
                    help="also google/siglip-base-patch16-224 or Qwen/Qwen2.5-VL-3B-Instruct (GPU)")
    ap.add_argument("--image")
    ap.add_argument("--caption")
    ap.add_argument("--relation")
    ap.add_argument("--opposite")
    ap.add_argument("--subj-box", nargs=4, type=float)
    ap.add_argument("--obj-box", nargs=4, type=float)
    args = ap.parse_args()

    if args.image:
        img = Image.open(args.image).convert("RGB")
        caption, rel, opp, sbox, obox = args.caption, args.relation, args.opposite, args.subj_box, args.obj_box
    else:
        img, caption, rel, opp, sbox, obox = synthetic()

    scorer = make_scorer(args.model)
    ops = ["blur", "mean"] if "qwen" in args.model.lower() else ["drop", "blur", "mean"]
    it = item_input(img, caption, rel, [opp], sbox, obox, scorer.size)
    print(f"{args.model}\n\"{caption}\" vs. \"{opp}\"\n")
    print(f"{'operator':<9}{'m':>10}{'m_-a':>10}{'m_-b':>10}{'m_-ab':>10}{'G3':>10}")
    g3s = []
    for op in ops:
        game = memoize(scorer.with_operator(op).game(it.image, it.caption, it.regions, [it.phrase]))
        r = relational_triple(game, "subj", "obj", "rel")
        m = r.effect["rel"]
        ma, mb = m - r.pair_local["subj|rel"], m - r.pair_local["obj|rel"]
        mab = r.triple - m + ma + mb
        g3s.append(r.triple)
        print(f"{op:<9}{m:>10.4f}{ma:>10.4f}{mb:>10.4f}{mab:>10.4f}{r.triple:>10.4f}")
    if all(g > 0 for g in g3s):
        verdict = "directional, TRUE: the preference for the true relation needs both objects"
    elif all(g < 0 for g in g3s):
        verdict = "directional, negative: redundancy or a wrong answer (check the margins)"
    else:
        verdict = "not directional: operators disagree on the sign"
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
