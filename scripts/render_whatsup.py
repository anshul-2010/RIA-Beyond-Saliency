"""Render What's Up examples as the model sees them (T09 check).

Each tile is the letterboxed model frame (shown at 2x) with the subject box (orange),
object box (blue), and the ViT patches that patch-dropping removes for each region
(tinted). Below: the true caption, the relation phrase and its substitutes.

    py -3 scripts/render_whatsup.py --keys A/beer-bottle_on_armchair.jpeg ... --out results/whatsup_examples.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.clip_scorer import ClipScorer  # noqa: E402
from ria.datasets import whatsup  # noqa: E402
from ria.players import make_disjoint  # noqa: E402
from ria.runlog import REPO_ROOT  # noqa: E402

SUBJ, OBJ = (240, 120, 0), (0, 90, 230)
DEFAULT_KEYS = ["A/beer-bottle_on_armchair.jpeg", "A/notepad_under_armchair.jpeg", "A/cup_left_of_armchair.jpeg",
                "B/bowl_in-front_of_cup.jpeg", "B/can_behind_knife.jpeg"]


def patch_overlay(frame: Image.Image, keep, grid: int, patch: int, color, alpha=90) -> Image.Image:
    over = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for i in np.flatnonzero(~keep.numpy()):
        r, c = divmod(int(i), grid)
        d.rectangle([c * patch, r * patch, (c + 1) * patch - 1, (r + 1) * patch - 1], fill=(*color, alpha))
    return Image.alpha_composite(frame.convert("RGBA"), over)


def tile(mi, scorer: ClipScorer, font, scale=2):
    img = mi.image
    drops = scorer.exclusive_patch_drops(make_disjoint(mi.regions))      # exactly what the scorer drops
    for name, color in (("subj", SUBJ), ("obj", OBJ)):
        img = patch_overlay(img, ~drops[name], scorer.grid, scorer.patch, color)
    img = img.convert("RGB").resize((img.width * scale, img.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for name, color in (("subj", SUBJ), ("obj", OBJ)):
        b = [v * scale for v in mi.boxes[name]]
        d.rectangle(b, outline=color, width=2)
    ex = mi.example
    w, text_h = img.width, 92
    canvas = Image.new("RGB", (w, img.height + text_h), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = img.height + 4
    drops = {n: int(v.sum()) for n, v in drops.items()}
    lines = [(ex.key, (90, 90, 90)),
             (f"true: {mi.caption}", (0, 0, 0)),
             (f"phrase '{mi.phrase.span}' -> {', '.join(mi.phrase.substitutes)}", (0, 0, 0)),
             (f"{ex.true.subj} {ex.box_scores[ex.true.subj]:.2f}, {drops['subj']} patches", SUBJ),
             (f"{ex.true.obj} {ex.box_scores[ex.true.obj]:.2f}, {drops['obj']} patches"
              + (f"   flags: {','.join(ex.flags)}" if ex.flags else ""), OBJ)]
    for text, color in lines:
        d.text((4, y), text, fill=color, font=font)
        y += 17
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "whatsup_examples.png")
    args = ap.parse_args()

    scorer = ClipScorer.__new__(ClipScorer)          # geometry only; no model load
    scorer.size, scorer.patch, scorer.grid, scorer.patch_overlap = args.size, args.patch, args.size // args.patch, 0.25
    exs = {e.key: e for s in whatsup.SUBSETS for e in whatsup.load(s, require_boxes=True)}
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    tiles = [tile(whatsup.to_model_input(exs[k], args.size), scorer, font) for k in args.keys]
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), max(t.height for t in tiles)), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 8
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"{len(tiles)} examples -> {args.out}")


if __name__ == "__main__":
    main()
