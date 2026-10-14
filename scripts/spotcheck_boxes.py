"""Render cached boxes on a random sample of images as one contact sheet (T08 spot-check).

    py -3 scripts/spotcheck_boxes.py --n 30 --out results/spotcheck_whatsup.png

Subject boxes are drawn in orange, object boxes in blue; flagged images get a red title.
"""

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.datasets import whatsup  # noqa: E402
from ria.detect import BoxCache  # noqa: E402
from ria.runlog import REPO_ROOT  # noqa: E402

SUBJ, OBJ, TILE = (240, 120, 0), (0, 90, 230), 300


def tile(ex, rec, font):
    im = Image.open(ex.image_path).convert("RGB")
    k = TILE / max(im.size)
    im = im.resize((round(im.width * k), round(im.height * k)))
    canvas = Image.new("RGB", (TILE, TILE + 34), "white")
    canvas.paste(im, ((TILE - im.width) // 2, 34 + (TILE - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    ox, oy = (TILE - im.width) // 2, 34 + (TILE - im.height) // 2
    for noun, color in ((ex.true.subj, SUBJ), (ex.true.obj, OBJ)):
        b, s = rec["boxes"][noun]["box"], rec["boxes"][noun]["score"]
        x0, y0, x1, y1 = (ox + b[0] * k, oy + b[1] * k, ox + b[2] * k, oy + b[3] * k)
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        d.text((x0 + 3, y0 + 2), f"{noun} {s:.2f}", fill=color, font=font)
    d.text((4, 2), ex.true.text[:48], fill=(200, 0, 0) if rec["flags"] else (0, 0, 0), font=font)
    d.text((4, 17), ex.key[:48] + ("  " + ",".join(rec["flags"]) if rec["flags"] else ""), fill=(90, 90, 90), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--flagged", action="store_true", help="sample only flagged images")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "spotcheck_whatsup.png")
    args = ap.parse_args()

    cache = BoxCache(REPO_ROOT / "data" / "boxes" / f"whatsup_{args.detector.split('/')[-1]}.json")
    exs = [ex for s in ("A", "B") for ex in whatsup.load(s) if ex.key in cache]
    if args.flagged:
        exs = [ex for ex in exs if cache[ex.key]["flags"]]
    sample = random.Random(args.seed).sample(exs, min(args.n, len(exs)))
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    tiles = [tile(ex, cache[ex.key], font) for ex in sample]
    rows = (len(tiles) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * TILE, rows * (TILE + 34)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % args.cols) * TILE, (i // args.cols) * (TILE + 34)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"{len(tiles)} images -> {args.out}")
    for ex in sample:
        print(f"  {ex.key}  flags={cache[ex.key]['flags']}")


if __name__ == "__main__":
    main()
