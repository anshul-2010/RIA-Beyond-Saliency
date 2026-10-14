"""E1: does phrase-conditioned attribution localise the phrase's object? (T22)

    python scripts/e1_flickr.py --config configs/e1_clip.json [--per-type 125]

Flickr30k Entities test phrases, stratified by entity type. For each phrase the target is
the phrase-contrastive score c = s(I, T) - s(I, T with the phrase -> "something"), and
each method gives a patch map:
  gradcam / gradcam_signed   token Grad-CAM of c
  chefer                     Chefer et al. relevance of c
  ria_occlusion              first-order RIA: c(full) - c(without patch j), every patch j
                             removed in turn (patch dropping), one batched pass
Metrics per phrase: pointing game (map peak within 15 px of a ground-truth box, the
standard tolerance of Zhang et al. 2018; also reported strictly) and energy in box
(share of the map's positive mass inside the boxes). Image zip fetched from the HF mirror.
"""

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria import baselines as B  # noqa: E402
from ria.clip_scorer import box_to_frame, letterbox  # noqa: E402
from ria.datasets import flickr30k_entities as F  # noqa: E402
from ria.players import PhrasePlayer  # noqa: E402
from ria.runlog import Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402

FILLER = "something"


def sample(phrases, per_type, seed):
    by = defaultdict(list)
    for p in phrases:
        by[p.types[0]].append(p)
    rng = random.Random(seed)
    out = []
    for t in sorted(by):
        xs = by[t][:]
        rng.shuffle(xs)
        out += xs[:per_type]
    return out


def box_mask(boxes, size):
    m = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        m[max(int(y0), 0):min(int(np.ceil(y1)), size), max(int(x0), 0):min(int(np.ceil(x1)), size)] = True
    return m


def dilate_union(mask, tol):
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, structure=np.ones((2 * tol + 1, 2 * tol + 1), bool))


def metrics(patch_map, mask, scorer, tol=15):
    g, p = scorer.grid, scorer.patch
    frac = mask.reshape(g, p, g, p).mean(axis=(1, 3)).flatten()        # box coverage of each patch
    flat = np.asarray(patch_map, float).flatten()
    peak = int(np.argmax(flat))
    r, c = divmod(peak, g)
    cy, cx = min(r * p + p // 2, scorer.size - 1), min(c * p + p // 2, scorer.size - 1)
    hit_strict = bool(mask[cy, cx])
    hit = bool(dilate_union(mask, tol)[cy, cx]) if tol > 0 else hit_strict
    pos = np.clip(flat, 0, None)
    energy = float((pos * frac).sum() / pos.sum()) if pos.sum() > 0 else 0.0
    return {"hit": hit, "hit_strict": hit_strict, "energy": energy, "box_share": float(frac.mean())}


@torch.no_grad()
def occlusion_map(scorer, frame, text, phrase):
    """c(full) - c(without patch j) for every patch, patch dropping, batched."""
    t_full = scorer.encode_text([text])[0]
    t_sub = scorer.encode_text(scorer._texts(text, [phrase])).mean(0)
    target = t_full - t_sub
    n = scorer.grid ** 2
    pixels = scorer.to_tensor(frame)[None]
    base = float(scorer.encode_image(pixels)[0] @ target)
    out = np.zeros(n)
    for i in range(0, n, 64):
        idx = range(i, min(i + 64, n))
        keep = torch.ones(len(idx), n, dtype=torch.bool)
        for row, j in enumerate(idx):
            keep[row, j] = False
        emb = scorer.encode_image(pixels.expand(len(idx), -1, -1, -1), keep)
        out[i:i + len(idx)] = base - (emb @ target).cpu().numpy()
    return out.reshape(scorer.grid, scorer.grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--per-type", type=int, default=125)
    ap.add_argument("--image-zip", default=None, help="local Flickr30k image zip (default: fetch from HF)")
    cfg = load_config(ap)

    phrases = sample(F.load("test"), cfg["per_type"], cfg["seed"])
    zip_path = Path(cfg["image_zip"]) if cfg["image_zip"] else F.fetch_image_zip()
    scorer = make_scorer(cfg["model"])
    print(f"{len(phrases)} phrases, model {cfg['model']} on {scorer.device}", flush=True)

    with Run(cfg["name"] or "e1_flickr", cfg) as log:
        t0 = time.perf_counter()
        for n, p in enumerate(phrases, 1):
            img = F.image(p.image_id, zip_path)
            frame = letterbox(img, scorer.size)
            mask = box_mask([box_to_frame(b, img.size, scorer.size) for b in p.boxes], scorer.size)
            phrase = PhrasePlayer("p", p.phrase, [FILLER])
            maps = {"gradcam": B.gradcam(scorer, frame, p.sentence, phrase),
                    "gradcam_signed": B.gradcam(scorer, frame, p.sentence, phrase, signed=True),
                    "chefer": B.chefer(scorer, frame, p.sentence, phrase),
                    "ria_occlusion": occlusion_map(scorer, frame, p.sentence, phrase)}
            log.log({"key": p.key, "type": p.types[0], "phrase": p.phrase, "n_boxes": len(p.boxes),
                     "metrics": {m: metrics(v, mask, scorer) for m, v in maps.items()}})
            if n % 100 == 0 or n == len(phrases):
                el = time.perf_counter() - t0
                print(f"  {n}/{len(phrases)}  {el / n:.2f}s/phrase  eta {el / n * (len(phrases) - n) / 60:.0f} min", flush=True)
        print(f"logged {log.n} records to {log.dir}", flush=True)


if __name__ == "__main__":
    main()
