"""Sanity check for the E3 loaders (T11): model accuracy on hard negatives.

    py -3 scripts/check_hardneg_accuracy.py --model openai/clip-vit-base-patch32 \
        --sugarcrepe replace_rel swap_obj --aro 2000

Accuracy = fraction of items where s(I, true) > s(I, foil). Compare with the published
numbers (SugarCrepe paper Table 2; ARO paper Table 1) to validate the loaders.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from ria.clip_scorer import letterbox  # noqa: E402
from ria.datasets.compositional import load_aro_relation, load_sugarcrepe  # noqa: E402
from ria.runlog import Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402


def center_crop(image, size):
    """CLIP-style preprocessing: resize the shorter side to `size`, then crop the centre."""
    w, h = image.size
    k = size / min(w, h)
    image = image.resize((max(size, round(w * k)), max(size, round(h * k))), Image.BICUBIC)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--sugarcrepe", nargs="*", default=["replace_rel"])
    ap.add_argument("--aro", type=int, default=2000, help="random ARO VG-Relation items (0 = none, -1 = all)")
    ap.add_argument("--frames", nargs="+", default=["letterbox"], choices=["letterbox", "center"],
                    help="letterbox = pad to square (RIA default); center = CLIP-style centre crop")
    cfg = load_config(ap)

    scorer = make_scorer(cfg["model"])
    sets = {f"sugarcrepe/{s}": load_sugarcrepe(s) for s in cfg["sugarcrepe"]}
    if cfg["aro"]:
        aro = load_aro_relation()
        if cfg["aro"] > 0:
            aro = random.Random(cfg["seed"]).sample(aro, cfg["aro"])
        sets["aro/vg_relation"] = aro

    with Run(cfg["name"] or "hardneg_accuracy", cfg) as log:
        for name, items in sets.items():
            for frame in cfg["frames"]:
                prep = letterbox if frame == "letterbox" else center_crop
                t0, right, failed = time.perf_counter(), 0, 0
                for i, x in enumerate(items, 1):
                    try:
                        img = prep(x.image(), scorer.size)
                    except FileNotFoundError:
                        failed += 1
                        continue
                    m = scorer.margin(img, x.caption, [x.foil])
                    right += m > 0
                    log.log({"set": name, "frame": frame, "key": x.key, "margin": m, "relation": x.relation})
                    if i % 500 == 0:
                        print(f"  {name} [{frame}] {i}/{len(items)}  acc so far {right / (i - failed):.3f}", flush=True)
                n = len(items) - failed
                print(f"{name} [{frame}]: accuracy {right / max(n, 1):.3f} on {n} items ({failed} unavailable, "
                      f"{time.perf_counter() - t0:.0f}s)", flush=True)
                log.log({"set": name, "frame": frame, "summary": True, "accuracy": right / max(n, 1), "n": n,
                         "unavailable": failed})


if __name__ == "__main__":
    main()
