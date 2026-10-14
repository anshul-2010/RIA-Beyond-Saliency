"""Detect and cache object boxes for What's Up captions (T08).

    py -3 scripts/detect_boxes.py --config configs/detect_whatsup.json [--limit 30]

Boxes go to data/boxes/whatsup_<detector>.json (resumable); a run log with per-image
flags goes to results/. Re-running skips images already in the cache.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.datasets import whatsup  # noqa: E402
from ria.detect import BoxCache, OwlDetector, detect_all  # noqa: E402
from ria.runlog import REPO_ROOT, Run, load_config  # noqa: E402


def cache_path(model_name: str) -> Path:
    return REPO_ROOT / "data" / "boxes" / f"whatsup_{model_name.split('/')[-1]}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--subsets", nargs="+", default=["A", "B"])
    ap.add_argument("--limit", type=int, default=None, help="per subset, for quick checks")
    ap.add_argument("--flag-score", type=float, default=0.2)
    cfg = load_config(ap)

    items = []
    for s in cfg["subsets"]:
        exs = whatsup.load(s)[: cfg["limit"]]
        items += [(ex.key, ex.image_path, [ex.true.subj, ex.true.obj]) for ex in exs]

    cache = BoxCache(cache_path(cfg["detector"]))
    todo = sum(k not in cache for k, _, _ in items)
    print(f"{len(items)} images, {todo} to detect, cache {cache.path}")
    det = OwlDetector(cfg["detector"], flag_score=cfg["flag_score"])

    with Run(cfg["name"] or "detect_whatsup", cfg) as log:
        t0 = time.perf_counter()

        def on_item(key, res):
            done = log.n + 1
            rate = (time.perf_counter() - t0) / done
            log.log({"key": key, **res})
            if done % 10 == 0 or done == todo:
                print(f"  {done}/{todo}  {rate:.1f}s/image  eta {rate * (todo - done) / 60:.0f} min", flush=True)

        detect_all(det, items, cache, on_item=on_item)
        flagged = {k: cache[k]["flags"] for k, _, _ in items if cache[k]["flags"]}
        print(f"done; {len(flagged)} of {len(items)} images flagged for review")
        for k, f in list(flagged.items())[:20]:
            print(f"  {k}: {', '.join(f)}")


if __name__ == "__main__":
    main()
