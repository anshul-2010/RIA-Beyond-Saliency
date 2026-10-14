"""E2 replication on VSR (natural COCO photos), all models in one run (T10).

    python scripts/e2_vsr.py --models openai/clip-vit-base-patch16 google/siglip-base-patch16-224 Qwen/Qwen2.5-VL-3B-Instruct

For each true VSR caption whose relation has an opposite: the margin against the
opposite-relation caption, and RIA terms for (subj, obj), (subj, context), (obj, context)
with the opposite relation as the substitute, under each model's operators. Records use
the E2 format, so scripts/analyze_e2.py analyses them directly (no counterfactual test:
VSR has no paired images). Boxes: OWLv2, cached in data/boxes/vsr_owlv2.json.
"""

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e3_hardneg import ria_records  # noqa: E402
from ria.datasets import vsr  # noqa: E402
from ria.datasets.common import item_input  # noqa: E402
from ria.detect import BoxCache, OwlDetector, detect_all  # noqa: E402
from ria.runlog import REPO_ROOT, Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402

BOX_CACHE = REPO_ROOT / "data" / "boxes" / "vsr_owlv2.json"
OPERATORS = {"qwen": ["blur", "mean"]}          # token replacement is invalid for Qwen (E2)
DEFAULT_OPS = ["drop", "blur", "mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai/clip-vit-base-patch16"])
    ap.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--limit", type=int, default=0)
    cfg = load_config(ap)

    items = vsr.load()[: cfg["limit"] or None]
    cache = BoxCache(BOX_CACHE)
    todo = [x for x in items if x.key not in cache]
    if todo:
        print(f"detecting boxes for {len(todo)} items", flush=True)
        det = OwlDetector(cfg["detector"])
        detect_all(det, [(x.key, x.image, [x.subj, x.obj]) for x in todo], cache)
        del det
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    for model in cfg["models"]:
        scorer = make_scorer(model)
        ops = OPERATORS.get("qwen" if "qwen" in model.lower() else "", DEFAULT_OPS)
        name = f"e2_vsr_{model.split('/')[-1].split('-')[0].lower()}"
        with Run(name, {**cfg, "model": model, "operators": ops}) as log:
            if BOX_CACHE.exists():
                shutil.copy2(BOX_CACHE, log.dir / BOX_CACHE.name)
            t0 = time.perf_counter()
            for n, x in enumerate(items, 1):
                rec = cache.data.get(x.key)
                if not rec or {x.subj, x.obj} - set(rec["boxes"]):
                    continue
                try:
                    img = x.image()
                except FileNotFoundError:
                    continue
                it = item_input(img, x.caption, x.relation, [x.opposite], rec["boxes"][x.subj]["box"],
                                rec["boxes"][x.obj]["box"], scorer.size)
                t = time.perf_counter()
                out = {"key": x.key, "subset": "vsr", "group": x.key, "relation": x.relation, "opposite": x.opposite,
                       "caption": x.caption, "flags": rec["flags"],
                       "box_scores": {k: v["score"] for k, v in rec["boxes"].items()},
                       "margins": {x.opposite: scorer.margin(it.image, x.caption, [x.foil])},
                       "ria": ria_records(scorer, it, ops), "seconds": {}}
                out["seconds"]["ria"] = time.perf_counter() - t
                log.log(out)
                if n % 100 == 0 or n == len(items):
                    el = time.perf_counter() - t0
                    print(f"  {model}: {n}/{len(items)}  {el / n:.1f}s/item", flush=True)
            print(f"logged {log.n} records to {log.dir}", flush=True)
        del scorer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
