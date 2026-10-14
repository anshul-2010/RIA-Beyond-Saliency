"""E5 task-conditioning check (T25): different phrases in the same image must give
different maps, each concentrated on its own object.

    py -3 scripts/e5_task_conditioning.py --n 40

What's Up images: maps for the subject noun and for the object noun of the same caption
(each contrasted with "something"). Per method:
  map correlation   Spearman correlation of the two maps (lower = better conditioned)
  own-object share  fraction of images where each phrase's map puts more positive mass on
                    its own object's patches than on the other object's
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e1_flickr as E  # noqa: E402
from ria import baselines as B  # noqa: E402
from ria.datasets import whatsup  # noqa: E402
from ria.players import PhrasePlayer  # noqa: E402
from ria.runlog import Run, load_config  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402


def maps(scorer, mi, noun):
    phrase = PhrasePlayer("p", noun, ["something"])
    return {"ria_occlusion": E.occlusion_map(scorer, mi.image, mi.caption, phrase),
            "gradcam_signed": B.gradcam(scorer, mi.image, mi.caption, phrase, signed=True),
            "chefer": B.chefer(scorer, mi.image, mi.caption, phrase)}


def mass(m, owned):
    pos = np.clip(np.asarray(m, float).flatten(), 0, None)
    return float(pos[owned].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--n", type=int, default=40)
    cfg = load_config(ap)
    exs = [e for s in ("A", "B") for e in whatsup.load(s, require_boxes=True) if not e.flags]
    exs = random.Random(cfg["seed"]).sample(exs, cfg["n"])
    sc = make_scorer(cfg["model"], device="cpu")

    corr, own = {}, {}
    with Run(cfg["name"] or "e5_task_conditioning", cfg) as log:
        for i, ex in enumerate(exs, 1):
            mi = whatsup.to_model_input(ex, sc.size)
            sets = B.region_patch_sets(sc, mi.regions)
            ms, mo = maps(sc, mi, ex.true.subj), maps(sc, mi, ex.true.obj)
            for meth in ms:
                c = spearmanr(np.ravel(ms[meth]), np.ravel(mo[meth])).statistic
                ok = (mass(ms[meth], sets["subj"]) > mass(ms[meth], sets["obj"])) + \
                     (mass(mo[meth], sets["obj"]) > mass(mo[meth], sets["subj"]))
                corr.setdefault(meth, []).append(c)
                own.setdefault(meth, []).append(ok / 2)
                log.log({"key": ex.key, "method": meth, "map_corr": c, "own_share": ok / 2})
            if i % 10 == 0:
                print(f"  {i}/{len(exs)}", flush=True)
        print("\n| method | map correlation (subject vs object phrase) | own-object share |\n|---|---|---|")
        for meth in corr:
            print(f"| {meth} | {np.nanmean(corr[meth]):.2f} | {np.mean(own[meth]):.2f} |")
            log.log({"summary": True, "method": meth, "map_corr": float(np.nanmean(corr[meth])),
                     "own_share": float(np.mean(own[meth]))})


if __name__ == "__main__":
    main()
