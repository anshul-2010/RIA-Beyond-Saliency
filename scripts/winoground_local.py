"""Winoground, run locally (licence: the data and anything derived from it stay on this
machine). Independent replication of the E2 counterfactual test.

    py -3 scripts/winoground_local.py --model openai/clip-vit-base-patch16 [--limit 10]

Relation examples only. For image i with caption T_i and the swapped caption T_j as foil:
  margin       m_i = s(I_i, T_i) - s(I_i, T_j)
  joint term   interaction of the two entity regions on that margin (contrastive game,
               symmetric in the two regions), under every removal operator
Two entities per caption: the first two nouns (quantity / prepositional nouns resolved);
for "person ... person" the detector's two best non-overlapping boxes. Boxes are detected
with OWL-ViT B/32 on CPU and cached in <winoground>/ria_boxes.json (outside the repo).

Counterfactual test: among images the model gets right, does a joint term that agrees in
sign across operators and favours the true caption predict that the model is ALSO right
on the other image?
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria.datasets import winoground as W  # noqa: E402
from ria.datasets.common import item_input  # noqa: E402
from ria.detect import iou  # noqa: E402
from ria.interactions import pair_terms  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402

OUT_DIR = W.DEFAULT_ROOT / "ria_results"          # outside the repository, with the data
BOXES = W.DEFAULT_ROOT / "ria_boxes.json"


ARG_DEPS = {"nsubj", "nsubjpass", "dobj", "conj", "attr", "ROOT", "appos", "dative"}


def two_nouns(caption: str) -> list[str] | None:
    """The two main entities: clause arguments (subjects, objects, conjuncts, objects of
    prepositions attached to verbs), skipping nouns inside noun modifiers ("with earrings")."""
    from ria.text import PREP_NOUNS, _is_noun, _nlp, _noun_phrase, _resolve
    doc = _nlp()(caption)

    def is_arg(t):
        if t.dep_ in ARG_DEPS:
            return True
        return t.dep_ == "pobj" and t.head.dep_ in ("prep", "agent") and t.head.head.pos_ in ("VERB", "AUX")

    picked, seen = [], set()
    for pool in ([t for t in doc if _is_noun(t) and is_arg(t)], [t for t in doc if _is_noun(t)]):
        for t in pool:
            r = _resolve(t)
            if r.i in seen or r.lower_ in PREP_NOUNS or r.dep_ == "compound":
                continue
            seen.add(r.i)
            picked.append(_noun_phrase(doc, r.i))
            if len(picked) == 2:
                return picked
    return None


def detect_two(det, image, nouns):
    """Two boxes; for a repeated noun, the two best non-overlapping candidates."""
    if nouns[0] != nouns[1]:
        res = det.detect(image, nouns)
        return [res["boxes"][nouns[0]]["box"], res["boxes"][nouns[1]]["box"]], res["flags"]
    cands = det.candidates(image, [nouns[0]])[nouns[0]]
    first = cands[0]
    second = next((c for c in cands[1:] if iou(c["box"], first["box"]) < 0.3), None)
    if second is None:
        return None, ["no_second_instance"]
    flags = [f"low_score:{nouns[0]}" for c in (first, second) if c["score"] < 0.1]
    return [first["box"], second["box"]], flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--detector", default="google/owlvit-base-patch32")
    ap.add_argument("--operators", nargs="+", default=["drop", "blur", "mean"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    exs = [e for e in W.load() if e.collapsed_tag == "Relation"][: args.limit or None]
    cache = json.loads(BOXES.read_text(encoding="utf-8")) if BOXES.exists() else {}
    todo = [(e, i) for e in exs for i in (0, 1) if f"{e.id}/{i}" not in cache]
    if todo:
        from ria.detect import OwlDetector
        det = OwlDetector(args.detector, device="cpu")
        t0 = time.perf_counter()
        for n, (e, i) in enumerate(todo, 1):
            nouns = two_nouns(e.captions[0])
            boxes, flags = detect_two(det, e.image(i), nouns) if nouns else (None, ["no_nouns"])
            cache[f"{e.id}/{i}"] = {"nouns": nouns, "boxes": boxes, "flags": flags}
            if n % 25 == 0 or n == len(todo):
                BOXES.write_text(json.dumps(cache), encoding="utf-8")
                print(f"  detected {n}/{len(todo)}  {(time.perf_counter() - t0) / n:.1f}s/image", flush=True)
        del det

    scorer = make_scorer(args.model, device="cpu")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.model.split('/')[-1]}.jsonl"
    t0 = time.perf_counter()
    with open(out, "w", encoding="utf-8") as fh:
        for n, e in enumerate(exs, 1):
            for i in (0, 1):
                rec_box = cache[f"{e.id}/{i}"]
                if not rec_box["boxes"]:
                    continue
                img = e.image(i)
                it = item_input(img, e.captions[i], e.captions[i].split()[0], [], *rec_box["boxes"], scorer.size)
                foil = e.captions[1 - i]
                per_op = {}
                for op in args.operators:
                    g = scorer.with_operator(op).contrastive_game(it.image, e.captions[i], [foil], it.regions[:2])
                    per_op[op] = pair_terms(g, "subj", "obj")
                fh.write(json.dumps({"id": e.id, "image": i, "tag": e.tag, "caption": e.captions[i], "foil": foil,
                                     "nouns": rec_box["nouns"], "flags": rec_box["flags"],
                                     "margin": per_op[args.operators[0]]["score"], "per_op": per_op}) + "\n")
            if n % 20 == 0 or n == len(exs):
                print(f"  scored {n}/{len(exs)} examples  {(time.perf_counter() - t0) / n:.1f}s/example", flush=True)
    print(f"wrote {out}")
    analyse(out, args.operators)


def analyse(path, ops):
    from scipy.stats import fisher_exact
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    by = {(r["id"], r["image"]): r for r in recs}
    t = np.zeros((2, 2), int)
    right = [r["margin"] > 0 for r in recs]
    for r in recs:
        mate = by.get((r["id"], 1 - r["image"]))
        if mate is None or not r["margin"] > 0:
            continue
        g = np.array([r["per_op"][o]["G2"] for o in ops])
        if not ((g > 0).all() or (g < 0).all()):
            continue
        t[0 if g.mean() > 0 else 1, 0 if mate["margin"] > 0 else 1] += 1
    print(f"\n{Path(path).name}: {len(recs)} images, model right on {np.mean(right):.2f}")
    print(f"  among images the model gets right: joint term true -> mate right {t[0, 0]}/{t[0].sum()}, "
          f"joint term wrong -> mate right {t[1, 0]}/{t[1].sum()}")
    if t.sum(axis=1).min() > 0:
        odds, p = fisher_exact(t, alternative="greater")
        print(f"  odds ratio {odds:.2f}, one-sided Fisher p {p:.3f}")


if __name__ == "__main__":
    main()
