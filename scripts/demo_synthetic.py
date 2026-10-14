"""End-to-end RIA demo on a synthetic spatial-relation minimal pair.

Image A: red square above blue circle. Image B: the same shapes, swapped vertically.
Caption: "a red square above a blue circle". The relation term should differ between
A and B while the object terms stay comparable.

    py -3 scripts/demo_synthetic.py --config configs/demo_synthetic.json [--k 3]

Each run writes results/<stamp>_demo_synthetic/{config.json, results.jsonl}.
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ria import PhrasePlayer, RegionPlayer  # noqa: E402
from ria.interactions import pair_terms, robust_triple  # noqa: E402
import numpy as np  # noqa: E402
from ria.scorers import make_scorer  # noqa: E402
from ria.runlog import Run, load_config  # noqa: E402
from ria.shortlist import phrase_relevance_map, rank_regions  # noqa: E402

UPPER, LOWER = (72, 20, 152, 100), (72, 124, 152, 204)


def draw(square_box, circle_box, size):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    d.rectangle(square_box, fill=(220, 30, 30))
    d.ellipse(circle_box, fill=(30, 60, 220))
    return img


def fmt(r, t):
    star = "*" if r.significant[t] else " "
    return f"{r.mean[t]:>+9.4f}±{r.sd[t]:<7.4f}{star}"


def run(scorer, log, name, square_box, circle_box, caption, phrase, operators, k):
    img = draw(square_box, circle_box, scorer.size)
    a = RegionPlayer.from_box("square", square_box, scorer.size)
    b = RegionPlayer.from_box("circle", circle_box, scorer.size)
    c = RegionPlayer.complement("context", [a, b])
    games = scorer.method_games(img, caption, [a, b, c], phrase, operators)
    t0 = time.perf_counter()
    rows = [robust_triple(games, x, y, phrase.name, k=k)
            for x, y in (("square", "circle"), ("square", "context"), ("circle", "context"))]
    dt = time.perf_counter() - t0
    ranking = rank_regions(scorer, phrase_relevance_map(scorer, img, caption, phrase), [a, b, c])
    print(f"\n== {name}   ({len(games)} method configs, {dt:.1f}s for 3 triples)")
    print(f"   gradient shortlist for {phrase.span!r}: " + ", ".join(f"{k_}={v:.2e}" for k_, v in ranking.items()))
    print(f"   {'triple':<24}{'div a|p (mean±sd)':>19}{'div b|p':>18}{'G3':>18}   rho")
    for r in rows:
        rho = "undefined" if r.rho is None else f"{r.rho:.2f}±{r.rho_sd:.2f}"
        print(f"   {r.a + ',' + r.b + ',' + r.p:<24}{fmt(r, 'a|p')}{fmt(r, 'b|p')}{fmt(r, 'G3')}   {rho}")
        log.log({"kind": "robust_triple", "image": name, "caption": caption, **r.as_row(),
                 "configs": r.labels, "shortlist": ranking, "seconds_for_3_triples": dt})
    return rows


def contrastive(scorer, log, name, square_box, circle_box, caption, foil, operators):
    """Margin s(T) - s(T') and its square x circle interaction, per removal operator."""
    img = draw(square_box, circle_box, scorer.size)
    a = RegionPlayer.from_box("square", square_box, scorer.size)
    b = RegionPlayer.from_box("circle", circle_box, scorer.size)
    rows = {op: pair_terms(scorer.with_operator(op).contrastive_game(img, caption, [foil], [a, b]), "square", "circle")
            for op in operators}
    g2 = np.array([r["G2"] for r in rows.values()])
    margin = next(iter(rows.values()))["score"]
    per_op = ", ".join(f"{op} {r['G2']:+.4f}" for op, r in rows.items())
    sd = g2.std(ddof=1) if len(g2) > 1 else 0.0
    print(f"   {name:<40} margin {margin:+.4f}   square x circle on margin {g2.mean():+.4f} ± {sd:.4f}   ({per_op})")
    log.log({"kind": "contrastive", "image": name, "caption": caption, "foil": foil, "margin": margin,
             "per_operator": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--operators", nargs="+", default=["drop", "blur", "mean"], choices=["drop", "blur", "mean"])
    ap.add_argument("--substitutes", nargs="+", default=["below", "next to"])
    ap.add_argument("--k", type=float, default=2.0, help="noise floor: |mean| must exceed k * sd")
    cfg = load_config(ap)

    scorer = make_scorer(cfg["model"])
    caption = "a red square above a blue circle"
    phrase = PhrasePlayer("rel", "above", cfg["substitutes"])

    with Run(cfg["name"] or "demo_synthetic", cfg) as log:
        for cap in (caption, "a red square below a blue circle"):
            for nm, sq, ci in (("A (square on top)", UPPER, LOWER), ("B (circle on top)", LOWER, UPPER)):
                s = scorer.game(draw(sq, ci, scorer.size), cap, [], [])([frozenset()])[frozenset()]
                print(f"s({nm}, {cap!r}) = {s:.4f}")
                log.log({"kind": "caption_score", "image": nm, "caption": cap, "score": s})

        for nm, sq, ci in (("A: square above circle (caption true)", UPPER, LOWER),
                           ("B: circle above square (caption false)", LOWER, UPPER)):
            run(scorer, log, nm, sq, ci, caption, phrase, cfg["operators"], cfg["k"])
        foil = phrase.apply(caption, "below")
        print(f"\n== contrastive: {caption!r} vs {foil!r}")
        for nm, sq, ci in (("A: square above circle (caption true)", UPPER, LOWER),
                           ("B: circle above square (caption false)", LOWER, UPPER)):
            contrastive(scorer, log, nm, sq, ci, caption, foil, cfg["operators"])
        print("\n* = clears the noise floor (|mean| > k·sd across operators × substitutes)")
        print(f"logged {log.n} records to {log.dir}")


if __name__ == "__main__":
    main()
