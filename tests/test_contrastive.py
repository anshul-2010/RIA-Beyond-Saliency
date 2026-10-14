"""Contrastive margin games (T04)."""

import math
import os

import numpy as np
import pytest

from ria.interactions import pair_terms, powerset, relational_triple


def test_margin_interaction_equals_third_order_term_on_random_games():
    # For a game over {a, b, p} where removing p means "use the foil caption",
    # the margin game is m(S) = v(S) - v(S + p), and its a-b interaction is Gamma3.
    rng = np.random.default_rng(0)
    for _ in range(20):
        table = {s: rng.normal() for s in powerset("abp")}
        game = lambda subsets: {s: table[s] for s in subsets}
        margin = lambda subsets: {s: table[s] - table[s | {"p"}] for s in subsets}
        g3 = relational_triple(game, "a", "b", "p").triple
        assert math.isclose(pair_terms(margin, "a", "b")["G2"], g3, abs_tol=1e-12)
        # and the first-order margin effect of a is the a-p pairwise interaction (local)
        assert math.isclose(pair_terms(margin, "a", "b")["d[a]"],
                            relational_triple(game, "a", "b", "p").pair_local["a|p"], abs_tol=1e-12)


def _model_cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "config.json"), str)
    except Exception:
        return False


MODEL = "openai/clip-vit-base-patch32"


@pytest.mark.skipif(not _model_cached(MODEL), reason="CLIP not cached locally")
def test_clip_contrastive_game_matches_triple_and_margin():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from PIL import Image, ImageDraw

    from ria.clip_scorer import ClipScorer
    from ria.players import PhrasePlayer, RegionPlayer

    scorer = ClipScorer(MODEL)
    img = Image.new("RGB", (scorer.size, scorer.size), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((72, 20, 152, 100), fill=(220, 30, 30))
    d.ellipse((72, 124, 152, 204), fill=(30, 60, 220))
    a = RegionPlayer.from_box("a", (72, 20, 152, 100), scorer.size)
    b = RegionPlayer.from_box("b", (72, 124, 152, 204), scorer.size)
    text, foil = "a red square above a blue circle", "a red square below a blue circle"

    g3 = relational_triple(scorer.game(img, text, [a, b], [PhrasePlayer("p", "above", ["below"])]), "a", "b", "p").triple
    terms = pair_terms(scorer.contrastive_game(img, text, [foil], [a, b]), "a", "b")
    assert math.isclose(terms["G2"], g3, abs_tol=1e-5)

    s_true = scorer.game(img, text, [], [])([frozenset()])[frozenset()]
    s_foil = scorer.game(img, foil, [], [])([frozenset()])[frozenset()]
    assert math.isclose(scorer.margin(img, text, [foil]), s_true - s_foil, abs_tol=1e-5)

    with pytest.raises(ValueError):
        scorer.contrastive_game(img, text, [], [a])
