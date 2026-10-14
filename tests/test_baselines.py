"""Baselines (T19): shared region aggregation and model-backed sanity checks."""

import os

import numpy as np
import pytest

from ria import baselines as B

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def test_aggregate_uses_owned_patches():
    scores = np.arange(4.0)
    sets = {"a": np.array([True, True, False, False]), "b": np.array([False, False, True, False]),
            "empty": np.zeros(4, dtype=bool)}
    out = B.aggregate(scores, sets)
    assert out["a"] == {"mean": 0.5, "sum": 1.0}
    assert out["b"] == {"mean": 2.0, "sum": 2.0}
    assert out["empty"] == {"mean": 0.0, "sum": 0.0}


def _cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "config.json"), str)
    except Exception:
        return False


CLIP = "openai/clip-vit-base-patch32"


@pytest.mark.skipif(not _cached(CLIP), reason="CLIP not cached")
def test_first_order_maps_have_grid_shape_and_are_nonnegative():
    from PIL import Image, ImageDraw

    from ria.players import PhrasePlayer
    from ria.scorers import make_scorer
    sc = make_scorer(CLIP)
    img = Image.new("RGB", (sc.size, sc.size), "white")
    ImageDraw.Draw(img).rectangle((60, 20, 160, 100), fill=(220, 30, 30))
    phrase = PhrasePlayer("rel", "above", ["below"])
    for m in (B.gradcam(sc, img, "a red square above a circle", phrase),
              B.gradcam(sc, img, "a red square above a circle", None),
              B.chefer(sc, img, "a red square above a circle", phrase)):
        assert m.shape == (sc.grid, sc.grid) and (m >= 0).all() and np.isfinite(m).all()


@pytest.mark.skipif(not (B.FIXLIP_DIR / "src" / "fixlip.py").exists() or not _cached(CLIP),
                    reason="FIxLIP or CLIP not available")
def test_fixlip_phrase_players_for_clip():
    fx = B.FixlipBaseline.__new__(B.FixlipBaseline)
    from transformers import AutoProcessor
    fx.processor, fx.is_clip = AutoProcessor.from_pretrained(CLIP), True
    # tokens: <bos> a mug to the right of a knife <eos>; players drop BOS/EOS
    assert fx.phrase_players("a mug to the right of a knife", "to the right of") == [2, 3, 4, 5]
    assert fx.phrase_players("a lemon on a table", "on") == [2]


@pytest.mark.skipif(not _cached(CLIP), reason="CLIP not cached")
def test_phrase_gradcam_matches_shortlist_and_signed_keeps_negatives():
    from PIL import Image, ImageDraw

    from ria.players import PhrasePlayer
    from ria.scorers import make_scorer
    from ria.shortlist import phrase_relevance_map
    sc = make_scorer(CLIP)
    img = Image.new("RGB", (sc.size, sc.size), "white")
    ImageDraw.Draw(img).ellipse((40, 120, 140, 200), fill=(30, 60, 220))
    phrase = PhrasePlayer("rel", "above", ["below", "next to"])
    text = "a red square above a blue circle"
    relu = B.gradcam(sc, img, text, phrase)
    assert np.allclose(relu, phrase_relevance_map(sc, img, text, phrase), atol=1e-12)
    signed = B.gradcam(sc, img, text, phrase, signed=True)
    assert np.allclose(np.maximum(signed, 0), relu, atol=1e-12) and (signed < 0).any()


SIGLIP = "google/siglip-base-patch16-224"


@pytest.mark.skipif(not _cached(SIGLIP), reason="SigLIP not cached")
def test_slow_sentencepiece_offsets_cover_phrase_tokens():
    from transformers import AutoProcessor
    fx = B.FixlipBaseline.__new__(B.FixlipBaseline)
    fx.processor, fx.is_clip = AutoProcessor.from_pretrained(SIGLIP), False
    text = "A mug to the right of a knife"
    ids, _ = B.token_offsets(fx.processor.tokenizer, text)
    toks = fx.processor.tokenizer.convert_ids_to_tokens(ids)
    players = fx.phrase_players(text, "to the right of")
    non_pad = [i for i, t in enumerate(ids) if t != fx.processor.tokenizer.pad_token_id]
    assert [toks[non_pad[p]] for p in players] == ["▁to", "▁the", "▁right", "▁of"]
    text2 = "A lemon on a table"
    ids2, _ = B.token_offsets(fx.processor.tokenizer, text2)
    toks2 = fx.processor.tokenizer.convert_ids_to_tokens(ids2)
    non_pad2 = [i for i, t in enumerate(ids2) if t != fx.processor.tokenizer.pad_token_id]
    assert [toks2[non_pad2[p]] for p in fx.phrase_players(text2, "on")] == ["▁on"]


@pytest.mark.skipif(not _cached(SIGLIP), reason="SigLIP not cached")
def test_siglip_pooling_rewrite_matches_library_and_chefer_runs():
    import torch
    from PIL import Image, ImageDraw

    from ria.players import PhrasePlayer
    from ria.scorers import make_scorer
    sc = make_scorer(SIGLIP)
    head = sc.model.vision_model.head
    x = torch.randn(2, sc.grid ** 2, sc.model.config.vision_config.hidden_size)
    with torch.no_grad():
        pooled, attn = B._pooling_attention(head, x)
        ref = head(x)
        assert torch.allclose((pooled + head.mlp(head.layernorm(pooled)))[:, 0], ref, atol=1e-5)
        assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)
    img = Image.new("RGB", (sc.size, sc.size), "white")
    ImageDraw.Draw(img).rectangle((60, 20, 160, 100), fill=(220, 30, 30))
    m = B.chefer(sc, img, "a red square above a circle", PhrasePlayer("rel", "above", ["below"]))
    assert m.shape == (sc.grid, sc.grid) and np.isfinite(m).all() and m.max() > 0
