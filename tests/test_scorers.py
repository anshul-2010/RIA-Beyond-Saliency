"""Scorer forward passes match the library and implement true token removal (T17)."""

import os

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "config.json"), str)
    except Exception:
        return False


MODELS = [m for m in ("openai/clip-vit-base-patch32", "google/siglip-base-patch16-224") if _cached(m)]


@pytest.fixture(scope="module", params=MODELS or [pytest.param(None, marks=pytest.mark.skip("no model cached"))])
def scorer(request):
    from ria.scorers import make_scorer
    return make_scorer(request.param)


def _pixels(scorer, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, scorer.size, scorer.size, generator=g)


@torch.no_grad()
def test_full_image_matches_library_features(scorer):
    px = _pixels(scorer)
    ref = torch.nn.functional.normalize(scorer.model.get_image_features(pixel_values=px), dim=-1)
    assert torch.allclose(scorer.encode_image(px), ref, atol=1e-5)
    keep_all = torch.ones(2, scorer.grid ** 2, dtype=torch.bool)
    assert torch.allclose(scorer.encode_image(px, keep_all), ref, atol=1e-5)


@torch.no_grad()
def test_masking_equals_deleting_tokens(scorer):
    """Hiding patch tokens with the attention mask == running the model without them."""
    px = _pixels(scorer, seed=1)[:1]
    n = scorer.grid ** 2
    keep = torch.ones(1, n, dtype=torch.bool)
    keep[0, 10:40] = False
    masked = scorer.encode_image(px, keep)

    vm = scorer.model.vision_model
    h = vm.embeddings(px)                                       # position embeddings already added
    if hasattr(vm, "pre_layrnorm"):
        h = vm.pre_layrnorm(h)
    idx = torch.cat([torch.arange(scorer.n_prefix), scorer.n_prefix + torch.nonzero(keep[0]).flatten()])
    out = vm.encoder(inputs_embeds=h[:, idx])
    last = out[0] if isinstance(out, tuple) else out.last_hidden_state
    if scorer.n_prefix:                                         # CLIP: CLS token -> projection
        deleted = scorer.model.visual_projection(vm.post_layernorm(last[:, 0]))
    else:                                                       # SigLIP: attention-pooling head
        deleted = vm.head(vm.post_layernorm(last))
    deleted = torch.nn.functional.normalize(deleted, dim=-1)
    assert torch.allclose(masked, deleted, atol=1e-4)
    assert not torch.allclose(masked, scorer.encode_image(px), atol=1e-3)


@torch.no_grad()
def test_text_features_match_library(scorer):
    texts = ["a cup on a table", "a dog under a chair"]
    if scorer.n_prefix == 0:
        tok = scorer.tokenizer(texts, padding="max_length", max_length=scorer.max_text_len, return_tensors="pt")
    else:
        tok = scorer.tokenizer(texts, padding=True, return_tensors="pt")
    ref = torch.nn.functional.normalize(scorer.model.get_text_features(**tok), dim=-1)
    assert torch.allclose(scorer.encode_text(texts), ref, atol=1e-5)
