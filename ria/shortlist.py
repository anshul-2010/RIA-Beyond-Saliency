"""Phrase-conditioned gradient relevance, used to shortlist regions before interventions.

The target is the phrase-contrastive score s(I, T) - mean_sub s(I, T \\ p), so the
map shows where the image supports *this phrase* rather than the caption overall.
Relevance is Grad-CAM on tokens: ReLU(sum_d grad * activation) at the input of the
last vision block.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from .clip_scorer import ClipScorer
from .players import PhrasePlayer, RegionPlayer, make_disjoint


def phrase_relevance_map(scorer: ClipScorer, image, text: str, phrase: PhrasePlayer) -> np.ndarray:
    """Return a (grid x grid) relevance map for `phrase`; `image` is in the model frame."""
    captured = {}

    def hook(_module, args):
        h = args[0]
        h.retain_grad()
        captured["h"] = h

    layer = scorer.model.vision_model.encoder.layers[-1]
    handle = layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            t_full = scorer.encode_text([text])[0]
            t_sub = scorer.encode_text(scorer._texts(text, [phrase])).mean(0)
        with torch.enable_grad():
            img = scorer.encode_image(scorer.to_tensor(image)[None])[0]
            (img @ (t_full - t_sub)).backward()
    finally:
        handle.remove()
        scorer.model.zero_grad(set_to_none=True)
    n = scorer.n_prefix                                   # skip CLS-like prefix tokens
    h, g = captured["h"][0, n:], captured["h"].grad[0, n:]
    rel = torch.relu((g * h).sum(-1)).detach().cpu().numpy()
    return rel.reshape(scorer.grid, scorer.grid)


def rank_regions(scorer: ClipScorer, rel_map: np.ndarray, regions: Sequence[RegionPlayer]) -> Dict[str, float]:
    """Mean patch relevance inside each region, weighted by patch overlap; sorted high to low."""
    p = scorer.patch
    out = {}
    for r in make_disjoint(regions):
        frac = r.mask.reshape(scorer.grid, p, scorer.grid, p).mean(axis=(1, 3))
        out[r.name] = float((frac * rel_map).sum() / max(frac.sum(), 1e-8))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
