"""QwenVLScorer game logic with the model stubbed out (no 3B download needed)."""

import numpy as np
import torch
from PIL import Image

from ria.interactions import powerset, relational_triple
from ria.players import PhrasePlayer, RegionPlayer
from ria.qwen_scorer import QwenVLScorer


class StubQwen(QwenVLScorer):
    """Visual token = mean red value of its 28 px cell; score = text-weighted sum of tokens."""

    def __init__(self, operator="token"):
        self.size, self.patch, self.grid = 448, 28, 16
        self.operator, self.patch_overlap, self.batch_size, self.device = operator, 0.25, 8, "cpu"
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.std = torch.ones(3).view(3, 1, 1)
        self._cache = {"image": None, "vis": {}, "score": {}}
        self.vision_calls = self.llm_calls = 0

    def visual_tokens(self, image):
        self.vision_calls += 1
        a = np.asarray(image, dtype=np.float32)[..., 0] / 255.0
        return torch.from_numpy(a.reshape(16, 28, 16, 28).mean(axis=(1, 3)).reshape(-1, 1))

    def _yes_no(self, text, image, visuals):
        self.llm_calls += len(visuals)
        w = 2.0 if "above" in text else 1.0
        return [float(w * v.sum()) for v in visuals]


def _scene():
    img = Image.new("RGB", (448, 448), (0, 0, 0))
    img.paste((255, 0, 0), (56, 56, 168, 168))          # red square, 4 x 4 cells
    a = RegionPlayer.from_box("a", (56, 56, 168, 168), 448)
    b = RegionPlayer.from_box("b", (280, 280, 392, 392), 448)
    return img, a, b


def test_token_removal_replaces_only_region_tokens():
    img, a, b = _scene()
    sc = StubQwen("token")
    v = sc.game(img, "a square above a circle", [a, b], [])(powerset(["a", "b"]))
    # caption weight 2; 16 red cells = 1 each; blank image is mean grey, pixel 128 -> g per cell.
    # Removing a swaps its 16 cells 1 -> g; removing b swaps 16 black cells 0 -> g.
    g = 128 / 255
    assert np.isclose(v[frozenset()], 2 * 16.0)
    assert np.isclose(v[frozenset({"a"})], 2 * 16 * g)
    assert np.isclose(v[frozenset({"b"})], 2 * (16.0 + 16 * g))
    assert np.isclose(v[frozenset({"a", "b"})], 2 * (16 * g + 16 * g))


def test_caches_reuse_visuals_and_scores_across_substitute_games():
    img, a, b = _scene()
    sc = StubQwen("token")
    for sub in ("below", "next to"):
        g = sc.game(img, "a square above a circle", [a, b], [PhrasePlayer("p", "above", [sub])])
        r = relational_triple(g, "a", "b", "p")
        assert np.isfinite(r.triple)
    # 4 image versions + 1 blank image computed once; true caption scored once for all 4 versions
    assert sc.vision_calls == 2
    assert sc.llm_calls == 4 * 3                        # true caption + 2 substitutes, 4 versions each


def test_cache_resets_for_a_new_image():
    img, a, b = _scene()
    sc = StubQwen("token")
    sc.game(img, "x above y", [a], [])([frozenset()])
    sc.game(img.copy(), "x above y", [a], [])([frozenset()])
    assert sc.vision_calls == 4                          # base + blank, twice
