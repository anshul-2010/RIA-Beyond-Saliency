"""Winoground loader (data stays local; skipped when absent)."""

import pytest

from ria.datasets import winoground as W


def test_metrics_definition():
    good = {(0, 0): 2, (1, 0): 1, (0, 1): 0, (1, 1): 3}
    assert W.scores_to_metrics(good) == {"text": True, "image": True, "group": True}
    text_only = {(0, 0): 2, (1, 0): 1, (0, 1): 3, (1, 1): 4}
    assert W.scores_to_metrics(text_only) == {"text": True, "image": False, "group": False}


@pytest.mark.skipif(not (W.DEFAULT_ROOT / "data" / "examples.jsonl").exists(), reason="Winoground not present")
def test_load():
    exs = W.load()
    assert len(exs) == 400
    assert {e.collapsed_tag for e in exs} == {"Relation", "Object", "Both"}
    assert exs[0].image(0).mode == "RGB"
