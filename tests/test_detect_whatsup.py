import pytest

from ria.datasets.whatsup import DEFAULT_ROOT, SUBSETS, load, parse_caption
from ria.detect import OwlDetector, iou


@pytest.mark.parametrize("text,subj,rel,obj", [
    ("A beer bottle on a armchair", "beer bottle", "on", "armchair"),
    ("A mug to the right of a knife", "mug", "to the right of", "knife"),
    ("A bowl in front of a cup", "bowl", "in front of", "cup"),
    ("A wineglass under a table", "wineglass", "under", "table"),
    ("An apple behind a orange", "apple", "behind", "orange"),
])
def test_parse_caption(text, subj, rel, obj):
    c = parse_caption(text)
    assert (c.subj, c.rel, c.obj) == (subj, rel, obj)


def test_parse_caption_rejects_other_text():
    with pytest.raises(ValueError):
        parse_caption("two dogs playing")


@pytest.mark.skipif(not (DEFAULT_ROOT / SUBSETS["A"]).exists(), reason="What's Up annotations not downloaded")
def test_all_whatsup_captions_parse_and_share_objects():
    for subset in SUBSETS:
        exs = load(subset)
        assert len(exs) > 400
        for ex in exs:
            objs = {(c.subj, c.obj) for c in ex.captions}
            assert len(objs) == 1, ex.key            # every option names the same two objects


def test_iou():
    assert iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    assert abs(iou([0, 0, 2, 2], [1, 0, 3, 2]) - 1 / 3) < 1e-9


class FakeDetector(OwlDetector):
    """Bypasses model loading; returns fixed candidates."""
    def __init__(self, cands):
        self.cands, self.same_box_iou, self.flag_score, self.model_name = cands, 0.7, 0.2, "fake"

    def candidates(self, image, nouns):
        return self.cands


class _Img:
    size = (100, 100)


def test_same_box_is_reassigned_to_weaker_noun():
    det = FakeDetector({
        "cup": [{"box": [10, 10, 50, 50], "score": 0.9}],
        "table": [{"box": [11, 11, 50, 50], "score": 0.4}, {"box": [0, 40, 100, 100], "score": 0.3}],
    })
    res = det.detect(_Img(), ["cup", "table"])
    assert res["boxes"]["table"]["box"] == [0, 40, 100, 100]
    assert "reassigned:table" in res["flags"]


def test_low_scores_are_flagged():
    det = FakeDetector({"cup": [{"box": [0, 0, 10, 10], "score": 0.05}],
                        "plate": [{"box": [50, 50, 90, 90], "score": 0.6}]})
    assert det.detect(_Img(), ["cup", "plate"])["flags"] == ["low_score:cup"]
