import json

import numpy as np
import pytest
from PIL import Image

from ria.datasets import whatsup
from ria.players import PhrasePlayer


def test_phrase_replacement_matches_whole_words_only():
    p = PhrasePlayer("rel", "on", ["under"])
    assert p.apply("A lemon on a table", "under") == "A lemon under a table"
    with pytest.raises(ValueError):
        p.apply("A lemon near a table", "under")


@pytest.fixture
def tiny_whatsup(tmp_path):
    (tmp_path / "controlled_images").mkdir()
    entries, cache = [], {}
    for rel, fn in (("on", "cup_on_table"), ("under", "cup_under_table")):
        Image.new("RGB", (400, 200), "white").save(tmp_path / "controlled_images" / f"{fn}.jpeg")
        others = [r for r in ("on", "under", "to the left of", "to the right of") if r != rel]
        entries.append({"image_path": f"data/controlled_images/{fn}.jpeg",
                        "caption_options": [f"A cup {rel} a table"] + [f"A cup {r} a table" for r in others]})
        cache[f"A/{fn}.jpeg"] = {"size": [400, 200], "flags": [],
                                 "boxes": {"cup": {"box": [10, 20, 30, 60], "score": 0.9},
                                           "table": {"box": [0, 50, 400, 200], "score": 0.8}}}
    (tmp_path / "controlled_images_dataset.json").write_text(json.dumps(entries), encoding="utf-8")
    boxes = tmp_path / "boxes.json"
    boxes.write_text(json.dumps(cache), encoding="utf-8")
    return tmp_path, boxes


def test_load_attaches_boxes_and_groups_pairs(tiny_whatsup):
    root, boxes = tiny_whatsup
    exs = whatsup.load("A", root=root, boxes=boxes, require_boxes=True)
    assert exs[0].boxes["cup"] == [10, 20, 30, 60] and exs[0].box_scores["table"] == 0.8
    groups = whatsup.group_pairs(exs)
    assert list(groups) == [("A", "cup", "table")] and len(groups[("A", "cup", "table")]) == 2


def test_model_input_maps_boxes_into_letterboxed_frame(tiny_whatsup):
    root, boxes = tiny_whatsup
    mi = whatsup.to_model_input(whatsup.load("A", root=root, boxes=boxes)[0], size=200)
    # 400x200 image pads to a 400x400 square (100 px top and bottom), then scales by 0.5.
    assert mi.boxes["subj"] == [5.0, 60.0, 15.0, 80.0]
    assert mi.image.size == (200, 200)
    subj, obj, ctx = mi.regions
    assert not (ctx.mask & (subj.mask | obj.mask)).any()
    assert mi.caption_with("under") == mi.example.foils[0].text


def test_small_region_still_drops_a_patch():
    from ria.clip_scorer import ClipScorer
    s = ClipScorer.__new__(ClipScorer)          # no model needed for the patch logic
    s.grid, s.patch, s.patch_overlap = 14, 16, 0.25
    m = np.zeros((224, 224), dtype=bool)
    m[100:120, 50:54] = True                     # 4 px wide: under 25% of any patch
    keep = s.patch_keep(m)
    assert (~keep).sum() >= 1
    assert s.patch_keep(np.zeros_like(m)).all()          # empty region drops nothing


@pytest.mark.skipif(not whatsup.DEFAULT_BOXES.exists(), reason="box cache not present")
def test_real_data_every_group_has_all_four_relations():
    for subset in whatsup.SUBSETS:
        groups = whatsup.group_pairs(whatsup.load(subset, require_boxes=True))
        assert all(len({e.true.rel for e in g}) == 4 for g in groups.values())


def _geometry_scorer():
    from ria.clip_scorer import ClipScorer
    s = ClipScorer.__new__(ClipScorer)
    s.size, s.grid, s.patch, s.patch_overlap = 224, 14, 16, 0.25
    return s


def test_contained_object_survives_removing_its_container():
    from ria.players import RegionPlayer, make_disjoint
    s = _geometry_scorer()
    table = RegionPlayer.from_box("table", (32, 32, 192, 192), 224)
    cup = RegionPlayer.from_box("cup", (100, 100, 108, 130), 224)      # inside the table box
    ctx = RegionPlayer.complement("context", [table, cup])
    d_table, d_cup, d_ctx = make_disjoint([table, cup, ctx])
    assert not (d_table.mask & d_cup.mask).any() and d_cup.mask.sum() == cup.mask.sum()
    drops = s.exclusive_patch_drops([d_table, d_cup, d_ctx])
    assert drops["cup"].any()
    assert not (drops["table"] & drops["cup"]).any()
    assert not (drops["context"] & (drops["table"] | drops["cup"])).any()
