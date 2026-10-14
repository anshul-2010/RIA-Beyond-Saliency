"""E3 loaders (T11) and relation-argument extraction; no network needed."""

import pytest

from ria.datasets.compositional import ARO_JSON, SUGARCREPE_ROOT, _aro_parse, changed_span


@pytest.mark.parametrize("true,foil,span,foil_span", [
    ("A zebra is standing in an open field.", "A zebra is running across an open field.", "standing in", "running across"),
    ("A cat on a mat.", "A cat under a mat.", "on", "under"),
    ("THERE ARE WOMEN LAUGHING UNDER THE UMBRELLA", "There are women laughing next to the umbrella", "UNDER", "next to"),
])
def test_changed_span(true, foil, span, foil_span):
    assert changed_span(true, foil) == (span, foil_span)


def test_changed_span_rejects_identical():
    with pytest.raises(ValueError):
        changed_span("a dog", "A dog.")


def test_aro_parse():
    assert _aro_parse("the door is to the left of the shirt", "to the left of") == ("door", "shirt")
    assert _aro_parse("the man is wearing the shirt", "wearing") == ("man", "shirt")
    assert _aro_parse("the man is wearing the shirt", "on") is None


@pytest.mark.skipif(not (SUGARCREPE_ROOT / "replace_rel.json").exists(), reason="SugarCrepe not downloaded")
def test_sugarcrepe_replace_rel_loads():
    from ria.datasets.compositional import load_sugarcrepe
    items = load_sugarcrepe("replace_rel")
    assert len(items) == 1406
    assert sum(not x.span for x in items) == 8          # pure insertions: nothing replaced in the true caption


@pytest.mark.skipif(not ARO_JSON.exists(), reason="ARO annotations not downloaded")
def test_aro_all_captions_parse():
    from ria.datasets.compositional import load_aro_relation
    items = load_aro_relation()
    assert len(items) == 23937 and all(x.subj and x.obj for x in items)


def test_relation_args_dependency_cases():
    pytest.importorskip("spacy")
    from ria.text import relation_args
    try:
        assert relation_args("a black puppy playing with a hand as it stands in a black suitcase", "stands in") == ("puppy", "suitcase")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    assert relation_args("Scissors with orange handles are in a container with pens.", "in") == ("scissors", "container")
    assert relation_args("The wooden park benches are painted dark purple.", "painted dark purple") is None


def test_relation_args_resolves_prepositional_and_quantity_nouns():
    pytest.importorskip("spacy")
    from ria.text import relation_args
    try:
        assert relation_args("A man standing in front of a fruit stand.", "standing in front of") == ("man", "fruit stand")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    assert relation_args("A bunch of pottery sitting on a sidewalk", "sitting") == ("pottery", "sidewalk")
    assert relation_args("A zebra standing on top of a grass covered field.", "standing on top of")[1] == "field"


def test_phrase_type():
    pytest.importorskip("spacy")
    from ria.text import phrase_type
    try:
        assert phrase_type("A cat on a mat.", "on") == "preposition"
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    assert phrase_type("A man holding an umbrella.", "holding") == "verb"
    assert phrase_type("A cup to the left of a plate.", "to the left of") == "preposition"
    assert phrase_type("A man standing in front of a stand.", "standing in front of") == "verb"
