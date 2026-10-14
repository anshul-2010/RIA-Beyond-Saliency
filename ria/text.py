"""Caption parsing helpers (start of T06): the two nouns a relation phrase connects.

Captions in hard-negative benchmarks are often ungrammatical, so noun chunks from the
dependency parse are unreliable ("red striped bus drives"). We use part-of-speech tags
instead: the subject is the nearest noun before the relation span, the object the
nearest noun after it, each extended over preceding compound nouns ("skate board").
"""

from __future__ import annotations

import re
from functools import lru_cache

NOUN_TAGS = {"NOUN", "PROPN"}
# "in front of X", "on top of X": the real argument is X
PREP_NOUNS = {"front", "top", "side", "middle", "back", "bottom", "edge", "center", "centre", "corner", "rest"}
# "a bunch of pottery", "a herd of cattle": the visible thing is the "of" object
QUANTITY_NOUNS = {"bunch", "pile", "group", "herd", "flock", "couple", "pair", "trio", "lot", "number", "few",
                  "set", "stack", "row", "collection", "assortment", "crowd", "team", "variety", "slice", "slices",
                  "piece", "pieces", "plate", "bowl", "box", "groups", "kind", "type", "line"}


def _is_noun(t) -> bool:
    """Noun by both coarse and fine tag: excludes gerunds the tagger calls NOUN ("riding")."""
    return t.pos_ in NOUN_TAGS and t.tag_.startswith("NN")


def _of_object(t):
    """X in "<t> of X", if any."""
    for c in t.children:
        if c.dep_ == "prep" and c.lower_ == "of":
            objs = [g for g in c.children if g.dep_ == "pobj" and _is_noun(g)]
            if objs:
                return objs[0]
    return None


def _resolve(t):
    """Follow "front of X" / "bunch of X" to X (repeatedly)."""
    for _ in range(3):
        if t.lower_ in PREP_NOUNS or t.lower_ in QUANTITY_NOUNS:
            nxt = _of_object(t)
            if nxt is None:
                break
            t = nxt
        else:
            break
    return t


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def _noun_phrase(doc, i: int) -> str:
    """Noun at token i with directly preceding compound nouns."""
    j = i
    while j > 0 and _is_noun(doc[j - 1]) and doc[j - 1].dep_ == "compound":
        j -= 1
    return doc[j:i + 1].text.lower()


def _subject_of(tok):
    """Noun the relation word is predicated of, following the dependency tree."""
    for _ in range(4):                                       # climb through aux / conj / xcomp chains
        subj = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass") and _is_noun(c)]
        if subj:
            return subj[0]
        if tok.dep_ in ("acl", "relcl", "prep", "amod") and _is_noun(tok.head):
            return tok.head                                  # "a man sitting ...", "a cup on ..."
        if tok.head is tok:
            return None
        tok = tok.head
    return None


def _object_of(span_toks):
    """First noun governed by the span (verb object or prepositional object)."""
    for t in span_toks:
        for c in t.children:
            if c.dep_ in ("dobj", "pobj", "attr", "dative", "obj") and _is_noun(c):
                return c
            if c.dep_ == "prep":                             # verb + preposition outside the span
                objs = [g for g in c.children if g.dep_ == "pobj" and _is_noun(g)]
                if objs:
                    return objs[0]
    return None


def relation_args(caption: str, span: str) -> tuple[str, str] | None:
    """(subject noun, object noun) for the first whole-word occurrence of `span`, or None.

    Dependency tree first; nearest noun on each side as the fallback.
    """
    m = re.search(r"(?<!\w)" + re.escape(span) + r"(?!\w)", caption)
    if not m:
        return None
    doc = _nlp()(caption)
    span_toks = [t for t in doc if t.idx >= m.start() and t.idx + len(t) <= m.end()]
    if not span_toks:
        return None
    subj = _subject_of(span_toks[0])
    obj = _object_of(span_toks)
    if subj is None:
        before = [t for t in doc if t.idx + len(t) <= m.start() and _is_noun(t)]
        subj = before[-1] if before else None
    if obj is None:
        after = [t for t in doc if t.idx >= m.end() and _is_noun(t)]
        obj = after[0] if after else None
    if subj is None or obj is None:
        return None
    subj, obj = _resolve(subj), _resolve(obj)
    if subj.i == obj.i:
        return None
    while obj.i + 1 < len(doc) and _is_noun(doc[obj.i + 1]) and obj.dep_ == "compound":
        obj = doc[obj.i + 1]                                 # "skate board": take the head of the compound
    return _noun_phrase(doc, subj.i), _noun_phrase(doc, obj.i)


PHRASE_TYPES = ("preposition", "verb", "attribute", "noun", "other")


def phrase_type(caption: str, span: str) -> str:
    """Linguistic category of a relation span for E6, from the POS of its first content word.

    preposition: "on", "next to", "in front of"; verb: "holding", "sitting on";
    attribute: adjectives; noun: noun-headed spans; other: anything else.
    """
    m = re.search(r"(?<!\w)" + re.escape(span) + r"(?!\w)", caption)
    if not m:
        return "other"
    doc = _nlp()(caption)
    toks = [t for t in doc if t.idx >= m.start() and t.idx + len(t) <= m.end() and not t.is_punct
            and t.pos_ not in ("DET", "AUX", "PART")]
    if not toks:
        return "other"
    head = toks[0]
    if head.pos_ == "ADP" or (head.pos_ == "ADV" and len(toks) > 1 and toks[-1].pos_ == "ADP"):
        return "preposition"
    if head.pos_ == "VERB":
        return "verb"
    if head.pos_ == "ADJ":
        return "attribute"
    if head.pos_ in NOUN_TAGS:
        return "noun"
    return "other"
