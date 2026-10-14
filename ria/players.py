"""Players: image regions and text phrases that can be removed from an input."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class RegionPlayer:
    """An image region given as a boolean mask in the model's input frame (H x W)."""
    name: str
    mask: np.ndarray

    @classmethod
    def from_box(cls, name: str, box: Sequence[float], size: int) -> "RegionPlayer":
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        m = np.zeros((size, size), dtype=bool)
        m[max(y0, 0):min(y1, size), max(x0, 0):min(x1, size)] = True
        return cls(name, m)

    @classmethod
    def complement(cls, name: str, others: Sequence["RegionPlayer"]) -> "RegionPlayer":
        """Context region: everything outside the given regions."""
        union = np.zeros_like(others[0].mask)
        for o in others:
            union |= o.mask
        return cls(name, ~union)


def make_disjoint(regions: Sequence[RegionPlayer]) -> list[RegionPlayer]:
    """Remove overlaps, smaller regions first: a cup inside a table's box stays the cup's.

    Removing one player must not remove another, or the interaction terms mix them up.
    Order of the returned list matches the input.
    """
    claimed = np.zeros_like(regions[0].mask)
    out = {}
    for r in sorted(regions, key=lambda r: int(r.mask.sum())):
        out[r.name] = RegionPlayer(r.name, r.mask & ~claimed)
        claimed |= r.mask
    return [out[r.name] for r in regions]


@dataclass
class PhrasePlayer:
    """A text span removed by substitution; scores are averaged over substitutes.

    `span` is the exact substring of the caption (first occurrence is replaced).
    Substitutes keep the sentence grammatical, e.g. "above" -> ["below", "next to"].
    """
    name: str
    span: str
    substitutes: list[str] = field(default_factory=lambda: ["something"])

    def apply(self, text: str, substitute: str) -> str:
        """Replace the first whole-word occurrence of the span ("on" never matches "lemon")."""
        pattern = re.compile(r"(?<!\w)" + re.escape(self.span) + r"(?!\w)")
        if not pattern.search(text):
            raise ValueError(f"span {self.span!r} not found in {text!r}")
        return pattern.sub(lambda _: substitute, text, count=1)
