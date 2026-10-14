"""Model inputs for any (image, caption, relation span, two object boxes) item."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from ..players import PhrasePlayer, RegionPlayer


@dataclass
class ItemInput:
    image: Image.Image                 # letterboxed to size x size
    boxes: dict[str, list[float]]      # "subj" / "obj" boxes in the frame
    regions: list[RegionPlayer]        # subj, obj, context
    phrase: PhrasePlayer
    caption: str


def item_input(image: Image.Image, caption: str, span: str, substitutes: Sequence[str],
               subj_box: Sequence[float], obj_box: Sequence[float], size: int) -> ItemInput:
    """Letterbox `image`, map boxes (original pixels) into the frame, build regions and the phrase."""
    from ..clip_scorer import box_to_frame, letterbox

    frame = letterbox(image, size)
    boxes = {"subj": box_to_frame(subj_box, image.size, size), "obj": box_to_frame(obj_box, image.size, size)}
    subj = RegionPlayer.from_box("subj", boxes["subj"], size)
    obj = RegionPlayer.from_box("obj", boxes["obj"], size)
    ctx = RegionPlayer.complement("context", [subj, obj])
    return ItemInput(frame, boxes, [subj, obj, ctx], PhrasePlayer("rel", span, list(substitutes)), caption)
