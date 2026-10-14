"""What's Up (Kamath et al., EMNLP 2023): controlled spatial-relation images.

Subset A (controlled_images): on, under, left of, right of. 412 images.
Subset B (controlled_clevr): left of, right of, in front of, behind. 408 images.
Each entry has an image path and four caption options; option 0 is correct.

Expected layout (download with scripts/download_whatsup.sh or by hand):
    data/whatsup/controlled_images_dataset.json
    data/whatsup/controlled_clevr_dataset.json
    data/whatsup/controlled_images/*.jpeg
    data/whatsup/controlled_clevr/*.jpeg
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..players import PhrasePlayer, RegionPlayer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "whatsup"
DEFAULT_BOXES = REPO_ROOT / "data" / "boxes" / "whatsup_owlv2-base-patch16-ensemble.json"
SUBSETS = {"A": "controlled_images_dataset.json", "B": "controlled_clevr_dataset.json"}

OPPOSITE = {"on": "under", "under": "on", "to the left of": "to the right of", "to the right of": "to the left of",
            "in front of": "behind", "behind": "in front of"}
RELATIONS = ("to the left of", "to the right of", "in front of", "behind", "under", "on")
_CAPTION = re.compile(r"^an? (?P<subj>.+?) (?P<rel>" + "|".join(map(re.escape, RELATIONS)) + r") an? (?P<obj>.+)$",
                      re.IGNORECASE)


@dataclass(frozen=True)
class Caption:
    text: str
    subj: str
    rel: str
    obj: str


def parse_caption(text: str) -> Caption:
    """'A beer bottle on a armchair' -> Caption(subj='beer bottle', rel='on', obj='armchair')."""
    m = _CAPTION.match(text.strip().rstrip("."))
    if not m:
        raise ValueError(f"unparsed What's Up caption: {text!r}")
    return Caption(text, m["subj"].lower(), m["rel"].lower(), m["obj"].lower())


@dataclass
class Example:
    subset: str
    image_path: Path
    captions: list[Caption]                       # index 0 is the true caption
    boxes: dict[str, list[float]] = field(default_factory=dict)   # noun -> [x0, y0, x1, y1], original pixels
    box_scores: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.subset}/{self.image_path.name}"

    @property
    def true(self) -> Caption:
        return self.captions[0]

    @property
    def foils(self) -> list[Caption]:
        return self.captions[1:]

    @property
    def pair_key(self) -> tuple[str, str, str]:
        """Same subset and objects -> same key; images in a group differ only in the relation."""
        return (self.subset, self.true.subj, self.true.obj)


def load(subset: str = "A", root: Path = DEFAULT_ROOT, boxes: Path | None = DEFAULT_BOXES,
         require_boxes: bool = False) -> list[Example]:
    """Load one subset; attach cached detector boxes when the cache exists."""
    root = Path(root)
    entries = json.loads((root / SUBSETS[subset]).read_text(encoding="utf-8"))
    cache = json.loads(Path(boxes).read_text(encoding="utf-8")) if boxes and Path(boxes).exists() else {}
    if require_boxes and not cache:
        raise FileNotFoundError(f"no box cache at {boxes}; run scripts/detect_boxes.py (T08)")
    out = []
    for e in entries:
        rel_path = Path(e["image_path"])
        if rel_path.parts and rel_path.parts[0] == "data":
            rel_path = Path(*rel_path.parts[1:])
        ex = Example(subset, root / rel_path, [parse_caption(c) for c in e["caption_options"]])
        rec = cache.get(ex.key)
        if rec:
            ex.boxes = {n: b["box"] for n, b in rec["boxes"].items()}
            ex.box_scores = {n: b["score"] for n, b in rec["boxes"].items()}
            ex.flags = list(rec["flags"])
        elif require_boxes:
            raise KeyError(f"no boxes cached for {ex.key}")
        out.append(ex)
    return out


def group_pairs(examples: list[Example]) -> dict[tuple[str, str, str], list[Example]]:
    """Minimal-pair groups: same objects photographed in two or more relations."""
    groups: dict[tuple[str, str, str], list[Example]] = defaultdict(list)
    for ex in examples:
        groups[ex.pair_key].append(ex)
    return {k: sorted(v, key=lambda e: e.true.rel) for k, v in groups.items() if len({e.true.rel for e in v}) > 1}


@dataclass
class ModelInput:
    """One example in the model's square input frame, ready for a scorer's game."""
    example: Example
    image: Image.Image                 # letterboxed to size x size
    boxes: dict[str, list[float]]      # "subj" / "obj" boxes in the frame
    regions: list[RegionPlayer]        # subj, obj, context (everything else)
    phrase: PhrasePlayer               # the relation, substitutes = the subset's other relations
    caption: str                       # the true caption

    def caption_with(self, relation: str) -> str:
        return self.phrase.apply(self.caption, relation)


def to_model_input(ex: Example, size: int) -> ModelInput:
    """Letterbox the image to `size` and map boxes into that frame (see clip_scorer.letterbox)."""
    from ..clip_scorer import box_to_frame, letterbox

    if not ex.boxes:
        raise ValueError(f"{ex.key} has no boxes; load with a box cache (T08)")
    with Image.open(ex.image_path) as im:
        orig_size = im.size
        frame = letterbox(im, size)
    boxes = {"subj": box_to_frame(ex.boxes[ex.true.subj], orig_size, size),
             "obj": box_to_frame(ex.boxes[ex.true.obj], orig_size, size)}
    subj = RegionPlayer.from_box("subj", boxes["subj"], size)
    obj = RegionPlayer.from_box("obj", boxes["obj"], size)
    ctx = RegionPlayer.complement("context", [subj, obj])
    phrase = PhrasePlayer("rel", ex.true.rel, [c.rel for c in ex.foils])
    return ModelInput(ex, frame, boxes, [subj, obj, ctx], phrase, ex.true.text)
