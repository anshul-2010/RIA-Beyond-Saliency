"""VSR, Visual Spatial Reasoning (Liu et al., TACL 2023), random split, for an E2 replication
on natural COCO photos (T10).

Captions are "The <subj> is <relation> the <obj>." with a true/false label; images are COCO
2017 (fetched one at a time from images.cocodataset.org). We use TRUE captions whose
relation has a clear opposite; the hard negative swaps in the opposite relation.
Data: huggingface.co/datasets/cambridgeltl/vsr_random (ungated), cached in data/vsr.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .compositional import fetch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "vsr"
HF = "https://huggingface.co/datasets/cambridgeltl/vsr_random/resolve/main/{split}.jsonl"
COCO_DIR = REPO_ROOT / "data" / "coco" / "train_val2017"

OPPOSITE = {}
for a, b in (("behind", "in front of"), ("on top of", "beneath"), ("at the right side of", "at the left side of"),
             ("above", "below"), ("left of", "right of"), ("under", "over"), ("facing", "facing away from"),
             ("far away from", "close to"), ("far from", "near")):
    OPPOSITE[a], OPPOSITE[b] = b, a
PAIRS = [("behind", "in front of"), ("on top of", "beneath"), ("at the right side of", "at the left side of"),
         ("above", "below"), ("left of", "right of"), ("under", "over"), ("facing", "facing away from"),
         ("far away from", "close to"), ("far from", "near")]


@dataclass
class VsrItem:
    key: str
    caption: str
    relation: str
    opposite: str
    subj: str
    obj: str
    image_name: str
    image_link: str

    @property
    def foil(self) -> str:
        return re.sub(r"(?<!\w)" + re.escape(self.relation) + r"(?!\w)", self.opposite, self.caption, count=1)

    def image(self) -> Image.Image:
        path = fetch(self.image_link, COCO_DIR / self.image_name)
        with Image.open(path) as im:
            return im.convert("RGB").copy()


def parse(caption: str, relation: str) -> tuple[str, str] | None:
    m = re.match(r"^the (.+?) (?:is|are) " + re.escape(relation) + r" (?:the |a |an )?(.+?)\.?$", caption.strip(), re.I)
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def load(splits=("dev", "test"), root: Path = DEFAULT_ROOT) -> list[VsrItem]:
    root = Path(root)
    out = []
    for split in splits:
        path = root / f"{split}.jsonl"
        if not path.exists():
            fetch(HF.format(split=split), path)
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            rel = r["relation"]
            if r["label"] != 1 or rel not in OPPOSITE:
                continue
            args = parse(r["caption"], rel)
            if not args:
                continue
            out.append(VsrItem(f"{split}/{i}", r["caption"].strip(), rel, OPPOSITE[rel], args[0], args[1],
                               r["image"], r["image_link"]))
    return out
