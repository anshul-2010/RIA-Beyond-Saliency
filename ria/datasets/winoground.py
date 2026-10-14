"""Winoground (Thrush et al., CVPR 2022): 400 examples of two images x two captions that
use the same words in a different order.

LICENCE: Meta Images Research License. The data must stay on this machine: it may not be
hosted on or transferred to third parties (sec. 3c(2)), so never upload it to Kaggle or
commit it. Evaluation-only research use; no analysis or labelling of people's attributes
(sec. 3c(7-8)); up to 100 images in publications with Getty attribution (sec. 2c, 3a).

Expected location: PhD_Apply/winoground (the Hugging Face snapshot), images read straight
from data/images.zip without extracting.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT.parent / "winoground"


@dataclass
class WinoExample:
    id: int
    captions: tuple[str, str]
    images: tuple[str, str]      # names inside images.zip, without extension
    tag: str
    collapsed_tag: str           # Relation / Object / Both
    num_main_preds: int
    root: Path

    def image(self, i: int) -> Image.Image:
        return _read_image(str(self.root), self.images[i])


@lru_cache(maxsize=4)
def _zip(root: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(Path(root) / "data" / "images.zip")


def _read_image(root: str, name: str) -> Image.Image:
    with _zip(root).open(f"images/{name}.png") as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def load(root: Path = DEFAULT_ROOT) -> list[WinoExample]:
    root = Path(root)
    path = root / "data" / "examples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Winoground not found at {root} (expected data/examples.jsonl)")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        out.append(WinoExample(e["id"], (e["caption_0"], e["caption_1"]), (e["image_0"], e["image_1"]),
                               e["tag"], e["collapsed_tag"], e["num_main_preds"], root))
    return out


def scores_to_metrics(s: dict[tuple[int, int], float]) -> dict[str, bool]:
    """Winoground metrics from s[(caption, image)] for one example."""
    text = s[(0, 0)] > s[(1, 0)] and s[(1, 1)] > s[(0, 1)]
    image = s[(0, 0)] > s[(0, 1)] and s[(1, 1)] > s[(1, 0)]
    return {"text": text, "image": image, "group": text and image}
