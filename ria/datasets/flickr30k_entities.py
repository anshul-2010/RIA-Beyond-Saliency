"""Flickr30k Entities (Plummer et al., ICCV 2015) for E1 phrase localisation (T14).

Annotations: github.com/BryanPlummer/flickr30k_entities (annotations.zip and split lists),
read straight from the zip. Each sentence marks phrases inline,
"[/EN#547/people Several climbers] are climbing [/EN#548/other the rock]", and the XML
gives each entity's boxes (an entity can have several; some have none or are "scene").

Images: Flickr30k, for non-commercial research use under the dataset's terms. The full
image zip (~4.4 GB) is fetched inside Kaggle runs from the Hugging Face mirror
nlphuji/flickr30k (`fetch_image_zip`); images are read lazily from it.
"""

from __future__ import annotations

import io
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "flickr30k_entities"
GITHUB = "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/{name}"
ANNOTATIONS_URL = "https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip"
PHRASE = re.compile(r"\[/EN#(\d+)/([^\s\]]+) ([^\]]+)\]")


@dataclass
class Phrase:
    key: str                  # "<image id>/<sentence index>/<entity id>"
    image_id: str
    sentence: str             # plain text, markup removed
    phrase: str
    start: int                # character span of the phrase in `sentence`
    end: int
    types: list[str]
    entity: str
    boxes: list[list[float]] = field(default_factory=list)   # x0, y0, x1, y1 in image pixels
    size: tuple[int, int] = (0, 0)


def ensure_annotations(root: Path = DEFAULT_ROOT) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "annotations.zip").exists():
        urllib.request.urlretrieve(ANNOTATIONS_URL, root / "annotations.zip")
    for split in ("train", "val", "test"):
        if not (root / f"{split}.txt").exists():
            urllib.request.urlretrieve(GITHUB.format(name=f"{split}.txt"), root / f"{split}.txt")
    return root


def parse_sentence(marked: str) -> tuple[str, list[tuple[str, list[str], str, int, int]]]:
    """Strip markup; return (plain sentence, [(entity, types, phrase, start, end), ...])."""
    plain, spans, pos = [], [], 0
    for m in PHRASE.finditer(marked):
        plain.append(marked[pos:m.start()])
        start = sum(len(p) for p in plain)
        text = m.group(3)
        plain.append(text)
        spans.append((m.group(1), m.group(2).split("/"), text, start, start + len(text)))
        pos = m.end()
    plain.append(marked[pos:])
    return "".join(plain), spans


def parse_boxes(xml_text: str) -> tuple[dict[str, list[list[float]]], tuple[int, int]]:
    root = ET.fromstring(xml_text)
    size = root.find("size")
    wh = (int(size.find("width").text), int(size.find("height").text)) if size is not None else (0, 0)
    boxes: dict[str, list[list[float]]] = {}
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        for name in obj.findall("name"):
            boxes.setdefault(name.text, []).append([float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")])
    return boxes, wh


def load(split: str = "test", root: Path = DEFAULT_ROOT) -> list[Phrase]:
    """Every phrase in `split` whose entity has at least one box (non-visual phrases dropped)."""
    root = ensure_annotations(root)
    ids = [l.strip() for l in (root / f"{split}.txt").read_text().splitlines() if l.strip()]
    z = zipfile.ZipFile(root / "annotations.zip")
    out = []
    for img in ids:
        boxes, wh = parse_boxes(z.read(f"Annotations/{img}.xml").decode("utf-8"))
        for si, line in enumerate(z.read(f"Sentences/{img}.txt").decode("utf-8").splitlines()):
            plain, spans = parse_sentence(line.strip())
            for ent, types, text, s, e in spans:
                if ent in boxes and "notvisual" not in types:
                    out.append(Phrase(f"{img}/{si}/{ent}", img, plain, text, s, e, types, ent, boxes[ent], wh))
    return out


def fetch_image_zip() -> Path:
    """Download the Flickr30k image zip from the Hugging Face mirror (for Kaggle runs)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("nlphuji/flickr30k", "flickr30k-images.zip", repo_type="dataset"))


@lru_cache(maxsize=2)
def _image_zip(path: str):
    z = zipfile.ZipFile(path)
    index = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".jpg")}
    return z, index


def image(image_id: str, zip_path: Path) -> Image.Image:
    z, index = _image_zip(str(zip_path))
    with z.open(index[f"{image_id}.jpg"]) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")
