"""Compositional hard-negative benchmarks for E3 (T11): SugarCrepe and ARO VG-Relation.

SugarCrepe (Hsieh et al., NeurIPS 2023): COCO val2017 images; each item has a true
caption and one hard negative. `replace_rel` swaps the relation phrase. Annotations
from github.com/RAIVNLab/sugar-crepe (data/*.json); images fetched one at a time from
images.cocodataset.org and cached under data/coco/val2017.

ARO VG-Relation (Yuksekgonul et al., ICLR 2023): Visual Genome crops; the false caption
swaps subject and object ("the shirt is to the left of the door"). Annotation JSON
from the ARO / What's Up release; images fetched from Stanford's Visual Genome host
and cached under data/vg/images, then cropped to the item's box.
"""

from __future__ import annotations

import difflib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SUGARCREPE_ROOT = REPO_ROOT / "data" / "sugarcrepe"
COCO_DIR = REPO_ROOT / "data" / "coco" / "val2017"
ARO_JSON = REPO_ROOT / "data" / "aro" / "visual_genome_relation.json"
VG_DIR = REPO_ROOT / "data" / "vg" / "images"

COCO_URL = "http://images.cocodataset.org/val2017/{name}"
VG_URLS = ("https://cs.stanford.edu/people/rak248/VG_100K_2/{name}", "https://cs.stanford.edu/people/rak248/VG_100K/{name}")
SUGARCREPE_URL = "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/{split}.json"
SUGARCREPE_SPLITS = ("replace_rel", "replace_obj", "replace_att", "swap_obj", "swap_att", "add_obj", "add_att")


def fetch(urls, dest: Path, timeout: int = 60) -> Path:
    """Download the first URL that works into `dest` (cached)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in ([urls] if isinstance(urls, str) else urls):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except Exception as e:                    # try the next host
            errors.append(f"{url}: {e}")
    raise FileNotFoundError("could not fetch " + "; ".join(errors))


def changed_span(true: str, foil: str) -> tuple[str, str]:
    """Words that differ between two captions: ('drives down', 'is stuck in')."""
    a, b = true.split(), foil.split()
    norm = lambda ws: [w.lower().strip(".,;:!?\"'") for w in ws]       # compare case- and punctuation-free
    blocks = [op for op in difflib.SequenceMatcher(a=norm(a), b=norm(b), autojunk=False).get_opcodes()
              if op[0] != "equal"]
    if not blocks:
        raise ValueError(f"captions are identical: {true!r}")
    i1, i2 = blocks[0][1], blocks[-1][2]
    j1, j2 = blocks[0][3], blocks[-1][4]
    strip = lambda ws: " ".join(ws).strip(".,;:!?")
    return strip(a[i1:i2]), strip(b[j1:j2])


@dataclass
class HardNegative:
    dataset: str
    key: str
    caption: str
    foil: str
    span: str                 # changed phrase in the true caption
    foil_span: str            # what replaces it in the foil
    image_ref: str            # file name for the fetcher
    subj: str | None = None   # nouns, when the item states them
    obj: str | None = None
    relation: str | None = None
    crop: tuple[int, int, int, int] | None = None

    def image(self) -> Image.Image:
        if self.dataset.startswith("sugarcrepe"):
            path = fetch(COCO_URL.format(name=self.image_ref), COCO_DIR / self.image_ref)
        else:
            path = fetch([u.format(name=self.image_ref) for u in VG_URLS], VG_DIR / self.image_ref)
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.crop:
                x, y, w, h = self.crop
                im = im.crop((x, y, x + w, y + h))
            return im.copy()


def load_sugarcrepe(split: str = "replace_rel", root: Path = SUGARCREPE_ROOT) -> list[HardNegative]:
    path = Path(root) / f"{split}.json"
    if not path.exists():
        fetch(SUGARCREPE_URL.format(split=split), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for k, d in data.items():
        span, foil_span = changed_span(d["caption"], d["negative_caption"])
        out.append(HardNegative(f"sugarcrepe/{split}", f"{split}/{k}", d["caption"], d["negative_caption"],
                                span, foil_span, d["filename"]))
    return out


def _aro_parse(caption: str, relation: str) -> tuple[str, str] | None:
    m = re.match(r"^the (.+?) (?:is|are) " + re.escape(relation) + r" (?:the |a |an )?(.+)$", caption.strip())
    return (m.group(1), m.group(2)) if m else None


def load_aro_relation(path: Path = ARO_JSON) -> list[HardNegative]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for i, d in enumerate(data):
        rel = d["relation_name"]
        parsed = _aro_parse(d["true_caption"], rel)
        subj, obj = parsed if parsed else (None, None)
        out.append(HardNegative("aro/vg_relation", f"vg_relation/{i}", d["true_caption"], d["false_caption"],
                                span=d["true_caption"], foil_span=d["false_caption"], image_ref=d["image_path"],
                                subj=subj, obj=obj, relation=rel,
                                crop=(d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"])))
    return out
