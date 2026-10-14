"""Open-vocabulary object boxes (OWLv2 / OWL-ViT) for caption nouns, cached to JSON.

For each image we query the nouns we need and keep the single best box per noun.
If two nouns land on the same box (IoU above `same_box_iou`), the weaker noun takes
its best box that does not overlap the other. Detections below `flag_score` are
marked for manual review.

The cache maps an example key to
    {"size": [w, h], "boxes": {noun: {"box": [x0, y0, x1, y1], "score": s}}, "flags": [...]}
with boxes in original image pixels. It is written atomically every `save_every`
images, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection, OwlViTForObjectDetection


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class OwlDetector:
    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble", device: str | None = None,
                 same_box_iou: float = 0.7, flag_score: float = 0.2, top_k: int = 10):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        cls = Owlv2ForObjectDetection if "owlv2" in model_name else OwlViTForObjectDetection
        self.model = cls.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.is_v2 = "owlv2" in model_name
        self.same_box_iou, self.flag_score, self.top_k = same_box_iou, flag_score, top_k

    @torch.no_grad()
    def candidates(self, image: Image.Image, nouns: Sequence[str]) -> dict[str, list[dict]]:
        """Top-k boxes per noun, best first, in original image pixels."""
        image = image.convert("RGB")
        w, h = image.size
        queries = [[f"a photo of a {n}" for n in nouns]]
        inputs = self.processor(text=queries, images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        # OWLv2 pads to a square at the bottom/right, so boxes are relative to that square.
        side = max(w, h) if self.is_v2 else None
        target = torch.tensor([[side, side]] if self.is_v2 else [[h, w]], device=self.device)
        logits = out.logits[0].sigmoid()                        # (num_boxes, num_queries)
        boxes = self._to_pixels(out.pred_boxes[0], target[0])   # (num_boxes, 4)
        res = {}
        for qi, noun in enumerate(nouns):
            scores, idx = logits[:, qi].topk(min(self.top_k, logits.shape[0]))
            res[noun] = [{"box": [round(float(v), 1) for v in self._clip(boxes[i], w, h)], "score": round(float(s), 4)}
                         for s, i in zip(scores, idx)]
        return res

    @staticmethod
    def _to_pixels(cxcywh: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        cx, cy, bw, bh = cxcywh.unbind(-1)
        hh, ww = size[0], size[1]
        return torch.stack([(cx - bw / 2) * ww, (cy - bh / 2) * hh, (cx + bw / 2) * ww, (cy + bh / 2) * hh], -1)

    @staticmethod
    def _clip(b, w, h):
        return [min(max(float(b[0]), 0), w), min(max(float(b[1]), 0), h),
                min(max(float(b[2]), 0), w), min(max(float(b[3]), 0), h)]

    def detect(self, image: Image.Image, nouns: Sequence[str]) -> dict:
        cands = self.candidates(image, nouns)
        chosen = {n: cands[n][0] for n in nouns}
        flags = []
        if len(nouns) == 2:
            n1, n2 = nouns
            if iou(chosen[n1]["box"], chosen[n2]["box"]) > self.same_box_iou:
                weak, strong = (n1, n2) if chosen[n1]["score"] < chosen[n2]["score"] else (n2, n1)
                alt = next((c for c in cands[weak] if iou(c["box"], chosen[strong]["box"]) <= self.same_box_iou), None)
                if alt is not None:
                    chosen[weak] = alt
                    flags.append(f"reassigned:{weak}")
                else:
                    flags.append("same_box")
        flags += [f"low_score:{n}" for n in nouns if chosen[n]["score"] < self.flag_score]
        return {"size": list(image.size), "boxes": chosen, "flags": flags}


class BoxCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> dict:
        return self.data[key]

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def detect_all(detector: OwlDetector, items: Iterable[tuple[str, object, Sequence[str]]], cache: BoxCache,
               save_every: int = 20, on_item=None) -> int:
    """items: (key, image path or zero-argument image loader, nouns). Skips cached keys.

    Returns the number newly detected. Items whose loader raises FileNotFoundError are skipped.
    """
    n = 0
    for key, path, nouns in items:
        if key in cache:
            continue
        if callable(path):
            try:
                im = path()
            except FileNotFoundError:
                continue
            res = detector.detect(im, list(nouns))
        else:
            with Image.open(path) as im:
                res = detector.detect(im, list(nouns))
        res["model"] = detector.model_name
        cache.put(key, res)
        n += 1
        if on_item:
            on_item(key, res)
        if n % save_every == 0:
            cache.save()
    cache.save()
    return n
