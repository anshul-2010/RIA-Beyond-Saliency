"""Generative VLM scorer (Qwen2.5-VL) with the same game interface as the dual encoders (T18).

Score: log-odds that the model answers "Yes" to
    Does this caption correctly describe the image? Caption: "<T>". Answer yes or no.
computed as logit(Yes) - logit(No) for the first answer token.

The image is fed at a fixed 448 x 448, so the vision tower yields a 16 x 16 grid of
merged visual tokens (one per 28 x 28 px cell, row-major). Regions map onto that grid
with the same exclusive patch logic as the encoders (patch = 28, grid = 16).

Removal operators:
    blur, mean   pixel-level, as for the encoders
    token        the region's visual tokens are replaced by those of a mean-colour image
                 (the closest analogue of patch dropping; the LLM keeps every position)
Vision features are computed once per image version and reused for every caption.
"""

from __future__ import annotations

from typing import Dict, Literal, Sequence

import numpy as np
import torch
from PIL import Image

from .clip_scorer import ClipScorer
from .interactions import Subset
from .players import PhrasePlayer, RegionPlayer, make_disjoint

VlmOperator = Literal["token", "blur", "mean"]
PROMPT = 'Does this caption correctly describe the image? Caption: "{caption}". Answer yes or no.'


class QwenVLScorer(ClipScorer):
    n_prefix = 0

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str | None = None,
                 operator: VlmOperator = "token", patch_overlap: float = 0.25, batch_size: int = 8,
                 size: int = 448):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=size * size, max_pixels=size * size)
        vc = self.model.config.vision_config
        self.patch = vc.patch_size * vc.spatial_merge_size          # 28 px per merged visual token
        self.size, self.grid = size, size // self.patch
        self.device, self.operator = device, operator
        self.patch_overlap, self.batch_size = patch_overlap, batch_size
        ip = self.processor.image_processor
        self.mean = torch.tensor(ip.image_mean).view(3, 1, 1)
        self.std = torch.tensor(ip.image_std).view(3, 1, 1)
        tok = self.processor.tokenizer
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tok.encode("No", add_special_tokens=False)[0]
        self.image_token_id = self.model.config.image_token_id
        self._text_cache: Dict[str, dict] = {}
        # per-image caches, shared by with_operator() copies: visual tokens and Yes/No scores
        self._cache: dict = {"image": None, "vis": {}, "score": {}}

    # ---- inputs ---------------------------------------------------------------
    def _prompt_inputs(self, text: str, image: Image.Image) -> dict:
        """Tokenised prompt for `text`; independent of image content because the size is fixed."""
        if text not in self._text_cache:
            messages = [{"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": PROMPT.format(caption=text)}]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = self.processor(text=[prompt], images=[image], return_tensors="pt")
            n_img = int((enc["input_ids"] == self.image_token_id).sum())
            if n_img != self.grid * self.grid:
                raise ValueError(f"expected {self.grid ** 2} visual tokens, got {n_img}; is the image {self.size}px?")
            self._text_cache[text] = {k: enc[k].to(self.device) for k in ("input_ids", "attention_mask", "image_grid_thw")}
        return self._text_cache[text]

    @torch.no_grad()
    def visual_tokens(self, image: Image.Image) -> torch.Tensor:
        """(grid*grid, hidden) merged visual tokens, row-major over the 16 x 16 cell grid."""
        enc = self.processor.image_processor(images=[image], return_tensors="pt")
        pv = enc["pixel_values"].to(self.device, self.model.dtype)
        feats = self.model.model.get_image_features(pv, enc["image_grid_thw"].to(self.device))
        feats = torch.cat(list(feats)) if isinstance(feats, (list, tuple)) else feats
        return feats

    @torch.no_grad()
    def _yes_no(self, text: str, image: Image.Image, visuals: Sequence[torch.Tensor]) -> list[float]:
        """logit(Yes) - logit(No) for one caption and several visual-token sets."""
        enc = self._prompt_inputs(text, image)
        ids, mask, thw = enc["input_ids"], enc["attention_mask"], enc["image_grid_thw"]
        base = self.model.get_input_embeddings()(ids)                          # (1, L, H)
        pos, _ = self.model.model.get_rope_index(ids, thw, None, None, mask)  # (3, 1, L)
        img_pos = (ids[0] == self.image_token_id).nonzero().flatten()
        out = []
        for i in range(0, len(visuals), self.batch_size):
            chunk = visuals[i:i + self.batch_size]
            emb = base.expand(len(chunk), -1, -1).clone()
            for j, v in enumerate(chunk):
                emb[j, img_pos] = v.to(emb.dtype)
            logits = self.model(inputs_embeds=emb, attention_mask=mask.expand(len(chunk), -1),
                                position_ids=pos.expand(-1, len(chunk), -1)).logits[:, -1].float()
            out += (logits[:, self.yes_id] - logits[:, self.no_id]).tolist()
        return out

    def _image_cache(self, image) -> dict:
        if self._cache["image"] is not image:                 # new example: drop the old one's entries
            self._cache.update(image=image, vis={}, score={})
        return self._cache

    # ---- removal --------------------------------------------------------------
    def _visuals_for(self, image, keys, regions_by, drops) -> Dict[Subset, torch.Tensor]:
        cache = self._image_cache(image)["vis"]
        missing = [k for k in keys if (self.operator, k) not in cache]
        if missing:
            for k, v in self._compute_visuals(image, missing, regions_by, drops).items():
                cache[(self.operator, k)] = v
        return {k: cache[(self.operator, k)] for k in keys}

    def _compute_visuals(self, image, keys, regions_by, drops) -> Dict[Subset, torch.Tensor]:
        base = self.visual_tokens(image)
        if self.operator == "token":
            gray = Image.new("RGB", image.size, tuple(int(round(v * 255)) for v in self.mean.flatten().tolist()))
            blank = self.visual_tokens(gray)
            out = {}
            for k in keys:
                v = base.clone()
                for n in k:
                    v[drops[n].to(v.device)] = blank[drops[n].to(v.device)]
                out[k] = v
            return out
        out = {}
        for k in keys:
            m = np.zeros((self.size, self.size), dtype=bool)
            for n in k:
                m |= regions_by[n].mask
            out[k] = self.visual_tokens(self.removed_image(image, m)) if m.any() else base
        return out

    # ---- games ----------------------------------------------------------------
    def game(self, image: Image.Image, text: str, regions: Sequence[RegionPlayer], phrases: Sequence[PhrasePlayer]):
        regions = make_disjoint(regions) if regions else []
        regions_by = {r.name: r for r in regions}
        drops = self.exclusive_patch_drops(regions) if regions else {}
        phrases_by = {p.name: p for p in phrases}

        def value_fn(subsets: Sequence[Subset]) -> Dict[Subset, float]:
            img_keys = list(dict.fromkeys(frozenset(n for n in s if n in regions_by) for s in subsets))
            visuals = self._visuals_for(image, img_keys, regions_by, drops)
            texts = {}
            for s in subsets:
                tp = frozenset(n for n in s if n in phrases_by)
                texts[tp] = self._texts(text, [phrases_by[n] for n in sorted(tp)])
            cache = self._image_cache(image)["score"]
            scores = {}
            for t in {t for ts in texts.values() for t in ts}:
                todo = [k for k in img_keys if (self.operator, t, k) not in cache]
                if todo:
                    for k, v in zip(todo, self._yes_no(t, image, [visuals[k] for k in todo])):
                        cache[(self.operator, t, k)] = v
                scores[t] = {k: cache[(self.operator, t, k)] for k in img_keys}
            out = {}
            for s in subsets:
                rm = frozenset(n for n in s if n in regions_by)
                tp = frozenset(n for n in s if n in phrases_by)
                out[s] = float(np.mean([scores[t][rm] for t in texts[tp]]))
            return out

        return value_fn

    def contrastive_game(self, image, text, foils, regions):
        if not foils:
            raise ValueError("contrastive_game needs at least one foil caption")
        g_true = self.game(image, text, regions, [])
        g_foils = [self.game(image, f, regions, []) for f in foils]

        def value_fn(subsets):
            t = g_true(subsets)
            fs = [g(subsets) for g in g_foils]
            return {s: t[s] - float(np.mean([f[s] for f in fs])) for s in subsets}

        return value_fn

    def method_games(self, image, text, regions, phrase, operators=("token", "blur", "mean")):
        return super().method_games(image, text, regions, phrase, operators)

    def encode_image(self, *a, **k):
        raise NotImplementedError("QwenVLScorer has no image embedding; use game() / margin()")

    def encode_text(self, *a, **k):
        raise NotImplementedError("QwenVLScorer has no text embedding; use game() / margin()")
