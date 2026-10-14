"""CLIP scorer with region removal by patch dropping, blurring or mean fill.

Images are letterboxed to a square and resized to the model resolution, so a box
in that frame maps directly onto the patch grid. Patch dropping masks the dropped
patch tokens out of every attention layer, which keeps the remaining inputs
in-distribution (no grey squares the model has never seen).
"""

from __future__ import annotations

import copy
from itertools import product
from typing import Dict, Literal, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import CLIPModel, CLIPTokenizer

from .interactions import Subset
from .players import PhrasePlayer, RegionPlayer, make_disjoint

Operator = Literal["drop", "blur", "mean"]


def letterbox(image: Image.Image, size: int, fill=(124, 116, 104)) -> Image.Image:
    """Pad to a square around the centre, then resize to size x size."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image.convert("RGB"), ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def box_to_frame(box: Sequence[float], orig_size: tuple[int, int], size: int) -> list[float]:
    """Map an (x0, y0, x1, y1) box in original pixels into the letterboxed frame."""
    w, h = orig_size
    side = max(w, h)
    ox, oy = (side - w) / 2, (side - h) / 2
    k = size / side
    x0, y0, x1, y1 = box
    return [(x0 + ox) * k, (y0 + oy) * k, (x1 + ox) * k, (y1 + oy) * k]


class ClipScorer:
    """CLIP dual encoder. Subclasses (e.g. SiglipScorer) override loading and the encoders;
    removal, games and contrastive scoring are shared."""

    n_prefix = 1        # tokens before the patch tokens (CLIP: one CLS token)

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: str | None = None,
                 operator: Operator = "drop", patch_overlap: float = 0.25, batch_size: int = 16):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name, attn_implementation="eager").to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        cfg = self.model.config.vision_config
        self.size, self.patch = cfg.image_size, cfg.patch_size
        self.grid = self.size // self.patch
        self.device, self.operator = device, operator
        self.patch_overlap, self.batch_size = patch_overlap, batch_size
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def with_operator(self, operator: Operator) -> "ClipScorer":
        """A view of this scorer (same loaded model) with a different removal operator."""
        other = copy.copy(self)
        other.operator = operator
        return other

    # ---- encoders -------------------------------------------------------------
    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (x - self.mean) / self.std

    def patch_keep(self, removed_mask: np.ndarray | None) -> torch.Tensor:
        """Boolean keep-vector over patches; a patch is dropped if overlap >= threshold.

        A small region may not cover `patch_overlap` of any patch; then the patches it
        overlaps most are dropped, so removing a non-empty region is never a no-op.
        """
        if removed_mask is None or not removed_mask.any():
            return torch.ones(self.grid * self.grid, dtype=torch.bool)
        p = self.patch
        frac = removed_mask.reshape(self.grid, p, self.grid, p).mean(axis=(1, 3))
        thresh = min(self.patch_overlap, float(frac.max()))
        return torch.from_numpy(frac < thresh).flatten()

    def exclusive_patch_drops(self, regions: Sequence[RegionPlayer]) -> Dict[str, torch.Tensor]:
        """Patches each region drops, with every patch owned by at most one region.

        Smaller regions claim patches first, so a mixed patch holding a small object and
        part of a large one stays with the small object. A region left with nothing keeps
        its single best patch even if shared, so removal is never a no-op.
        """
        claimed = torch.zeros(self.grid * self.grid, dtype=torch.bool)
        drops = {}
        for r in sorted(regions, key=lambda r: int(r.mask.sum())):
            want = ~self.patch_keep(r.mask)
            own = want & ~claimed
            if r.mask.any() and not own.any():
                p = self.patch
                frac = torch.from_numpy(r.mask.reshape(self.grid, p, self.grid, p).mean(axis=(1, 3))).flatten()
                own = torch.zeros_like(want)
                own[int(frac.argmax())] = True
            drops[r.name] = own
            claimed |= own
        return drops

    def encode_image(self, pixels: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        vm = self.model.vision_model
        h = vm.pre_layrnorm(vm.embeddings(pixels.to(self.device)))
        mask = self._attention_mask(keep, h.dtype) if keep is not None else None
        out = vm.encoder(inputs_embeds=h, attention_mask=mask)
        last = out[0] if isinstance(out, tuple) else out.last_hidden_state
        pooled = vm.post_layernorm(last[:, 0])
        return torch.nn.functional.normalize(self.model.visual_projection(pooled), dim=-1)

    def _token_keep(self, keep: torch.Tensor) -> torch.Tensor:
        """Patch keep-vector -> token keep-vector (prefix tokens always kept)."""
        prefix = torch.ones(keep.shape[0], self.n_prefix, dtype=torch.bool)
        return torch.cat([prefix, keep], dim=1).to(self.device)

    def _attention_mask(self, keep: torch.Tensor, dtype) -> torch.Tensor:
        """Additive (B, 1, T, T) mask hiding dropped patch tokens from every query."""
        full = self._token_keep(keep)
        add = torch.zeros(full.shape, dtype=dtype, device=self.device).masked_fill(~full, torch.finfo(dtype).min)
        t = full.shape[1]
        return add[:, None, None, :].expand(-1, 1, t, t)

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tok = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.model.get_text_features(**tok), dim=-1)

    # ---- removal --------------------------------------------------------------
    def removed_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if self.operator == "blur":
            other = image.filter(ImageFilter.GaussianBlur(radius=self.size / 16))
        else:  # mean fill with the model's normalisation mean (zero after normalising)
            other = Image.new("RGB", image.size, tuple(int(round(v * 255)) for v in self.mean.flatten().tolist()))
        return Image.composite(other, image, Image.fromarray(mask.astype(np.uint8) * 255))

    # ---- game -----------------------------------------------------------------
    def game(self, image: Image.Image, text: str, regions: Sequence[RegionPlayer],
             phrases: Sequence[PhrasePlayer]):
        """Return value_fn(subsets) -> {subset: score} for this (image, text) pair.

        `image` must already be in the model frame (see `letterbox`).
        """
        regions = make_disjoint(regions) if regions else []
        regions_by = {r.name: r for r in regions}
        drops = self.exclusive_patch_drops(regions) if regions else {}
        phrases_by = {p.name: p for p in phrases}

        @torch.no_grad()
        def value_fn(subsets: Sequence[Subset]) -> Dict[Subset, float]:
            img_keys, txt_keys = {}, {}
            for s in subsets:
                rm = frozenset(n for n in s if n in regions_by)
                tp = frozenset(n for n in s if n in phrases_by)
                img_keys[rm] = None
                txt_keys[tp] = self._texts(text, [phrases_by[n] for n in sorted(tp)])
            img_emb = self._image_embeddings(image, list(img_keys), regions_by, drops)
            all_texts = sorted({t for ts in txt_keys.values() for t in ts})
            txt_emb = dict(zip(all_texts, self.encode_text(all_texts)))
            out = {}
            for s in subsets:
                rm = frozenset(n for n in s if n in regions_by)
                tp = frozenset(n for n in s if n in phrases_by)
                sims = [float(img_emb[rm] @ txt_emb[t]) for t in txt_keys[tp]]
                out[s] = float(np.mean(sims))
            return out

        return value_fn

    def contrastive_game(self, image: Image.Image, text: str, foils: Sequence[str],
                         regions: Sequence[RegionPlayer]):
        """value_fn over image regions only: v(S) = s(I \\ S, T) - mean_f s(I \\ S, T'_f).

        The margin is positive when the model prefers the true caption. Foils are whole
        captions, so any hard negative works (relation swaps, SugarCrepe replacements).
        With a single foil that differs from T only in phrase p, the a-b interaction of
        this game equals the third-order term Gamma(a, b, p) with that foil as substitute.
        """
        if not foils:
            raise ValueError("contrastive_game needs at least one foil caption")
        regions = make_disjoint(regions) if regions else []
        regions_by = {r.name: r for r in regions}
        drops = self.exclusive_patch_drops(regions) if regions else {}

        @torch.no_grad()
        def value_fn(subsets: Sequence[Subset]) -> Dict[Subset, float]:
            unknown = {n for s in subsets for n in s} - set(regions_by)
            if unknown:
                raise KeyError(f"contrastive games have image players only; got {sorted(unknown)}")
            img_emb = self._image_embeddings(image, list(dict.fromkeys(subsets)), regions_by, drops)
            t_true, *t_foils = self.encode_text([text, *foils])
            foil_mean = torch.stack(t_foils).mean(0)
            return {s: float(img_emb[s] @ t_true - img_emb[s] @ foil_mean) for s in subsets}

        return value_fn

    def margin(self, image: Image.Image, text: str, foils: Sequence[str]) -> float:
        """s(I, T) - mean_f s(I, T'_f) on the full image."""
        return self.contrastive_game(image, text, foils, [])([frozenset()])[frozenset()]

    def method_games(self, image: Image.Image, text: str, regions: Sequence[RegionPlayer],
                     phrase: PhrasePlayer, operators: Sequence[Operator] = ("drop", "blur", "mean")):
        """One game per (operator, substitute) choice, keyed "operator/substitute".

        Feed the result to `interactions.robust_triple` to get the noise floor.
        """
        games = {}
        for op in operators:
            scorer = self.with_operator(op)
            for sub in phrase.substitutes:
                single = PhrasePlayer(phrase.name, phrase.span, [sub])
                games[f"{op}/{sub}"] = scorer.game(image, text, regions, [single])
        return games

    def _texts(self, text: str, removed: Sequence[PhrasePlayer]) -> list[str]:
        if not removed:
            return [text]
        texts = []
        for subs in product(*[p.substitutes for p in removed]):
            t = text
            for p, sub in zip(removed, subs):
                t = p.apply(t, sub)
            texts.append(t)
        return texts

    def _image_embeddings(self, image, keys, regions_by, drops) -> Dict[Subset, torch.Tensor]:
        base = self.to_tensor(image)
        masks = []
        for k in keys:
            m = np.zeros((self.size, self.size), dtype=bool)
            for n in k:
                m |= regions_by[n].mask
            masks.append(m)
        embs = []
        for i in range(0, len(keys), self.batch_size):
            chunk = masks[i:i + self.batch_size]
            if self.operator == "drop":
                pixels = base.unsqueeze(0).expand(len(chunk), -1, -1, -1)
                keep = torch.stack([self._keep_for(k, drops) for k in keys[i:i + self.batch_size]])
                embs.append(self.encode_image(pixels, keep))
            else:
                pixels = torch.stack([self.to_tensor(self.removed_image(image, m)) if m.any() else base for m in chunk])
                embs.append(self.encode_image(pixels))
        return dict(zip(keys, torch.cat(embs)))

    def _keep_for(self, removed: Subset, drops: Dict[str, torch.Tensor]) -> torch.Tensor:
        keep = torch.ones(self.grid * self.grid, dtype=torch.bool)
        for n in removed:
            keep &= ~drops[n]
        return keep
