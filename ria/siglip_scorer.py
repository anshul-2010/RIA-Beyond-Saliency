"""SigLIP dual encoder with the same removal and game interface as ClipScorer.

Differences from CLIP handled here: no CLS token (n_prefix = 0), attention-pooling head
(dropped patches are hidden from the pooling probe too), no projection layer, mean/std
of 0.5, and text padded to the fixed length the model was trained with.
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoTokenizer, SiglipModel

from .clip_scorer import ClipScorer, Operator


class SiglipScorer(ClipScorer):
    n_prefix = 0

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str | None = None,
                 operator: Operator = "drop", patch_overlap: float = 0.25, batch_size: int = 16):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipModel.from_pretrained(model_name, attn_implementation="eager").to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        cfg = self.model.config.vision_config
        self.size, self.patch = cfg.image_size, cfg.patch_size
        self.grid = self.size // self.patch
        self.device, self.operator = device, operator
        self.patch_overlap, self.batch_size = patch_overlap, batch_size
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.max_text_len = self.model.config.text_config.max_position_embeddings

    def encode_image(self, pixels: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        vm = self.model.vision_model
        h = vm.embeddings(pixels.to(self.device))
        mask = self._attention_mask(keep, h.dtype) if keep is not None else None
        out = vm.encoder(inputs_embeds=h, attention_mask=mask)
        last = vm.post_layernorm(out[0] if isinstance(out, tuple) else out.last_hidden_state)
        head = vm.head
        probe = head.probe.repeat(last.shape[0], 1, 1)
        pad = ~self._token_keep(keep) if keep is not None else None       # True = ignore this token
        pooled = head.attention(probe, last, last, key_padding_mask=pad)[0]
        pooled = pooled + head.mlp(head.layernorm(pooled))
        return torch.nn.functional.normalize(pooled[:, 0], dim=-1)

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tok = self.tokenizer(list(texts), padding="max_length", max_length=self.max_text_len,
                             truncation=True, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.model.get_text_features(**tok), dim=-1)
