"""Baselines for E1/E2 (T19), scored on the same regions as RIA.

Every method returns {region_name: {"mean": ..., "sum": ...}} over the patches each
region owns (`ClipScorer.exclusive_patch_drops`, the same assignment the interventions
use), so baselines and RIA see identical regions.

    gradcam            phrase-conditioned token Grad-CAM (1st order)
    gradcam_signed     the same without the ReLU (gradient x activation), keeps evidence against
    gradcam_caption    Grad-CAM on the whole-caption score (1st order, not phrase-specific)
    chefer             phrase-conditioned attention relevance, Chefer et al. 2021 (1st order)
    interaction2       local region x phrase interaction from removal games (2nd order)
    fixlip             official FIxLIP weighted-Banzhaf patch x token interactions,
                       summed over region patches x phrase tokens (2nd order)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from .clip_scorer import ClipScorer
from .interactions import robust_triple
from .players import PhrasePlayer, RegionPlayer, make_disjoint

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXLIP_DIR = REPO_ROOT / "third_party" / "fixlip"
RegionScores = Dict[str, Dict[str, float]]


def region_patch_sets(scorer: ClipScorer, regions: Sequence[RegionPlayer]) -> Dict[str, np.ndarray]:
    drops = scorer.exclusive_patch_drops(make_disjoint(regions))
    return {name: d.numpy() for name, d in drops.items()}


def aggregate(patch_scores: np.ndarray, patch_sets: Dict[str, np.ndarray]) -> RegionScores:
    flat = np.asarray(patch_scores, dtype=float).reshape(-1)
    return {n: {"mean": float(flat[m].mean()) if m.any() else 0.0, "sum": float(flat[m].sum())}
            for n, m in patch_sets.items()}


# ---- 1st order ------------------------------------------------------------------------

def _target(scorer: ClipScorer, text: str, phrase: PhrasePlayer | None) -> torch.Tensor:
    """Text-side target: the caption embedding, or caption minus its phrase-substituted versions."""
    with torch.no_grad():
        t_full = scorer.encode_text([text])[0]
        if phrase is None:
            return t_full
        return t_full - scorer.encode_text(scorer._texts(text, [phrase])).mean(0)


def gradcam(scorer: ClipScorer, image, text: str, phrase: PhrasePlayer | None, signed: bool = False) -> np.ndarray:
    """Token Grad-CAM at the input of the last vision block: ReLU(sum_d grad * activation).

    phrase=None targets the whole-caption score s(I, T); otherwise the phrase-contrastive
    score. signed=True drops the ReLU (gradient x activation), keeping evidence against.
    """
    target = _target(scorer, text, phrase)
    captured = {}

    def hook(_m, args):
        args[0].retain_grad()
        captured["h"] = args[0]

    handle = scorer.model.vision_model.encoder.layers[-1].register_forward_pre_hook(hook)
    try:
        with torch.enable_grad():
            (scorer.encode_image(scorer.to_tensor(image)[None])[0] @ target).backward()
    finally:
        handle.remove()
        scorer.model.zero_grad(set_to_none=True)
    n = scorer.n_prefix
    cam = (captured["h"].grad[0, n:] * captured["h"][0, n:]).sum(-1)
    cam = cam if signed else torch.relu(cam)
    return cam.detach().cpu().numpy().reshape(scorer.grid, scorer.grid)


def _image_with_attentions(scorer: ClipScorer, pixels: torch.Tensor):
    """Image embedding plus every attention map (kept in the graph) and, for SigLIP, the pooling map."""
    vm = scorer.model.vision_model
    h = vm.embeddings(pixels.to(scorer.device))
    if scorer.n_prefix:                                        # CLIP
        h = vm.pre_layrnorm(h)
    out = vm.encoder(inputs_embeds=h, output_attentions=True)
    attns = list(out.attentions)
    for a in attns:
        a.retain_grad()
    last = out.last_hidden_state
    if scorer.n_prefix:
        emb = scorer.model.visual_projection(vm.post_layernorm(last[:, 0]))
        head_attn = None
    else:                                                      # SigLIP attention-pooling head
        last = vm.post_layernorm(last)
        head = vm.head
        pooled, head_attn = _pooling_attention(head, last)
        head_attn.retain_grad()
        emb = (pooled + head.mlp(head.layernorm(pooled)))[:, 0]
    return torch.nn.functional.normalize(emb, dim=-1), attns, head_attn


def _pooling_attention(head, x: torch.Tensor):
    """SigLIP's attention-pooling step written out, so the attention map stays in the graph.

    nn.MultiheadAttention returns its weights as a reshaped view of the tensor it used,
    and gradients never reach that view; this recomputes the same thing explicitly.
    Returns (pooled (B, 1, E), attention (B, heads, 1, T)).
    """
    mha = head.attention
    b, t, e = x.shape
    nh, hd = mha.num_heads, e // mha.num_heads
    w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3)
    probe = head.probe.expand(b, -1, -1)
    q = torch.nn.functional.linear(probe, w_q, b_q).view(b, 1, nh, hd).transpose(1, 2)
    k = torch.nn.functional.linear(x, w_k, b_k).view(b, t, nh, hd).transpose(1, 2)
    v = torch.nn.functional.linear(x, w_v, b_v).view(b, t, nh, hd).transpose(1, 2)
    attn = torch.softmax(q @ k.transpose(-1, -2) / hd ** 0.5, dim=-1)        # (B, heads, 1, T)
    out = (attn @ v).transpose(1, 2).reshape(b, 1, e)
    return mha.out_proj(out), attn


def chefer(scorer: ClipScorer, image, text: str, phrase: PhrasePlayer | None) -> np.ndarray:
    """Chefer et al. (2021) relevance: R <- R + mean_heads((grad * A)^+) R over layers."""
    target = _target(scorer, text, phrase)
    with torch.enable_grad():
        emb, attns, head_attn = _image_with_attentions(scorer, scorer.to_tensor(image)[None])
        (emb[0] @ target).backward()
    scorer.model.zero_grad(set_to_none=True)
    t = attns[0].shape[-1]
    rel = torch.eye(t, device=attns[0].device)
    for a in attns:
        cam = (a.grad * a).clamp(min=0).mean(1)[0]            # (T, T)
        rel = rel + cam @ rel
    if scorer.n_prefix:
        patch_rel = rel[0, scorer.n_prefix:]
    else:
        cam = (head_attn.grad * head_attn).clamp(min=0).mean(1)[0]   # (1, T)
        patch_rel = (cam @ rel)[0]
    return patch_rel.detach().cpu().numpy().reshape(scorer.grid, scorer.grid)


# ---- 2nd order ------------------------------------------------------------------------

def interaction2(scorer: ClipScorer, image, text: str, regions: Sequence[RegionPlayer], phrase: PhrasePlayer,
                 operators=("drop", "blur", "mean")) -> RegionScores:
    """Local region x phrase interaction (everything else present), mean over operators x substitutes."""
    games = scorer.method_games(image, text, regions, phrase, operators)
    names = [r.name for r in regions]
    out = {}
    for i, name in enumerate(names):
        other = names[(i + 1) % len(names)]
        rt = robust_triple(games, name, other, phrase.name)
        vals = [e.pair_local[f"{name}|{phrase.name}"] for e in rt.estimates]
        out[name] = {"mean": float(np.mean(vals)), "sum": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    return out


def token_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and character spans. Fast tokenizers report offsets; for slow SentencePiece
    tokenizers (SigLIP) spans are rebuilt from the token strings, case-insensitively."""
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(text, return_offsets_mapping=True)
        return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    ids = tokenizer(text)["input_ids"]
    lower, pos, offsets = text.lower(), 0, []
    special = set(tokenizer.all_special_tokens)
    for t in tokenizer.convert_ids_to_tokens(ids):
        core = t.replace("▁", " ").strip() if t not in special else ""
        start = lower.find(core.lower(), pos) if core else -1
        if start < 0:
            offsets.append((pos, pos))                     # special or bare word-boundary token
            continue
        offsets.append((start, start + len(core)))
        pos = start + len(core)
    return ids, offsets


def _load_fixlip(src: Path):
    """Import FIxLIP's estimator and game modules without its package __init__,
    which also imports plotting code (matplotlib, datasets, ...) we do not need."""
    import importlib
    import types
    name = "_fixlip_src"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(src)]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.fixlip"), importlib.import_module(f"{name}.game_huggingface")


class FixlipBaseline:
    """Official FIxLIP (github.com/hbaniecki/fixlip, pinned by scripts/setup_fixlip.py)."""

    def __init__(self, model_name: str, device: str | None = None, budget_image: int = 2 ** 11,
                 budget_text: int = 2 ** 6, batch_size: int = 64, src_dir: Path = FIXLIP_DIR):
        if not (Path(src_dir) / "src" / "fixlip.py").exists():
            raise FileNotFoundError(f"FIxLIP not found in {src_dir}; run scripts/setup_fixlip.py")
        self._fixlip, self._game = _load_fixlip(Path(src_dir) / "src")
        from transformers import AutoModel, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.is_clip = "siglip" not in model_name.lower()
        self.budget_image, self.budget_text, self.batch_size = budget_image, budget_text, batch_size

    def phrase_players(self, text: str, span: str) -> list[int]:
        """Text-player indices of the tokens covering the first whole-word occurrence of `span`."""
        import re
        m = re.search(r"(?<!\w)" + re.escape(span) + r"(?!\w)", text)
        if not m:
            raise ValueError(f"{span!r} not in {text!r}")
        tokenizer = self.processor.tokenizer
        ids, offsets = token_offsets(tokenizer, text)
        pad = tokenizer.pad_token_id
        positions = [i for i, (s, e) in enumerate(offsets) if e > m.start() and s < m.end() and e > s]
        if self.is_clip:                                       # players exclude BOS (position 0) and EOS
            return [p - 1 for p in positions]
        players = [i for i, t in enumerate(ids) if t != pad]  # SigLIP: players are non-pad tokens in order
        return [players.index(p) for p in positions if p in players]

    def interactions(self, image, text: str):
        game = self._game.VisionLanguageGame(model=self.model, processor=self.processor, input_image=image,
                                  input_text=text, batch_size=self.batch_size)
        fx = self._fixlip.FIxLIP(n_players_image=game.n_players_image, n_players_text=game.n_players_text,
                    max_order=2, p=0.5, mode="banzhaf", random_state=0)
        iv = fx.approximate_crossmodal(game=game, budget_image=self.budget_image, budget_text=self.budget_text)
        return game, iv

    def region_scores(self, image, text: str, phrase: PhrasePlayer, patch_sets: Dict[str, np.ndarray],
                      pairs: Sequence[tuple[str, str]] = (("subj", "obj"),)) -> RegionScores:
        """Region x phrase sums of patch x token interactions, plus region x region image-image
        interactions (keyed "pair:a|b"), which ignore the phrase entirely."""
        game, iv = self.interactions(image, text)
        n_img = game.n_players_image
        if n_img != len(next(iter(patch_sets.values()))):
            raise ValueError(f"FIxLIP has {n_img} image players, regions use a different grid")
        look, vals = iv.interaction_lookup, iv.values

        def val(i, j):
            k = (i, j) if i < j else (j, i)
            return vals[look[k]] if k in look else 0.0

        tokens = [n_img + j for j in self.phrase_players(text, phrase.span)]
        per_patch = np.array([sum(val(i, j) for j in tokens) for i in range(n_img)])
        out = aggregate(per_patch, patch_sets)
        for a, b in pairs:
            ia, ib = np.flatnonzero(patch_sets[a]), np.flatnonzero(patch_sets[b])
            tot = float(sum(val(i, j) for i in ia for j in ib))
            out[f"pair:{a}|{b}"] = {"mean": tot / max(len(ia) * len(ib), 1), "sum": tot}
        return out
