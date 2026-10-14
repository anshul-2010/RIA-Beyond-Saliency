"""Pick a scorer class from a Hugging Face model name."""

from .clip_scorer import ClipScorer


def make_scorer(model_name: str, **kwargs) -> ClipScorer:
    if "qwen" in model_name.lower():
        from .qwen_scorer import QwenVLScorer
        return QwenVLScorer(model_name, **kwargs)
    if "siglip" in model_name.lower():
        from .siglip_scorer import SiglipScorer
        return SiglipScorer(model_name, **kwargs)
    return ClipScorer(model_name, **kwargs)
