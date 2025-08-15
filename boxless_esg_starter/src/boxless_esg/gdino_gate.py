from __future__ import annotations
from typing import List, Tuple
import numpy as np

_HAS_GDINO = True
_USE_HF = False  # whether load_model_hf is available

try:
    import torch
    # Try to import both APIs; different GroundingDINO builds expose different loaders
    from groundingdino.util.inference import predict  # always needed
    try:
        from groundingdino.util.inference import load_model_hf  # newer API
        _USE_HF = True
    except Exception:
        from groundingdino.util.inference import load_model  # older API
except Exception:
    _HAS_GDINO = False

_MODEL = None

def _device():
    if not _HAS_GDINO:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _try_load_hf(model_repo: str, ckpt: str):
    # For builds that have the HuggingFace loader
    try:
        return load_model_hf(repo_id=model_repo, filename=ckpt, device=_device())
    except Exception:
        return None

def _try_load_legacy():
    """
    Older API needs explicit config + checkpoint paths.
    If we can locate the config within the installed package but don't have a ckpt,
    we return None to gracefully fall back.
    """
    try:
        import importlib.resources as pkg
        # Try to resolve the default SwinT config shipped with the package
        cfg = pkg.files("groundingdino").joinpath("config", "GroundingDINO_SwinT_OGC.py")
        if not cfg or not cfg.is_file():
            return None
        # We do not have a local checkpoint file path by default,
        # so we cannot call load_model(cfg, ckpt) reliably.
        # Returning None makes the caller gracefully skip GDINO.
        return None
    except Exception:
        return None

def load_gdino(model_repo: str = "ShilongLiu/GroundingDINO",
               ckpt: str = "GroundingDINO_SwinT_OGC.pth"):
    """
    Load a lightweight GDINO model once and cache it.
    Supports both HF and legacy loaders. Returns None if loading fails.
    """
    global _MODEL
    if not _HAS_GDINO:
        return None
    if _MODEL is not None:
        return _MODEL

    model = None
    if _USE_HF:
        model = _try_load_hf(model_repo, ckpt)
    if model is None:
        model = _try_load_legacy()

    _MODEL = model
    return _MODEL

def gdino_boxes(image_rgb: np.ndarray, text: str,
                box_threshold: float = 0.15,
                text_threshold: float = 0.20,
                topn: int = 150) -> List[Tuple[int,int,int,int]]:
    """
    Return up to topn text‑aligned boxes (x1,y1,x2,y2). If GDINO isn't available/loaded,
    return [] so the pipeline can still run.
    """
    if not _HAS_GDINO:
        return []
    model = load_gdino()
    if model is None:
        return []

    img = image_rgb.astype(np.uint8, copy=False)
    boxes_cxcywh, logits, _ = predict(
        model=model, image=img, caption=text,
        box_threshold=box_threshold, text_threshold=text_threshold
    )
    H, W = img.shape[:2]
    out: List[Tuple[int,int,int,int]] = []
    for cx, cy, w, h in boxes_cxcywh:
        x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
        out.append((x1,y1,x2,y2))
    if out:
        import numpy as _np
        order = _np.argsort(-logits)[:topn]
        out = [out[i] for i in order]
    return out
