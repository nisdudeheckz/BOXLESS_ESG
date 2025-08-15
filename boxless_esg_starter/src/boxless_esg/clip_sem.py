# src/boxless_esg/clip_sem.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import open_clip
import torchvision.transforms as T

# Cache model & preprocess
_MODEL = None
_TXT_TOK = None
_PREPROCESS = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_clip():
    global _MODEL, _TXT_TOK, _PREPROCESS
    if _MODEL is not None:
        return _MODEL, _TXT_TOK, _PREPROCESS

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-336", pretrained="openai", device=_DEVICE
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
    _MODEL, _TXT_TOK, _PREPROCESS = model, tokenizer, preprocess
    return _MODEL, _TXT_TOK, _PREPROCESS

def _crop_from_box(img_np: np.ndarray, box_xyxy: Tuple[int,int,int,int], pad_ratio: float = 0.25) -> Image.Image:
    """Crop with extra context pad around the box (improves CLIP robustness)."""
    x1,y1,x2,y2 = box_xyxy
    H, W = img_np.shape[0], img_np.shape[1]
    w = max(1, x2-x1+1); h = max(1, y2-y1+1)
    px = int(w * pad_ratio); py = int(h * pad_ratio)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W-1, x2 + px); y2 = min(H-1, y2 + py)
    from PIL import Image
    return Image.fromarray(img_np[y1:y2+1, x1:x2+1])

@torch.no_grad()
def clip_scores_for_boxes(img_np: np.ndarray, boxes_xyxy: List[Tuple[int,int,int,int]], text: str) -> List[float]:
    """
    Returns cosine similarity (0..1-ish after clamping) between each crop and the text.
    Batched for speed. Safe on CPU/GPU.
    """
    model, tok, preprocess = _load_clip()

    # Prepare text
    txt_tokens = tok([text]).to(_DEVICE)
    txt_feat = model.encode_text(txt_tokens)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    # Prepare images (batched)
    pil_crops = [_crop_from_box(img_np, b) for b in boxes_xyxy]
    imgs = torch.stack([preprocess(im) for im in pil_crops]).to(_DEVICE)

    img_feat = model.encode_image(imgs)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)

    sims = (img_feat @ txt_feat.T).squeeze(1)   # [-1..1]
    sims = sims.clamp(-1, 1).float().cpu().tolist()
    # Map to 0..1 for mixing with other components
    sims01 = [0.5 * (s + 1.0) for s in sims]
    return sims01
