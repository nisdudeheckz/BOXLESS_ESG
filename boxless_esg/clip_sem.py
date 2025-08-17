# src/boxless_esg/clip_sem.py
from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip

# -------- Defaults you can override with env vars ----------
# Smaller, faster, much lighter on VRAM than ViT-L/14-336
_CLIP_MODEL   = os.getenv("BOXLESS_CLIP_MODEL", "ViT-B-16")
_CLIP_TAG     = os.getenv("BOXLESS_CLIP_TAG",   "openai")
_DEVICE_PREF  = os.getenv("BOXLESS_DEVICE",     "cuda")   # "cuda" | "cpu"
_CHUNK        = int(os.getenv("BOXLESS_CLIP_CHUNK", "16"))  # crops per forward
# -----------------------------------------------------------

_MODEL = None
_TXT_TOK = None
_PREPROCESS = None
_DEVICE = "cpu"


def _pick_device() -> str:
    if _DEVICE_PREF.lower() == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_clip():
    global _MODEL, _TXT_TOK, _PREPROCESS, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TXT_TOK, _PREPROCESS, _DEVICE

    _DEVICE = _pick_device()

    # Create model + preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(
        _CLIP_MODEL, pretrained=_CLIP_TAG, device=_DEVICE
    )
    tokenizer = open_clip.get_tokenizer(_CLIP_MODEL)

    _MODEL, _TXT_TOK, _PREPROCESS = model, tokenizer, preprocess
    return _MODEL, _TXT_TOK, _PREPROCESS, _DEVICE


@torch.no_grad()
def _encode_text(model, tokenizer, text: str, device: str) -> torch.Tensor:
    tokens = tokenizer([text])
    tokens = tokens.to(device)
    txt = model.encode_text(tokens)
    txt = F.normalize(txt, dim=-1)
    return txt  # [1, D]


def _pil_crops_from_boxes(img_np: np.ndarray,
                          boxes: List[Tuple[int,int,int,int]]) -> List[Image.Image]:
    im = Image.fromarray(img_np)
    crops = []
    for x1, y1, x2, y2 in boxes:
        # safety clamp
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(im.width  - 1, int(x2))
        y2 = min(im.height - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            # degenerate -> fallback to 1x1
            x2 = x1 + 1; y2 = y1 + 1
        crops.append(im.crop((x1, y1, x2, y2)))
    return crops


def _encode_images_chunked(model, preprocess, crops: List[Image.Image],
                           device: str, chunk: int) -> torch.Tensor:
    """
    Robustly encode a list of PIL crops:
    - processes in small chunks to fit VRAM
    - if CUDA OOM happens, falls back to CPU for the remaining batches
    Returns normalized features [N, D]
    """
    feats = []
    use_device = device

    i = 0
    while i < len(crops):
        batch = crops[i:i+chunk]
        try:
            batch_t = torch.stack([preprocess(c).to(use_device) for c in batch], dim=0)
            emb = model.encode_image(batch_t)
            emb = F.normalize(emb, dim=-1)
            feats.append(emb.detach().cpu())
            i += len(batch)
        except RuntimeError as e:
            # If it's CUDA OOM or cublas failure, fall back to CPU for the rest
            msg = str(e).lower()
            oomish = ("cuda out of memory" in msg or
                      "cublas_status_allocation_failed" in msg or
                      "cublas_status_execution_failed" in msg)
            if use_device == "cuda" and oomish:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                use_device = "cpu"
                # retry this same chunk on CPU
                continue
            else:
                raise

    return torch.cat(feats, dim=0) if feats else torch.zeros((0, 512))


@torch.no_grad()
def clip_scores_for_boxes(
    img_np: np.ndarray,
    boxes_xyxy: List[Tuple[int,int,int,int]],
    text: str
) -> List[float]:
    """
    Return CLIP similarity scores (0..1) between the text and each crop in boxes.
    Works on CUDA if available, gracefully falling back to CPU on OOM.
    """
    model, tokenizer, preprocess, device = _load_clip()

    # Encode text once
    txt_feat = _encode_text(model, tokenizer, text, device)  # [1, D]

    # Make crops
    crops = _pil_crops_from_boxes(img_np, boxes_xyxy)

    # Encode images in safe chunks (auto CPU fallback)
    img_feats = _encode_images_chunked(model, preprocess, crops, device, _CHUNK)  # [N, D]
    if img_feats.ndim != 2 or img_feats.shape[0] == 0:
        return [0.0] * len(boxes_xyxy)

    # Cosine sim -> scale to 0..1
    sims = (img_feats @ txt_feat.cpu().T).squeeze(-1)  # [N]
    sims = sims.clamp(-1, 1)  # numerical safety
    sims01 = (sims + 1.0) * 0.5
    return sims01.cpu().tolist()
