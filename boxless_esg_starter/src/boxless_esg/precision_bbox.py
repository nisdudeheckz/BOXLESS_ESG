# src/boxless_esg/precision_bbox.py
from __future__ import annotations
from typing import Optional, Dict, Tuple, List
import os, math, json
import numpy as np
from PIL import Image

# we reuse your existing helper; if it fails we gracefully skip precision mode
def _try_import_gdino():
    try:
        from .gdino_gate import gdino_boxes
        return gdino_boxes
    except Exception:
        return None

def _center(b: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return (x1+x2)/2.0, (y1+y2)/2.0

def _dist(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax,ay=_center(a); bx,by=_center(b)
    return float(math.hypot(ax-bx, ay-by))

def _union(bxs: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
    x1=min(b[0] for b in bxs); y1=min(b[1] for b in bxs)
    x2=max(b[2] for b in bxs); y2=max(b[3] for b in bxs)
    return (int(x1),int(y1),int(x2),int(y2))

def _pad(b: Tuple[int,int,int,int], W:int,H:int, ratio:float=0.10) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2=b
    w=x2-x1+1; h=y2-y1+1
    dx=int(ratio*w); dy=int(ratio*h)
    return (max(0,x1-dx), max(0,y1-dy), min(W-1,x2+dx), min(H-1,y2+dy))

def _detect_words(img_np: np.ndarray, words: List[str], gdino, topn=120,
                  box_thr=0.20, txt_thr=0.20):
    out=[]
    for w in words:
        try:
            for b in gdino(img_np, w, topn=topn, box_threshold=box_thr, text_threshold=txt_thr):
                out.append(b)
        except Exception:
            pass
    return out

def maybe_precision_bbox(img_pil: Image.Image,
                         img_np: np.ndarray,
                         text_query: str,
                         out_dir: str) -> Optional[Dict]:
    """
    Precision path for 'people near a cart' style queries.
    Returns dict with crop_path/mask_path/evidence_path, or None to skip.
    """
    t=text_query.lower()
    if not (("person" in t or "people" in t or "man" in t or "woman" in t)
            and ("cart" in t or "trolley" in t or "handcart" in t)):
        return None

    gdino = _try_import_gdino()
    if gdino is None:
        return None  # fall back to the normal pipeline

    W,H = img_np.shape[1], img_np.shape[0]

    people = _detect_words(img_np, ["person","man","woman","people"], gdino, topn=150, box_thr=0.20, txt_thr=0.20)
    carts  = _detect_words(img_np, ["cart","trolley","handcart","fruit cart"], gdino, topn=120, box_thr=0.15, txt_thr=0.20)
    if not people or not carts:
        return None

    diag = math.hypot(W,H)
    def pair_score(p, c):
        d = _dist(p.as_list(), c.as_list()) / (diag+1e-6)
        prox = math.exp(-(d/0.12)**2)
        conf = 0.5*(p.score + c.score)
        return 0.70*prox + 0.30*conf

    # choose best 2 people near the same cart if 'two' mentioned
    want_two = ("two" in t) or (" 2 " in t) or ("people" in t)
    best=None
    for c in carts:
        scored = sorted([(pair_score(p,c), p) for p in people], key=lambda x:-x[0])
        group = [scored[0][1]]
        s = scored[0][0]
        if want_two and len(scored) > 1:
            s += scored[1][0]
            group.append(scored[1][1])
        if best is None or s > best[0]:
            best = (s, c, group)

    s, cart, group = best
    parts = [cart.as_list()] + [p.as_list() for p in group]
    u = _pad(_union(parts), W, H, 0.10)

    # save narrative crop + dummy mask
    crop_path = os.path.join(out_dir, "narrative_crop.jpg")
    Image.fromarray(img_np).crop(u).save(crop_path)

    # simple bbox mask (white box on black)
    mask = np.zeros((H,W), dtype=np.uint8)
    x1,y1,x2,y2 = u
    mask[y1:y2+1, x1:x2+1] = 255
    mask_path = os.path.join(out_dir, "mask.png")
    from PIL import Image as _PILImage
    _PILImage.fromarray(mask).save(mask_path)

    evid_path = os.path.join(out_dir, "evidence.json")
    with open(evid_path, "w", encoding="utf-8") as f:
        json.dump({
            "preset": "precision_bbox",
            "text": text_query,
            "bbox_union": u,
            "people": [p.as_list() for p in group],
            "cart": cart.as_list(),
            "score": float(s)
        }, f, indent=2)

    return {"crop_path": crop_path, "mask_path": mask_path, "evidence_path": evid_path}
