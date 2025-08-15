# src/boxless_esg/relations.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

def _center_xy(b: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return (x1+x2)/2.0, (y1+y2)/2.0

def pairwise_centroid_dists(boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    """NxN matrix of Euclidean distances between box centroids."""
    C = np.array([_center_xy(b) for b in boxes], dtype=np.float32)
    diff = C[:,None,:] - C[None,:,:]
    return np.sqrt((diff**2).sum(-1))

def left_of(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int], tau: float = 0.0) -> float:
    """How much A is left of B (1 good, 0 bad). tau: slack in pixels."""
    ax = (a[0]+a[2])/2.0; bx = (b[0]+b[2])/2.0
    d = bx - ax - tau
    if d <= 0: return 0.0
    # logistic squash: farther left → closer to 1 (cap at ~1)
    return float(1.0 - np.exp(-d/100.0))

def right_of(a, b, tau: float = 0.0) -> float:
    """How much A is right of B."""
    return left_of(b, a, tau)

def near(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int], img_diag: float) -> float:
    """
    NEAR score in [0,1]: 1 if centroids almost coincide, decreases with distance.
    Uses image diagonal for scale; ~0.15*diag still has some credit.
    """
    ax,ay = _center_xy(a); bx,by = _center_xy(b)
    d = np.hypot(ax-bx, ay-by)
    # convert to affinity: exp(- (d / (0.15*diag))^2 )
    s = np.exp(- (d / (0.15*img_diag))**2 )
    return float(s)

def rel_score_for_box(
    idx: int,
    boxes_xyxy: List[Tuple[int,int,int,int]],
    img_wh: Tuple[int,int],
    want: str = "near"  # "near", "left_of", "right_of"
) -> float:
    """
    Aggregate relation score for one box vs others. For now:
    - 'near': max NEAR to any other kept proposal
    - 'left_of'/'right_of': best score across others
    """
    W,H = img_wh; img_diag = float(np.hypot(W, H))
    a = boxes_xyxy[idx]
    scores = []
    for j,b in enumerate(boxes_xyxy):
        if j==idx: continue
        if want == "near":
            scores.append(near(a,b,img_diag))
        elif want == "left_of":
            scores.append(left_of(a,b))
        elif want == "right_of":
            scores.append(right_of(a,b))
    return float(max(scores) if scores else 0.0)
