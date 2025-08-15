from __future__ import annotations
import json, math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

# ----------------------------
# Geometry primitives
# ----------------------------

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_list(self) -> List[int]:
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    def clamp(self, w: int, h: int) -> "Box":
        """Clamp the box to image bounds [0..w-1],[0..h-1] and fix inverted coords."""
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w - 1))
        y2 = max(0, min(self.y2, h - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return Box(x1, y1, x2, y2)

    def area(self) -> int:
        return max(0, self.x2 - self.x1 + 1) * max(0, self.y2 - self.y1 + 1)

def iou_boxes(a: Box, b: Box) -> float:
    """IoU between two axis-aligned boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw  = max(0, ix2 - ix1 + 1)
    ih  = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    return float(inter) / float(union + 1e-6)

# ----------------------------
# Mask helpers
# ----------------------------

def mask_from_box(box: Box, shape: Tuple[int, int]) -> np.ndarray:
    """Create a binary uint8 mask (255 inside box, else 0)."""
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = box.as_list()
    m[y1:y2 + 1, x1:x2 + 1] = 255
    return m

def bbox_from_mask(mask: np.ndarray) -> Box:
    """Tight bounding box around non-zero mask pixels."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return Box(0, 0, 0, 0)
    return Box(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def expand_box(box: Box, w: int, h: int, margin: float = 0.10) -> Box:
    """
    Expand box by a fraction of its size; keep inside image.
    margin=0.10 → add 10% on each side.
    """
    dx = int((box.x2 - box.x1 + 1) * margin)
    dy = int((box.y2 - box.y1 + 1) * margin)
    return Box(box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy).clamp(w, h)

# ----------------------------
# JSON I/O
# ----------------------------

def save_evidence_json(path: str, payload: Dict):
    """Pretty-print JSON (UTF‑8) to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
