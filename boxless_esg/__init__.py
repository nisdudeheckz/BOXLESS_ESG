from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Box:
    x1: int; y1: int; x2: int; y2: int
    def as_list(self) -> List[int]:
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]
    def clamp(self, w:int, h:int) -> "Box":
        x1 = max(0, min(self.x1, w-1)); y1 = max(0, min(self.y1, h-1))
        x2 = max(0, min(self.x2, w-1)); y2 = max(0, min(self.y2, h-1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return Box(x1, y1, x2, y2)
    def area(self) -> int:
        return max(0, self.x2 - self.x1 + 1) * max(0, self.y2 - self.y1 + 1)

def iou_boxes(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    return float(inter) / float(union + 1e-6)

def mask_from_box(box: Box, shape: Tuple[int,int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h,w), dtype=np.uint8)
    x1,y1,x2,y2 = box.as_list()
    m[y1:y2+1, x1:x2+1] = 255
    return m
