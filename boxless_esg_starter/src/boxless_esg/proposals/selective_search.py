# src/boxless_esg/proposals/selective_search.py
from typing import List, Tuple
import numpy as np
import cv2

def selective_search_boxes(image: np.ndarray, max_regions: int = 200, mode: str = "fast") -> List[Tuple[int,int,int,int]]:
    """
    Real Selective Search proposals using OpenCV contrib.
    Returns a list of (x1,y1,x2,y2) boxes, sorted by area (desc), capped at max_regions.
    """
    img = image
    if img.dtype != np.uint8:
        img = (np.clip(image, 0, 255)).astype(np.uint8)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast() if mode == "fast" else ss.switchToSelectiveSearchQuality()

    rects = ss.process()  # list of (x, y, w, h)
    H, W = img.shape[:2]
    boxes = []
    for (x, y, w, h) in rects:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w - 1), int(y + h - 1)
        # clamp and skip tiny regions (<0.1% of image area)
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
        if (x2 - x1 + 1) * (y2 - y1 + 1) >= 0.001 * (W * H):
            boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: (b[2]-b[0]+1)*(b[3]-b[1]+1), reverse=True)
    return boxes[:max_regions]
