from __future__ import annotations
from typing import List, Tuple
import re
import numpy as np
import cv2

# ---------------------------
# detectors (OpenCV cascades)
# ---------------------------
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_UPPERBODY_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
_FULLBODY_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# HOG person detector (fallback when faces are missed)
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def _detect_rects(gray: np.ndarray, which: str) -> List[Tuple[int, int, int, int]]:
    """Run a chosen Haar cascade and return xyxy rectangles."""
    if which == "face" and not _FACE_CASCADE.empty():
        rects = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    elif which == "upper" and not _UPPERBODY_CASCADE.empty():
        rects = _UPPERBODY_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(32, 32))
    elif which == "full" and not _FULLBODY_CASCADE.empty():
        rects = _FULLBODY_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(32, 32))
    else:
        rects = []
    out: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in rects:
        out.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
    return out


def detect_primitives(img_rgb: np.ndarray) -> dict:
    """Run tiny detectors once and cache results."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _detect_rects(gray, "face")
    uppers = _detect_rects(gray, "upper")
    fulls = _detect_rects(gray, "full")

    # HOG people (x, y, w, h) -> xyxy
    hog_rects, _ = _HOG.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
    people = [(int(x), int(y), int(x + w - 1), int(y + h - 1)) for (x, y, w, h) in hog_rects]

    return {"faces": faces, "uppers": uppers, "fulls": fulls, "people": people}


def _center_xy(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _nearest_dist(box: Tuple[int, int, int, int], refs: List[Tuple[int, int, int, int]]) -> float:
    if not refs:
        return 1e9
    cx, cy = _center_xy(box)
    best = 1e9
    for r in refs:
        rx, ry = _center_xy(r)
        d = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
        if d < best:
            best = d
    return float(best)


def affordance_from_text(text: str) -> str:
    """
    Map free-form text to a coarse affordance mode.
    Returns one of: 'talking', 'sitting', 'standing', 'walking', 'none'
    """
    t = text.lower()
    if re.search(r"\b(talk|spea|shout|chat)\b", t):
        return "talking"
    if re.search(r"\b(sit|sitting|seated)\b", t):
        return "sitting"
    if re.search(r"\b(stand|standing)\b", t):
        return "standing"
    if re.search(r"\b(walk|walking)\b", t):
        return "walking"
    return "none"


def affordance_scores_for_boxes(
    boxes_xyxy: List[Tuple[int, int, int, int]],
    primitives: dict,
    img_wh: Tuple[int, int],
    mode: str,
) -> List[float]:
    """
    Return a score in [0,1] per box, depending on the chosen affordance:
      - talking: near faces (fallback to people if no faces)
      - sitting: near upper-bodies
      - standing/walking: near full-bodies
      - none: zeros
    """
    W, H = img_wh
    diag = (W ** 2 + H ** 2) ** 0.5

    if mode == "talking":
        refs = primitives.get("faces", []) or primitives.get("people", [])
    elif mode == "sitting":
        refs = primitives.get("uppers", [])
    elif mode in ("standing", "walking"):
        refs = primitives.get("fulls", [])
    else:
        refs = []

    out: List[float] = []
    for b in boxes_xyxy:
        if refs:
            d = _nearest_dist(b, refs)
            s = float(np.exp(-((d / (0.12 * diag + 1e-6)) ** 2)))
        else:
            s = 0.0
        out.append(s)
    return out
