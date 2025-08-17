# src/boxless_esg/vis.py
import cv2
def draw_boxes(img_bgr, boxes, color=(0,0,255), thickness=2):
    out = img_bgr.copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(out, (x1,y1),(x2,y2), color, thickness)
    return out
