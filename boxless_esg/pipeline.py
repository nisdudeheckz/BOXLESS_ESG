from __future__ import annotations
import os, json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .gdino_gate import gdino_boxes
from .clip_sem import clip_scores_for_boxes
from .relations import rel_score_for_box
from .affordance import detect_primitives, affordance_from_text, affordance_scores_for_boxes
from .calibration import CalibModel, softmax_confidence
from .utils import Box, iou_boxes, mask_from_box, expand_box, save_evidence_json
from .proposals.selective_search import selective_search_boxes

# --------- weights / knobs ----------
WEIGHTS = dict(clip=0.70, gdino=0.05, rel=0.20, afford=0.05)

def _area_penalty(box, W, H, target_frac=0.08, k=8.0):
    x1,y1,x2,y2 = box
    a = max(1,(x2-x1+1)*(y2-y1+1))
    frac = a / (W*H)
    return float(np.exp(-k * max(0.0, frac - target_frac)))

def _iou_to_nearest(b: Box, refs: List[Box]) -> float:
    if not refs: return 0.0
    return max(iou_boxes(b, r) for r in refs)
# ------------------------------------

def _draw_overlay(base_im: Image.Image,
                  kept_xyxy: List[Tuple[int,int,int,int]],
                  best_box: Box,
                  out_path: str):
    im = base_im.copy()
    dr = ImageDraw.Draw(im)

    # green: kept proposals
    for (x1,y1,x2,y2) in kept_xyxy:
        dr.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)

    # yellow: winner
    bx = best_box.as_list()
    dr.rectangle(bx, outline=(255,255,0), width=5)

    im.save(out_path, quality=95)

def run_pipeline(
    image_path: str,
    text_query: str,
    out_dir: str,
    max_regions: int = 200,
    iou_dedupe: float = 0.85,
    gate_keep_k: int = 20,
    margin: float = 0.10,
    save_overlay: bool = False,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    img_np = np.array(im)

    # 0) auto-precision shortcut (your gdino-only preset)
    try:
        from .precision_bbox import maybe_precision_bbox
        prec = maybe_precision_bbox(im, img_np, text_query, out_dir)
    except Exception:
        prec = None
    if prec is not None:
        if save_overlay:
            # draw overlay around this precise bbox if present
            try:
                with open(prec["evidence_path"], "r", encoding="utf-8") as f:
                    ed = json.load(f)
                bb = Box(*ed["winner"]["box_raw"])
                _draw_overlay(im, [bb.as_list()], bb, os.path.join(out_dir, "overlay.jpg"))
            except Exception:
                pass
        return prec

    # 1) primitives & proposals
    prims = detect_primitives(img_np)
    raw_boxes = selective_search_boxes(img_np, max_regions=max_regions, mode="fast")

    # 2) dedupe
    boxes: List[Box] = []
    for (x1,y1,x2,y2) in raw_boxes:
        b = Box(x1,y1,x2,y2)
        if all(iou_boxes(b, kb) <= iou_dedupe for kb in boxes):
            boxes.append(b)
    if not boxes:
        raise RuntimeError("No proposals after dedupe.")

    # 3) GDINO gate
    gd_boxes_list = gdino_boxes(img_np, text_query, box_threshold=0.25, text_threshold=0.25, topn=30)
    gd_refs = [Box(*bb) for bb in gd_boxes_list]
    gate_scores = [_iou_to_nearest(b, gd_refs) for b in boxes]
    order = np.argsort(gate_scores)[::-1]
    keep_idx = order[:min(gate_keep_k, len(order))]
    kept = [boxes[i] for i in keep_idx]
    kept_gate = [float(gate_scores[i]) for i in keep_idx]
    kept_xyxy: List[Tuple[int,int,int,int]] = [b.as_list() for b in kept]

    # 4) heavy scoring (CLIP + relations + area penalty + afford)
    clip_sims = clip_scores_for_boxes(img_np, kept_xyxy, text_query)

    REL_MODE = "near"
    t = text_query.lower()
    if "left of" in t:  REL_MODE = "left_of"
    if "right of" in t: REL_MODE = "right_of"
    rel_scores = [rel_score_for_box(i, kept_xyxy, (W, H), want=REL_MODE)
                  for i in range(len(kept_xyxy))]
    rel_scores = [s * _area_penalty(b, W, H, 0.08, 8.0)
                  for s, b in zip(rel_scores, kept_xyxy)]

    AFF_MODE = affordance_from_text(text_query)
    afford_scores = affordance_scores_for_boxes(kept_xyxy, prims, (W,H), mode=AFF_MODE)

    finals, comps = [], []
    for b, s_clip, s_gate, s_rel, s_aff in zip(kept, clip_sims, kept_gate, rel_scores, afford_scores):
        final = (WEIGHTS["clip"]*s_clip +
                 WEIGHTS["gdino"]*s_gate +
                 WEIGHTS["rel"]*s_rel +
                 WEIGHTS["afford"]*s_aff)
        finals.append(final)
        comps.append({
            "box": b.as_list(),
            "clip": float(s_clip),
            "gdino": float(s_gate),
            "rel": float(s_rel),
            "afford": float(s_aff),
            "final": float(final),
        })

    best_i = int(np.argmax(finals))
    best_box = kept[best_i]
    best_mask = mask_from_box(best_box, (H, W))
    narr_box  = expand_box(best_box, W, H, margin=margin)

    # 5) calibration / abstain
    calib_path = os.path.join(out_dir, "calibration.json")
    if os.path.exists(calib_path):
        cm = CalibModel.load(calib_path)
        calibrated_prob = cm.prob(float(finals[best_i]))
    else:
        calibrated_prob = softmax_confidence([float(x) for x in finals], temperature=0.5)
    did_abstain = calibrated_prob < 0.35

    # 6) outputs
    Image.fromarray(best_mask).save(os.path.join(out_dir, "mask.png"))
    im.crop(tuple(narr_box.as_list())).save(os.path.join(out_dir, "narrative_crop.jpg"), quality=95)
    if save_overlay:
        _draw_overlay(im, kept_xyxy, best_box, os.path.join(out_dir, "overlay.jpg"))

    evidence = {
        "version": "boxless-esg-lesson4-clip-1.0",
        "image_id": os.path.basename(image_path),
        "query": text_query,
        "winner": {
            "box_raw": best_box.as_list(),
            "bbox_narrative": narr_box.as_list(),
            "final_score": float(finals[best_i]),
            "prob_calibrated": float(calibrated_prob),
            "abstain": bool(did_abstain)
        },
        "gate": {
            "gdino_boxes": [b.as_list() for b in gd_refs],
            "kept_gate_scores": kept_gate
        },
        "rel": {"mode": REL_MODE},
        "affordance": {"mode": AFF_MODE, "detected": {k: len(v) for k,v in prims.items()}},
        "scores": {"weights": WEIGHTS, "topk": comps},
        "proposals": {"after_dedupe": len(boxes), "kept_for_heavy": len(kept)},
        "note": "Heavy scoring uses CLIP similarity on top-K proposals; REL penalty discourages giant boxes."
    }
    save_evidence_json(os.path.join(out_dir, "evidence.json"), evidence)

    return {
        "mask_path": os.path.join(out_dir, "mask.png"),
        "crop_path": os.path.join(out_dir, "narrative_crop.jpg"),
        "overlay_path": os.path.join(out_dir, "overlay.jpg") if save_overlay else "",
        "evidence_path": os.path.join(out_dir, "evidence.json"),
    }
