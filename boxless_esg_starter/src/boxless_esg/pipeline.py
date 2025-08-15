# src/boxless_esg/pipeline.py
from __future__ import annotations

import os, json
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

from .utils import (
    Box, iou_boxes, mask_from_box, expand_box, save_evidence_json,
)
from .proposals.selective_search import selective_search_boxes
from .gdino_gate import gdino_boxes
from .clip_sem import clip_scores_for_boxes
from .relations import rel_score_for_box
from .affordance import (
    detect_primitives, affordance_from_text, affordance_scores_for_boxes,
)
from .calibration import CalibModel, softmax_confidence

# --------------------------------------------------------------------------------------
# Weights for final score. We will reweight dynamically if the gate has no signal.
WEIGHTS = dict(clip=0.70, gdino=0.05, rel=0.20, afford=0.05)


def _area_penalty(box_xyxy: Tuple[int,int,int,int],
                  W: int, H: int,
                  target_frac: float = 0.06,   # a bit tighter than 0.08
                  k: float = 10.0) -> float:
    """
    Penalize oversized boxes. Returns ~1.0 when the box area ~= target_frac * image,
    decays as the box grows larger.
    """
    x1, y1, x2, y2 = box_xyxy
    a = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
    A = W * H
    frac = a / float(A)
    return float(np.exp(-k * max(0.0, frac - target_frac)))


def _iou_to_nearest(b: Box, refs: List[Box]) -> float:
    if not refs:
        return 0.0
    return max(iou_boxes(b, r) for r in refs)


def _rel_people_cart_bonus(b: Box,
                           persons: List[Box],
                           carts: List[Box],
                           W: int, H: int) -> float:
    """
    0..1 proximity bonus encouraging boxes that sit near both persons and carts.
    """
    if not persons and not carts:
        return 0.0

    ax = (b.x1 + b.x2) / 2.0
    ay = (b.y1 + b.y2) / 2.0
    diag = (W ** 2 + H ** 2) ** 0.5

    def near_score(cands: List[Box]) -> float:
        if not cands:
            return 0.0
        dmin = min(
            ((ax - (c.x1 + c.x2) / 2.0) ** 2 + (ay - (c.y1 + c.y2) / 2.0) ** 2) ** 0.5
            for c in cands
        )
        # tighter distance -> larger bonus
        return float(np.exp(-4.0 * (dmin / diag)))

    s_p = near_score(persons)
    s_c = near_score(carts)
    return 0.5 * s_p + 0.5 * s_c


def run_pipeline(
    image_path: str,
    text_query: str,
    out_dir: str,
    max_regions: int = 200,
    iou_dedupe: float = 0.85,
    gate_keep_k: int = 20,
    margin: float = 0.10,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    img_np = np.array(im)

    # ---------- Optional "precision preset" hook (quick early exit) ----------
    try:
        # lazy import avoids circulars if present
        from .precision_bbox import maybe_precision_bbox  # type: ignore
        prec = maybe_precision_bbox(im, img_np, text_query, out_dir)
    except Exception:
        prec = None

    if prec is not None:
        return prec
    # ------------------------------------------------------------------------

    # 0) cheap primitives (faces/uppers/full bodies etc.)
    prims = detect_primitives(img_np)

    # 1) Proposals (Selective Search)
    raw_boxes = selective_search_boxes(img_np, max_regions=max_regions, mode="fast")

    # 2) Dedupe by IoU
    boxes: List[Box] = []
    for x1, y1, x2, y2 in raw_boxes:
        b = Box(x1, y1, x2, y2)
        if all(iou_boxes(b, kb) <= iou_dedupe for kb in boxes):
            boxes.append(b)

    if not boxes:
        raise RuntimeError("No proposals after IoU dedupe.")

    # 3) GDINO gate with query expansion + relaxed thresholds
    anchors = [
        "person", "people",
        "cart", "trolley", "handcart", "fruit cart", "street cart",
    ]

    gd_refs_all: List[Box] = []
    for txt in anchors:
        try:
            bxs = gdino_boxes(
                img_np,
                txt,
                box_threshold=0.15,   # relaxed
                text_threshold=0.20,  # relaxed
                topn=150              # more proposals
            )
            gd_refs_all.extend(Box(*bb) for bb in bxs)
        except Exception:
            # If the model isn't available, just skip; we stay robust.
            pass

    # NMS-style dedupe of GDINO references
    gd_refs: List[Box] = []
    for b in gd_refs_all:
        if all(iou_boxes(b, r) < 0.60 for r in gd_refs):
            gd_refs.append(b)

    gate_scores_all = [_iou_to_nearest(b, gd_refs) for b in boxes]
    order = np.argsort(gate_scores_all)[::-1]
    keep_idx = order[: min(gate_keep_k, len(order))]
    kept: List[Box] = [boxes[i] for i in keep_idx]
    kept_gate_scores: List[float] = [float(gate_scores_all[i]) for i in keep_idx]
    kept_xyxy: List[Tuple[int, int, int, int]] = [b.as_list() for b in kept]

    if not kept:
        # fallback: keep top-K by area inverse (small-to-medium)
        order_area = np.argsort(
            [-(kb.x2 - kb.x1 + 1) * (kb.y2 - kb.y1 + 1) for kb in boxes]
        )
        keep_idx = order_area[: min(gate_keep_k, len(order_area))]
        kept = [boxes[i] for i in keep_idx]
        kept_gate_scores = [0.0 for _ in kept]
        kept_xyxy = [b.as_list() for b in kept]

    # 4) Heavy scoring
    # 4a) CLIP semantic similarity on the kept proposals
    S_clip = np.asarray(clip_scores_for_boxes(img_np, kept_xyxy, text_query), dtype=float)

    # 4b) Relation score (near/left/right) + area penalty
    t = text_query.lower()
    REL_MODE = "near"
    if "left of" in t:
        REL_MODE = "left_of"
    if "right of" in t:
        REL_MODE = "right_of"

    rel_raw = [
        rel_score_for_box(i, kept_xyxy, (W, H), want=REL_MODE)
        for i in range(len(kept_xyxy))
    ]
    rel_raw = np.asarray(rel_raw, dtype=float)

    # area penalty to discourage very large boxes
    rel_pen = np.asarray(
        [_area_penalty(b, W, H) for b in kept_xyxy], dtype=float
    )
    S_rel = rel_raw * rel_pen

    # Optional class-aware people-cart proximity bonus for queries like yours
    wants_people = any(k in t for k in ["person", "people"])
    wants_cart = any(k in t for k in ["cart", "trolley", "handcart", "fruit cart"])
    if wants_people or wants_cart:
        persons = [r for r in gd_refs if "person" in t or "people" in t]  # simple filter
        carts = [r for r in gd_refs if any(k in t for k in ["cart", "trolley", "handcart", "fruit cart"])]
        bonus = np.asarray(
            [_rel_people_cart_bonus(kept[i], persons, carts, W, H) for i in range(len(kept))],
            dtype=float,
        )
        S_rel = np.minimum(1.0, S_rel + 0.25 * bonus)

    # 4c) Affordance (cheap, optional)
    AFF_MODE = affordance_from_text(text_query)  # e.g., talking/sitting/walking/none
    S_aff = np.asarray(affordance_scores_for_boxes(kept_xyxy, prims, (W, H), mode=AFF_MODE),
                       dtype=float)

    # 4d) Gate scores
    S_gd = np.asarray(kept_gate_scores, dtype=float)

    # 5) Combine with dynamic reweight if the gate is silent
    Wc, Wg, Wr, Wa = WEIGHTS["clip"], WEIGHTS["gdino"], WEIGHTS["rel"], WEIGHTS["afford"]
    if float(np.max(S_gd)) == 0.0:
        Wc = Wc + Wg
        Wg = 0.0

    final_scores = Wc * S_clip + Wg * S_gd + Wr * S_rel + Wa * S_aff

    # 6) Pick winner and save artifacts
    best_i = int(np.argmax(final_scores))
    best_box: Box = kept[best_i]
    best_mask = mask_from_box(best_box, (H, W))
    narr_box = expand_box(best_box, W, H, margin=margin)

    Image.fromarray(best_mask).save(os.path.join(out_dir, "mask.png"))
    im.crop(tuple(narr_box.as_list())).save(
        os.path.join(out_dir, "narrative_crop.jpg"), quality=95
    )

    # 7) Calibrated probability (or softmax fallback)
    calib_path = os.path.join(out_dir, "calibration.json")
    if os.path.exists(calib_path):
        try:
            cm = CalibModel.load(calib_path)
            prob_calibrated = float(cm.prob(float(final_scores[best_i])))
        except Exception:
            prob_calibrated = float(
                softmax_confidence([float(x) for x in final_scores], temperature=0.5)
            )
    else:
        prob_calibrated = float(
            softmax_confidence([float(x) for x in final_scores], temperature=0.5)
        )

    ABSTAIN_THRESHOLD = 0.10  # keep low until you fit a real calibration
    did_abstain = prob_calibrated < ABSTAIN_THRESHOLD

    # 8) Evidence JSON
    comps = []
    for b, sc, sg, sr, sa, sf in zip(kept, S_clip, S_gd, S_rel, S_aff, final_scores):
        comps.append(
            dict(
                box=b.as_list(),
                clip=float(sc),
                gdino=float(sg),
                rel=float(sr),
                afford=float(sa),
                final=float(sf),
            )
        )

    evidence = {
        "version": "boxless-esg-lesson4-clip-1.0",
        "image_id": os.path.basename(image_path),
        "query": text_query,
        "winner": {
            "box_raw": best_box.as_list(),
            "bbox_narrative": narr_box.as_list(),
            "final_score": float(final_scores[best_i]),
            "prob_calibrated": prob_calibrated,
            "abstain": bool(did_abstain),
        },
        "gate": {
            "gdino_boxes": [b.as_list() for b in gd_refs],
            "kept_gate_scores": kept_gate_scores,
        },
        "rel": {"mode": "near" if REL_MODE == "near" else REL_MODE},
        "affordance": {
            "mode": AFF_MODE,
            "detected": {k: len(v) for k, v in prims.items()},
        },
        "scores": {
            "weights": dict(clip=Wc, gdino=Wg, rel=Wr, afford=Wa),
            "topk": comps,
        },
        "proposals": {"after_dedupe": len(boxes), "kept_for_heavy": len(kept)},
        "note": "GDINO anchor expansion + area penalty + optional people-cart proximity bonus.",
    }
    save_evidence_json(os.path.join(out_dir, "evidence.json"), evidence)

    return {
        "mask_path": os.path.join(out_dir, "mask.png"),
        "crop_path": os.path.join(out_dir, "narrative_crop.jpg"),
        "evidence_path": os.path.join(out_dir, "evidence.json"),
    }
