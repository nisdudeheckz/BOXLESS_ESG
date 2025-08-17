# src/boxless_esg/targets.py
from __future__ import annotations

def parse_subject_and_targets(q: str):
    """
    Very light NLP: decide the SUBJECT we want to localize,
    and the TARGETS the subject should be near/left_of/right_of, etc.
    """
    t = q.lower()

    # subject
    subject = None
    if "people" in t or "persons" in t or "person" in t:
        subject = "person"
    elif "man" in t or "woman" in t:
        subject = "person"
    else:
        # fallback: try to localize the noun in the query as-is
        subject = q

    # possible anchors/targets we try to detect with GDINO
    vocab = [
        "cart", "fruit cart", "stall", "vendor cart",
        "car", "bus", "truck", "bike", "bicycle", "motorcycle",
        "table", "bench", "dog", "cat", "sign", "storefront",
    ]
    targets = [w for w in vocab if w in t]

    # simple relation mode
    mode = "near"
    if "left of" in t or "left-of" in t:
        mode = "left_of"
    elif "right of" in t or "right-of" in t:
        mode = "right_of"

    return subject, targets, mode
