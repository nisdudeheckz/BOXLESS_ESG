# src/boxless_esg/text_parse.py
from __future__ import annotations
from typing import List, Dict, Tuple, Set
import re

# Optional boosters (use if installed; otherwise we fallback gracefully)
try:
    import spacy  # type: ignore
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

# Minimal synonym table (extend as needed or via anchors.yaml)
_BUILTIN_SYNS: Dict[str, List[str]] = {
    "person": ["person","people","man","woman","boy","girl","guy","lady","worker","vendor"],
    "cart":   ["cart","trolley","handcart","fruit cart","street cart","pushcart","barrow"],
    "bike":   ["bike","bicycle","cycle"],
    "car":    ["car","vehicle","automobile","van","sedan","taxi"],
    "dog":    ["dog","puppy","canine"],
    "cat":    ["cat","kitten","feline"],
    "table":  ["table","desk","counter","stand"],
    "bag":    ["bag","backpack","sack","tote"],
}

_REL_WORDS = {
    "left_of":  ["left of","on the left of","to the left of","left"],
    "right_of": ["right of","on the right of","to the right of","right"],
    "near":     ["near","close to","next to","beside","by"],
    "above":    ["above","over","on top of"],
    "below":    ["below","under","beneath"],
    "in_front": ["in front of","ahead of","before"],
    "behind":   ["behind","at the back of"],
}

def _tokenize(txt: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", txt.lower())

def extract_noun_anchors(text: str) -> List[List[str]]:
    """
    Returns a list of anchor groups; each group is a list of synonyms/aliases.
    Group 0 = primary subject candidates, Group 1 = secondary object candidates, etc.
    Heuristics:
      - prefer spaCy noun chunks if available
      - otherwise take noun-ish tokens (fallback)
      - expand with a small builtin synonym table
    """
    t = text.lower()
    groups: List[List[str]] = []

    # try spaCy noun chunks (subject/object order approximated)
    if _NLP is not None:
        doc = _NLP(t)
        noun_chunks = [nc.text.strip().lower() for nc in doc.noun_chunks]
        seen: Set[str] = set()
        for nc in noun_chunks:
            # split simple compounds "fruit cart" -> keep phrase + head
            parts = [nc]
            toks = _tokenize(nc)
            if len(toks) >= 2:
                parts.extend(toks)
            expanded = _expand_syns(parts)
            key = " ".join(sorted(expanded))
            if key not in seen and expanded:
                groups.append(expanded)
                seen.add(key)
    else:
        # fallback: just pick top 2-3 content tokens (excluding relation words)
        toks = _tokenize(t)
        rel_vocab = set(sum(_REL_WORDS.values(), []))
        cand = [w for w in toks if w not in rel_vocab and len(w) >= 3][:3]
        if cand:
            groups.append(_expand_syns([cand[0]]))
        if len(cand) > 1:
            groups.append(_expand_syns([cand[1]]))
        if len(cand) > 2:
            groups.append(_expand_syns([cand[2]]))

    # ensure at least one group
    if not groups:
        groups = [_expand_syns(_tokenize(t)[:1] or ["object"])]

    return groups

def _expand_syns(words: List[str]) -> List[str]:
    """Expand with builtin synonyms; keeps original words too."""
    out: Set[str] = set()
    for w in words:
        out.add(w)
        # map via builtin table if we have a key match
        for k, syns in _BUILTIN_SYNS.items():
            if w == k or w in syns:
                out.update(syns)
    return sorted(out)

def parse_relation(text: str) -> str:
    t = text.lower()
    for key, phrases in _REL_WORDS.items():
        for p in phrases:
            if p in t:
                return key
    return "near"  # default

def relation_pairs(groups: List[List[str]]) -> List[Tuple[int,int]]:
    """
    Decide which group pairs to consider for relation bonuses.
    Default: first->second, first->third, ... (if they exist)
    """
    if len(groups) < 2:
        return []
    pairs = []
    for j in range(1, len(groups)):
        pairs.append((0, j))
    return pairs
