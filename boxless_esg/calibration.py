# src/boxless_esg/calibration.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

@dataclass
class CalibModel:
    kind: str  # "platt" | "isotonic" | "none"
    a: float = 1.0
    b: float = 0.0
    iso_x: Optional[List[float]] = None
    iso_y: Optional[List[float]] = None
    temperature: float = 1.0  # used for fallback softmax over top-K (optional)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "kind": self.kind,
                "a": self.a, "b": self.b,
                "iso_x": self.iso_x, "iso_y": self.iso_y,
                "temperature": self.temperature
            }, f, indent=2)

    @staticmethod
    def load(path: str) -> "CalibModel":
        with open(path) as f:
            d = json.load(f)
        return CalibModel(**d)

    def prob(self, s: float) -> float:
        """Map raw score s -> calibrated probability in [0,1]."""
        if self.kind == "platt":
            z = self.a * s + self.b
            return float(1.0 / (1.0 + np.exp(-z)))
        elif self.kind == "isotonic" and self.iso_x is not None and self.iso_y is not None:
            # piecewise linear interpolation
            x = np.array(self.iso_x); y = np.array(self.iso_y)
            return float(np.interp(s, x, y, left=y[0], right=y[-1]))
        else:
            # no calibration -> identity clip
            return float(np.clip(s, 0.0, 1.0))

def fit_platt(scores: List[float], labels: List[int]) -> CalibModel:
    """Fit logistic regression on single feature."""
    x = np.array(scores).reshape(-1, 1)
    y = np.array(labels).astype(int)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(x, y)
    a = float(lr.coef_[0][0]); b = float(lr.intercept_[0])
    return CalibModel(kind="platt", a=a, b=b)

def fit_isotonic(scores: List[float], labels: List[int]) -> CalibModel:
    x = np.array(scores); y = np.array(labels).astype(int)
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    yhat = ir.fit_transform(x, y)
    # store breakpoints for portable inference
    return CalibModel(kind="isotonic", iso_x=ir.X_thresholds_.tolist(), iso_y=ir.y_thresholds_.tolist())

def softmax_confidence(topk_final_scores: List[float], temperature: float = 0.5) -> float:
    """Fallback confidence: softmax of top-K with temperature; returns prob of argmax."""
    x = np.array(topk_final_scores, dtype=np.float32) / max(1e-6, temperature)
    x = x - x.max()
    p = np.exp(x); p = p / p.sum()
    return float(p.max())
