"""RapidOCR wrapper (ONNX, CPU par fast). Engine ek dafa load hota hai."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np


@dataclass(frozen=True)
class TextBox:
    text: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def to_list(self) -> list:
        return [self.text, round(self.score, 3),
                *(round(v, 1) for v in (self.x0, self.y0, self.x1, self.y1))]

    @classmethod
    def from_list(cls, v: list) -> TextBox:
        return cls(v[0], v[1], *v[2:6])


@cache
def _engine():
    from rapidocr import RapidOCR

    return RapidOCR(params={
        # Broadcast graphics hamesha seedhe hote hain — ulta-text classifier ki zaroorat nahi
        "Global.use_cls": False,
        "Global.log_level": "error",
        # Default (limit_type=min, 736) patli strip ko ~736x5000 tak bara kar deta tha → 5.5s/frame.
        # 'max' 960 se image chhoti hi rehti hai → ~0.6s/frame, accuracy same.
        "Det.limit_type": "max",
        "Det.limit_side_len": 960,
    })


def read_text(image: np.ndarray, min_score: float = 0.5) -> list[TextBox]:
    """BGR image → text boxes (pixel coords)."""
    res = _engine()(image)
    if res is None or res.boxes is None or res.txts is None:
        return []
    out = []
    for box, txt, score in zip(res.boxes, res.txts, res.scores, strict=True):
        if score < min_score or not txt.strip():
            continue
        xs, ys = box[:, 0], box[:, 1]
        out.append(TextBox(txt.strip(), float(score), float(xs.min()), float(ys.min()),
                           float(xs.max()), float(ys.max())))
    return out
