"""Scene text — scoreboard ke ilawa frame mein likha text (jersey ki peeth par 'QADIR 17',
boundary boards, dugout cards). Har camera shot ke beech ke 1–2 frames ka OCR.

Peeth par naam = player close-up mein hai lekin chehra nahi dikh raha (face recognition miss
kar deta) — is liye yeh face ka achha saathi signal hai.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from pipeline.cache import cache_dir, load_json, save_json
from pipeline.frames import Crop, iter_frames
from pipeline.scenes import Shot
from pipeline.signals.ocr import read_text
from pipeline.signals.scoreboard import BAND

CACHE_VERSION = 1

# Scoreboard band se upar ka hissa
ABOVE_BAND = Crop(0.0, 0.0, 1.0, BAND.y)


def sample_times(shots: list[Shot], long_shot: float = 6.0) -> list[float]:
    """Har shot ka beech; lamba shot ho to 1/3 aur 2/3 par do frames."""
    times = []
    for a, b in shots:
        if b - a >= long_shot:
            times += [a + (b - a) / 3, a + 2 * (b - a) / 3]
        else:
            times.append((a + b) / 2)
    return times


def scan_scene_text(
    path: str | Path,
    shots: list[Shot],
    *,
    use_cache: bool = True,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    """Returns {"frames": [{"t", "boxes": [[text, score, x0, y0, x1, y1], ...]}], ...}."""
    cache_file = cache_dir(path) / "scene_text.json"
    if use_cache and isinstance(cached := load_json(cache_file), dict):
        if cached.get("version") == CACHE_VERSION and cached.get("n_shots") == len(shots):
            return cached

    times = sample_times(shots)
    t0 = time.perf_counter()
    out = []
    for i, t in enumerate(times):
        frame = next(iter(iter_frames(path, fps=1, width=None, crop=ABOVE_BAND,
                                      start=t, end=t + 0.5)), None)
        if frame is not None:
            boxes = read_text(frame[1], min_score=0.6)
            if boxes:
                out.append({"t": round(t, 2), "boxes": [b.to_list() for b in boxes]})
        if on_progress:
            on_progress((i + 1) / len(times))

    result = {"version": CACHE_VERSION, "n_shots": len(shots), "n_samples": len(times),
              "elapsed": round(time.perf_counter() - t0, 1), "frames": out}
    save_json(cache_file, result)
    return result
