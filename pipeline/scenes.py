"""Shot / camera-cut detection.

ffmpeg se chhote frames (128px, 10 fps) lete hain aur har frame ka HSV histogram + chhoti
grayscale image pichhle frame se compare karte hain. Bara farq = camera cut.
PySceneDetect se ~3.5x tez (25 min video ≈ 70s CPU par) aur cricket broadcast ke wipe
transitions bhi pakarta hai. Result video ke cache mein save hota hai.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from pipeline.cache import cache_dir, load_json, save_json
from pipeline.frames import iter_frames
from pipeline.media import probe

Shot = tuple[float, float]

CACHE_VERSION = 2


@dataclass(frozen=True)
class ShotOptions:
    fps: float = 10.0
    hist_threshold: float = 0.25  # 0–1, color histogram ka farq
    pixel_threshold: float = 0.15  # 0–1, grayscale pixels ka ausat farq
    min_shot: float = 1.0  # do cuts ke darmiyan kam az kam itne seconds


def _features(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist /= max(hist.sum(), 1.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return hist, small


def find_cuts(
    times: np.ndarray, hist_diff: np.ndarray, pixel_diff: np.ndarray, opts: ShotOptions
) -> list[float]:
    """Threshold se upar wale frames, phir non-max suppression (min_shot ke andar sirf sab se
    mazboot cut rakho — wipe transitions 2–3 frames mein phaile hote hain)."""
    strength = hist_diff + pixel_diff
    cand = np.where((hist_diff >= opts.hist_threshold) | (pixel_diff >= opts.pixel_threshold))[0]
    kept: list[int] = []
    for i in cand[np.argsort(-strength[cand], kind="stable")]:
        if all(abs(times[i] - times[j]) >= opts.min_shot for j in kept):
            kept.append(int(i))
    return sorted(float(times[i]) for i in kept)


def detect_shots(
    path: str | Path,
    *,
    use_cache: bool = True,
    opts: ShotOptions | None = None,
) -> list[Shot]:
    """Video ko shots (camera cuts) mein todo. Returns [(start_sec, end_sec), ...]."""
    opts = opts or ShotOptions()
    cache_file = cache_dir(path) / "shots.json"
    cached = load_json(cache_file) if use_cache else None
    if (
        isinstance(cached, dict)
        and cached.get("version") == CACHE_VERSION
        and cached.get("opts") == opts.__dict__
    ):
        return [tuple(s) for s in cached["shots"]]

    times, hists, smalls = [], [], []
    for ts, frame in iter_frames(path, fps=opts.fps, width=128):
        h, s = _features(frame)
        times.append(ts)
        hists.append(h)
        smalls.append(s)

    duration = probe(path).duration
    if len(times) < 2:
        shots: list[Shot] = [(0.0, round(duration, 3))]
    else:
        H, S = np.asarray(hists), np.asarray(smalls)
        hist_diff = np.r_[0.0, 0.5 * np.abs(H[1:] - H[:-1]).sum(axis=1)]
        pixel_diff = np.r_[0.0, np.abs(S[1:] - S[:-1]).mean(axis=(1, 2))]
        cuts = [c for c in find_cuts(np.asarray(times), hist_diff, pixel_diff, opts) if c > 0]
        bounds = [0.0, *cuts, duration]
        shots = [(round(a, 3), round(b, 3)) for a, b in pairwise(bounds) if b > a]

    save_json(cache_file, {"version": CACHE_VERSION, "opts": opts.__dict__, "shots": shots})
    return shots
