"""Face recognition — OpenCV YuNet (detection) + SFace (recognition). Dono Apache-2.0, CPU fast.

1. Player ki reference images se face embeddings
2. Video ke har sampled frame mein saare faces detect → har face ki reference se similarity
3. Raw nateeja (boxes + similarity) cache — threshold baad mein badal sakte hain bina dobara scan
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

from pipeline.cache import cache_dir, load_json, save_json
from pipeline.frames import iter_frames
from pipeline.models import model_path
from pipeline.players import Player

CACHE_VERSION = 2
DET_MODEL = "face_detection_yunet_2023mar.onnx"
REC_MODEL = "face_recognition_sface_2021dec.onnx"

# SFace cosine similarity: OpenCV ka recommended "same person" threshold 0.363
SIM_MATCH = 0.363
SIM_STRONG = 0.45


@cache
def _recognizer() -> cv2.FaceRecognizerSF:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    return cv2.FaceRecognizerSF.create(str(model_path(REC_MODEL)), "")


@cache
def _detector(w: int, h: int, score: float = 0.75) -> cv2.FaceDetectorYN:
    """Har (size, score) ke liye ek detector — har frame par naya banana slow tha."""
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    det = cv2.FaceDetectorYN.create(str(model_path(DET_MODEL)), "", (w, h), score, 0.3, 500)
    det.setInputSize((w, h))
    return det


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread Windows par non-ASCII paths aur .jfif/.webp mein masla karta hai."""
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect_faces(img: np.ndarray, min_size: int = 20, score: float = 0.75) -> np.ndarray:
    """Returns YuNet rows: [x, y, w, h, 10 landmark coords, score]."""
    h, w = img.shape[:2]
    _, faces = _detector(w, h, score).detect(img)
    if faces is None:
        return np.zeros((0, 15), np.float32)
    return faces[(faces[:, 2] >= min_size) & (faces[:, 3] >= min_size)]


def embed(img: np.ndarray, face: np.ndarray) -> np.ndarray:
    rec = _recognizer()
    aligned = rec.alignCrop(img, face)
    feat = rec.feature(aligned).ravel()
    return feat / (np.linalg.norm(feat) + 1e-9)


def kit_hue(img: np.ndarray, face: np.ndarray) -> float:
    """Chehre ke neeche (seena / kit) ka ghalib rang — OpenCV hue 0–180. -1 = pata nahi.

    Cricket mein chehra akela kamzor signal hai (helmet, chhota size), aur doosri team ke
    players bhi match ho jaate hain. Kit ka rang (Pakistan green vs India blue) unhein alag
    kar deta hai."""
    x, y, w, h = (float(v) for v in face[:4])
    x0, x1 = int(max(0, x - 0.4 * w)), int(min(img.shape[1], x + 1.4 * w))
    y0, y1 = int(min(img.shape[0], y + 1.3 * h)), int(min(img.shape[0], y + 2.8 * h))
    patch = img[y0:y1, x0:x1]
    if patch.size < 30:
        return -1.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    colored = hsv[(hsv[:, 1] > 70) & (hsv[:, 2] > 50)]
    if len(colored) < 0.2 * len(hsv):
        return -1.0  # zyada tar safed / kaala (pads, background)
    hist = np.bincount(colored[:, 0] // 6, minlength=30)
    return float(np.argmax(hist) * 6 + 3)


def hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


@dataclass
class Reference:
    embeddings: np.ndarray  # (n, 128) normalized
    used: list[str]  # kin images se face mila
    skipped: list[str]  # kin mein face nahi mila
    kit_hues: list[float]  # reference images mein kit ka rang

    @property
    def key(self) -> str:
        return hashlib.sha1(self.embeddings.tobytes()).hexdigest()[:10]


def build_reference(player: Player) -> Reference:
    """Har image ka sab se bara face. Chhoti images ko bara karke dobara try (helmet / door)."""
    embs, used, skipped, hues = [], [], [], []
    for path in player.images:
        img = read_image(path)
        if img is None:
            skipped.append(path.name)
            continue
        found = None
        for scale in (1.0, 2.0):
            im = img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale)
            faces = detect_faces(im, min_size=24, score=0.7)
            if len(faces):
                found = (im, faces[np.argmax(faces[:, 2] * faces[:, 3])])
                break
        if found is None:
            skipped.append(path.name)
            continue
        embs.append(embed(*found))
        used.append(path.name)
        if (hue := kit_hue(*found)) >= 0:
            hues.append(hue)
    if not embs:
        raise ValueError(f"{player.name} ki kisi image mein face nahi mila: {skipped}")
    return Reference(np.stack(embs), used, skipped, hues)


def similarity(ref: Reference, emb: np.ndarray) -> float:
    """Kisi bhi reference image se sab se zyada cosine similarity."""
    return float((ref.embeddings @ emb).max())


def scan_faces(
    path: str | Path,
    player: Player,
    *,
    fps: float = 2.0,
    width: int | None = None,
    use_cache: bool = True,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    """Returns {"fps", "reference": {...}, "frames": [{"t", "faces": [[x,y,w,h,det,sim,hue]]}]}.

    `width=None` = original resolution (720p par chhote faces bhi milte hain).
    Sirf woh frames save hote hain jin mein koi face mila.
    """
    ref = build_reference(player)
    cache_file = cache_dir(path) / f"faces_{player.id}_{ref.key}.json"
    if use_cache and isinstance(cached := load_json(cache_file), dict):
        if cached.get("version") == CACHE_VERSION and cached.get("fps") == fps:
            return cached

    from pipeline.media import probe

    duration = probe(path).duration
    t0 = time.perf_counter()
    out: list[dict] = []
    n = 0
    for ts, frame in iter_frames(path, fps=fps, width=width):
        n += 1
        faces = detect_faces(frame)
        if len(faces):
            rows = []
            for f in faces:
                sim = similarity(ref, embed(frame, f))
                rows.append([round(float(v), 1) for v in f[:4]] +
                            [round(float(f[14]), 3), round(sim, 3), kit_hue(frame, f)])
            out.append({"t": round(ts, 2), "faces": rows})
        if on_progress and duration:
            on_progress(min(ts / duration, 1.0))

    result = {
        "version": CACHE_VERSION,
        "fps": fps,
        "reference": {"used": ref.used, "skipped": ref.skipped, "kit_hues": ref.kit_hues},
        "frames_scanned": n,
        "elapsed": round(time.perf_counter() - t0, 1),
        "frames": out,
    }
    save_json(cache_file, result)
    return result


@dataclass(frozen=True)
class FaceHit:
    t: float
    sim: float
    box: tuple[float, float, float, float]  # x, y, w, h (pixels)


def kit_matches(hue: float, kit_hues: list[float], tol: float = 18.0) -> bool | None:
    """True / False — ya None agar rang pata nahi (tab kit se faisla nahi karte)."""
    if hue < 0 or not kit_hues:
        return None
    return any(hue_distance(hue, k) <= tol for k in kit_hues)


def face_hits(scan: dict, min_sim: float = SIM_MATCH, *, check_kit: bool = True) -> list[FaceHit]:
    """Har frame ka sab se mazboot match (agar `min_sim` se upar). `check_kit`: doosri team
    ke rang wali kit ho to reject."""
    kit = scan.get("reference", {}).get("kit_hues", [])
    hits = []
    for fr in scan["frames"]:
        ok = [f for f in fr["faces"] if f[5] >= min_sim
              and not (check_kit and kit_matches(f[6], kit) is False)]
        if ok:
            best = max(ok, key=lambda f: f[5])
            hits.append(FaceHit(fr["t"], best[5], tuple(best[:4])))
    return hits
