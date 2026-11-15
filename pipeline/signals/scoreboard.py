"""Scoreboard / graphics OCR.

Do hisse:
1. `scan_graphics` — video ke neeche wale hisse (graphics band) ka OCR, raw boxes cache mein.
   Speed ke liye: agar pichhle OCR wale text boxes ke pixels nahi badle to OCR dobara nahi
   chalta (scoreboard sirf har ball ke baad badalta hai).
2. `parse_frame` — boxes ko roles mein baanto: batsmen (striker kaun), bowler, event label
   (FOUR / SIX / WICKET), aur cards (player name card, dismissal card).

Layout default ICC / ACC broadcast graphics ke liye hai (`Layout`); doosre broadcasters ke liye
naya Layout bana sakte hain.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.cache import cache_dir, load_json, save_json
from pipeline.frames import Crop, iter_frames
from pipeline.signals.ocr import TextBox, read_text

CACHE_VERSION = 1

# Graphics band: frame ke neeche ka 26% (scoreboard bar + upar aane wale cards)
BAND = Crop(0.0, 0.72, 1.0, 0.26)


@dataclass(frozen=True)
class Layout:
    """Band ke andar relative zones (0–1). Default = ICC/ACC 2025-26 graphics."""

    bar_top: float = 0.55  # is se neeche (cy) = scoreboard bar, upar = cards
    card_top: float = 0.5  # koi text is se upar ho = card frame (poora band card hai)
    bar_marker: str = "THIS OVER"  # yeh text ho to frame normal scoreboard bar hai
    batsmen_x: tuple[float, float] = (0.225, 0.47)  # names ka x0 / cx range
    center_x: tuple[float, float] = (0.49, 0.645)  # event / info box
    bowler_x: tuple[float, float] = (0.645, 0.80)
    # Striker ka bat-icon is column mein hota hai (names ke theek baayein). OCR kabhi icon ko
    # naam ke box mein mila deta hai, is liye box ke bajaye fixed column check karte hain.
    icon_x: tuple[float, float] = (0.237, 0.254)


EVENTS = {
    "FOUR": "four", "SIX": "six", "WICKET": "wicket", "OUT": "wicket",
    "NOT OUT": "not_out", "DROPPED": "dropped", "CAUGHT": "wicket",
    "FIFTY": "fifty", "HUNDRED": "hundred", "CENTURY": "hundred",
    "BOWLED": "wicket", "LBW": "wicket", "RUN OUT": "wicket", "STUMPED": "wicket",
    "WIDE": "wide", "NO BALL": "no_ball", "FREE HIT": "free_hit",
}

_ALPHA = re.compile(r"[A-Za-z]{2,}")
_SCORE = re.compile(r"\d{1,3}-\d{1,2}")
_OVERS = re.compile(r"\d{1,2}\.[0-5]")


# ---------------------------------------------------------------- scanning


@dataclass
class ScanStats:
    frames: int = 0
    ocr_runs: int = 0
    elapsed: float = 0.0


def _signature(gray: np.ndarray, boxes: list[TextBox]) -> np.ndarray | None:
    """Text boxes ke andar ke pixels (chhote karke) — change detection ke liye."""
    if not boxes:
        return None
    parts = []
    for b in boxes:
        y0, y1 = int(b.y0), max(int(b.y1), int(b.y0) + 1)
        x0, x1 = int(b.x0), max(int(b.x1), int(b.x0) + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size:
            parts.append(cv2.resize(patch, (16, 6), interpolation=cv2.INTER_AREA).ravel())
    return np.concatenate(parts).astype(np.float32) if parts else None


def _icon_brightness(frame: np.ndarray, box: TextBox, layout: Layout) -> float:
    """Box ki row mein, icon column ke safed pixels ka hissa (striker bat-icon)."""
    w = frame.shape[1]
    x0, x1 = int(layout.icon_x[0] * w), int(layout.icon_x[1] * w)
    y0, y1 = int(box.y0), int(box.y1)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float((patch.min(axis=2) > 200).mean())


def scan_graphics(
    path: str | Path,
    *,
    fps: float = 1.0,
    max_reuse: float = 3.0,
    change_threshold: float = 12.0,
    layout: Layout | None = None,
    use_cache: bool = True,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    """Returns {"fps", "band", "width", "height", "frames": [{"t", "boxes", "left"}], "stats"}.

    `max_reuse`: itne seconds se zyada purana OCR result reuse nahi hoga (naye cards pakarne ke
    liye). `change_threshold`: text-box pixels ka ausat farq (0–255) jis se OCR dobara chale.
    """
    layout = layout or Layout()
    cache_file = cache_dir(path) / "graphics_ocr.json"
    if use_cache and isinstance(cached := load_json(cache_file), dict):
        if cached.get("version") == CACHE_VERSION and cached.get("fps") == fps:
            return cached

    from pipeline.media import probe

    duration = probe(path).duration
    stats = ScanStats()
    t_start = time.perf_counter()
    frames_out: list[dict] = []
    last_boxes: list[TextBox] = []
    last_sig: np.ndarray | None = None
    last_ocr_t = -1e9
    h = w = 0

    for ts, frame in iter_frames(path, fps=fps, width=None, crop=BAND):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats.frames += 1

        reuse = False
        if last_boxes and ts - last_ocr_t <= max_reuse:
            sig = _signature(gray, last_boxes)
            if sig is not None and last_sig is not None and sig.shape == last_sig.shape:
                reuse = float(np.abs(sig - last_sig).mean()) < change_threshold

        if not reuse:
            last_boxes = read_text(frame)
            last_sig = _signature(gray, last_boxes)
            last_ocr_t = ts
            stats.ocr_runs += 1

        frames_out.append({
            "t": round(ts, 2),
            "boxes": [b.to_list() for b in last_boxes],
            "icon": [round(_icon_brightness(frame, b, layout), 3) for b in last_boxes],
        })
        if on_progress and duration:
            on_progress(min(ts / duration, 1.0))

    stats.elapsed = round(time.perf_counter() - t_start, 1)
    result = {
        "version": CACHE_VERSION, "fps": fps, "band": asdict(BAND),
        "width": w, "height": h, "frames": frames_out, "stats": asdict(stats),
    }
    save_json(cache_file, result)
    return result


# ---------------------------------------------------------------- parsing


@dataclass
class FrameGraphics:
    t: float
    batsmen: list[str] = field(default_factory=list)
    striker: str | None = None
    bowler: str | None = None
    event: str | None = None
    center_text: str = ""
    cards: list[str] = field(default_factory=list)  # name card / dismissal card lines
    team_score: str | None = None  # '319-7'
    overs: str | None = None  # '47.5' — highlights mein naya over/ball = nayi delivery

    @property
    def has_scoreboard(self) -> bool:
        return bool(self.batsmen or self.bowler)


def _name_text(text: str) -> str | None:
    """'NIQAB' / 'ABDUL QADIR' haan; '3-56 8.5', 'P3', 'THIS OVER' nahi."""
    t = text.strip()
    if not _ALPHA.search(t) or t.upper() in {"THIS OVER", "THIS", "OVER"}:
        return None
    if re.fullmatch(r"P\d", t, re.IGNORECASE):
        return None
    return t


def _event_from(text: str) -> str | None:
    up = re.sub(r"\s+", " ", text.upper()).strip()
    if up.startswith("LAST WICKET"):  # maloomati box, event nahi
        return None
    for key, ev in EVENTS.items():
        # Lambe words animation ke saath chipke aa sakte hain ('CFOUR') → substring kaafi.
        # Chhote ('OUT', 'LBW', 'SIX') ke liye poora word chahiye ('ABOUT' match na ho).
        pattern = key if len(key) >= 4 else rf"\b{key}\b"
        if re.search(pattern, up):
            return ev
    return None


def parse_frame(rec: dict, width: int, height: int, layout: Layout | None = None) -> FrameGraphics:
    layout = layout or Layout()
    boxes = [TextBox.from_list(b) for b in rec["boxes"]]
    icon = rec.get("icon", [0.0] * len(boxes))
    g = FrameGraphics(t=rec["t"])

    # Card frame (player name card / dismissal card): saara text card ka hai.
    # Lekin agar bar ka marker ('THIS OVER') maujood hai to yeh normal scoreboard hai — upar ka
    # text video ke andar ke logos / jerseys hain ('GATORA', 'apollo'), ignore.
    has_bar = any(layout.bar_marker in b.text.upper() for b in boxes)
    if not has_bar and any(b.cy / height < layout.card_top for b in boxes):
        g.cards = [b.text for b in sorted(boxes, key=lambda b: (b.cy, b.x0))
                   if _ALPHA.search(b.text)]
        return g

    bats: list[tuple[TextBox, float]] = []
    bowler_boxes: list[TextBox] = []
    center: list[TextBox] = []

    for b, lb in zip(boxes, icon, strict=True):
        rx0, rcx, rcy = b.x0 / width, b.cx / width, b.cy / height
        if rcy < layout.bar_top:
            continue  # bar ke upar ka text (video background) — ignore
        if layout.batsmen_x[0] <= rx0 and rcx < layout.batsmen_x[1]:
            if _name_text(b.text):
                bats.append((b, lb))
        elif layout.center_x[0] <= rcx < layout.center_x[1]:
            center.append(b)
        elif layout.bowler_x[0] <= rcx < layout.bowler_x[1]:
            if _name_text(b.text):
                bowler_boxes.append(b)
        elif rcx < layout.batsmen_x[0]:
            # Team zone: '319-7' (score) aur '47.5' (overs)
            txt = b.text.replace(" ", "")
            if _SCORE.fullmatch(txt):
                g.team_score = txt
            elif _OVERS.fullmatch(txt):
                g.overs = txt

    bats.sort(key=lambda x: x[0].cy)
    g.batsmen = [b.text for b, _ in bats]
    if len(bats) == 2:
        (b1, l1), (b2, l2) = bats
        if max(l1, l2) >= 0.05 and abs(l1 - l2) >= 0.04:
            g.striker = b1.text if l1 > l2 else b2.text
    elif len(bats) == 1 and bats[0][1] >= 0.08:
        g.striker = bats[0][0].text

    if bowler_boxes:
        g.bowler = min(bowler_boxes, key=lambda b: b.cy).text  # upar wali line = bowler

    center.sort(key=lambda b: (b.cy, b.x0))
    g.center_text = " ".join(b.text for b in center)
    g.event = _event_from(g.center_text)
    return g


def parse_scan(scan: dict, layout: Layout | None = None) -> list[FrameGraphics]:
    return [parse_frame(r, scan["width"], scan["height"], layout) for r in scan["frames"]]
