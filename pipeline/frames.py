"""Video se frames nikalna — ffmpeg pipe ke zariye (OpenCV seeking se bohat tez).

ffmpeg khud fps kam karta hai, resize/crop karta hai, aur raw BGR frames bhejta hai.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings
from pipeline.media import FFmpegError, probe


@dataclass(frozen=True)
class Crop:
    """Relative crop (0–1), e.g. scoreboard: Crop(0, 0.85, 1, 0.15)."""

    x: float
    y: float
    w: float
    h: float


def iter_frames(
    path: str | Path,
    *,
    fps: float | None = None,
    width: int | None = 640,
    crop: Crop | None = None,
    start: float = 0.0,
    end: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yields (timestamp_seconds, BGR frame). `width=None` = original size."""
    fps = fps or settings.sample_fps
    info = probe(path)

    # Output frame size calculate karo
    src_w, src_h = info.width, info.height
    filters: list[str] = [f"fps={fps}"]
    if crop:
        cw, ch = int(src_w * crop.w) // 2 * 2, int(src_h * crop.h) // 2 * 2
        cx, cy = int(src_w * crop.x), int(src_h * crop.y)
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
        src_w, src_h = cw, ch
    if width and width < src_w:
        out_w = width // 2 * 2
        out_h = round(src_h * out_w / src_w / 2) * 2
        filters.append(f"scale={out_w}:{out_h}")
    else:
        out_w, out_h = src_w, src_h

    args = [settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    if start > 0:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if end is not None:
        args += ["-t", f"{max(end - start, 0):.3f}"]
    args += ["-vf", ",".join(filters), "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    frame_bytes = out_w * out_h * 3
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    count = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
            yield start + count / fps, frame
            count += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        if proc.poll() is None:  # consumer ne beech mein chhor diya
            proc.kill()
        proc.wait()
    if count == 0 and proc.returncode != 0:
        raise FFmpegError(f"Frame extraction failed for {path} (ffmpeg exit {proc.returncode})")
