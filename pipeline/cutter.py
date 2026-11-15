"""Segments ko video clips mein kaatna (ffmpeg)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pipeline.media import run_ffmpeg
from pipeline.segments import Segment


def cut_clip(src: str | Path, start: float, end: float, out: str | Path, *, crf: int = 20) -> Path:
    """Frame-accurate cut (re-encode). Stream-copy tez hota hai lekin keyframe par hi katta hai,
    jis se clip ke shuru mein 1–5 sec ghalat aa jaate hain — is liye re-encode."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out),
    ])
    return out


def cut_segments(
    src: str | Path,
    segments: list[Segment],
    out_dir: str | Path,
    *,
    prefix: str = "clip",
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    paths = []
    for i, seg in enumerate(segments, 1):
        name = f"{prefix}_{i:03d}_{int(seg.start):05d}s.mp4"
        paths.append(cut_clip(src, seg.start, seg.end, out_dir / name))
        if on_progress:
            on_progress(i, len(segments))
    return paths
