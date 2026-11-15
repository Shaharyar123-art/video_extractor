"""ffmpeg / ffprobe helpers: video ki info nikalna aur ffmpeg commands chalana."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import settings


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    duration: float  # seconds
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_rate(rate: str) -> float:
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    proc = subprocess.run(
        [settings.ffprobe_bin, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vs is None:
        raise FFmpegError(f"No video stream in {path}")
    duration = float(data.get("format", {}).get("duration") or vs.get("duration") or 0)
    fps = _parse_rate(vs.get("avg_frame_rate") or "0/1") or _parse_rate(vs.get("r_frame_rate", "0/1"))
    return VideoInfo(
        path=str(path),
        duration=duration,
        width=int(vs["width"]),
        height=int(vs["height"]),
        fps=round(fps, 3),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        video_codec=vs.get("codec_name", ""),
    )


def run_ffmpeg(args: list[str]) -> None:
    """ffmpeg chalao; fail ho to stderr ke saath error."""
    cmd = [settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed: {proc.stderr.strip()}\ncmd: {' '.join(cmd)}")
