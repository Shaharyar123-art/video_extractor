"""Command-line tool (Phase 1). Chalayen:  uv run python -m pipeline.cli --help"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from app.config import settings
from pipeline.cutter import cut_segments
from pipeline.frames import Crop, iter_frames
from pipeline.media import probe
from pipeline.scenes import detect_shots
from pipeline.segments import MergeOptions, Segment, format_time, merge_segments, snap_to_shots

app = typer.Typer(help="VideoExtractor pipeline tools", no_args_is_help=True)
console = Console()


@app.command()
def info(video: Path) -> None:
    """Video ki details (resolution, duration, fps, audio)."""
    vi = probe(video)
    t = Table(show_header=False)
    for k, v in vi.to_dict().items():
        t.add_row(k, str(v))
    t.add_row("duration (mm:ss)", format_time(vi.duration))
    console.print(t)


@app.command()
def shots(video: Path, no_cache: bool = False) -> None:
    """Camera cuts / shots detect karo."""
    t0 = time.perf_counter()
    with console.status("Camera cuts dhoondh rahe hain…"):
        result = detect_shots(video, use_cache=not no_cache)
    dt = time.perf_counter() - t0
    lengths = [e - s for s, e in result]
    console.print(f"[green]{len(result)} shots[/] in {dt:.1f}s "
                  f"(avg {sum(lengths) / len(lengths):.1f}s, max {max(lengths):.1f}s)")
    for i, (s, e) in enumerate(result[:15], 1):
        console.print(f"  {i:3d}. {format_time(s)} → {format_time(e)}  ({e - s:.1f}s)")
    if len(result) > 15:
        console.print(f"  … aur {len(result) - 15}")


@app.command()
def frames(
    video: Path,
    out: Path = typer.Option(None, help="Output folder (default: storage/frames/<video>)"),
    fps: float = 0.5,
    start: float = 0.0,
    end: float = typer.Option(None),
    width: int = 960,
    crop: str = typer.Option(None, help="Relative crop 'x,y,w,h', e.g. scoreboard: 0,0.85,1,0.15"),
) -> None:
    """Frames JPG mein save karo (debug / dekhne ke liye)."""
    out = out or settings.storage_dir / "frames" / video.stem[:40]
    out.mkdir(parents=True, exist_ok=True)
    c = Crop(*map(float, crop.split(","))) if crop else None
    n = 0
    t0 = time.perf_counter()
    for ts, frame in iter_frames(video, fps=fps, width=width, crop=c, start=start, end=end):
        cv2.imwrite(str(out / f"{int(ts * 1000):08d}.jpg"), frame)
        n += 1
    console.print(f"[green]{n} frames[/] → {out}  ({time.perf_counter() - t0:.1f}s)")


@app.command()
def cut(
    video: Path,
    times: Path = typer.Option(..., help="JSON: answer-key format ya [{start, end}, ...]"),
    out: Path = typer.Option(None, help="Output folder (default: storage/clips/<video>)"),
    pad_before: float = 1.0,
    pad_after: float = 1.0,
    snap: bool = typer.Option(True, help="Clip ko qareebi camera cut par align karo"),
) -> None:
    """Diye gaye timestamps se clips banao."""
    data = json.loads(times.read_text(encoding="utf-8"))
    raw = data["segments"] if isinstance(data, dict) else data
    segs = [Segment.from_dict(d) for d in raw]
    vi = probe(video)

    merged = merge_segments(segs, vi.duration, MergeOptions(pad_before=pad_before, pad_after=pad_after))
    if snap:
        merged = snap_to_shots(merged, detect_shots(video))

    out = out or settings.storage_dir / "clips" / video.stem[:40]
    with Progress(console=console) as progress:
        task = progress.add_task("Cutting", total=len(merged))
        paths = cut_segments(video, merged, out,
                             on_progress=lambda i, n: progress.update(task, completed=i))
    for seg, p in zip(merged, paths, strict=True):
        console.print(f"  {format_time(seg.start)} → {format_time(seg.end)}  {p.name}")
    console.print(f"[green]{len(paths)} clips[/] → {out}")


if __name__ == "__main__":
    app()
