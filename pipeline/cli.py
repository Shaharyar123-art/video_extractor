"""Command-line tool. Chalayen:  uv run python -m pipeline.cli --help"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from app.config import settings
from pipeline.cutter import cut_segments
from pipeline.finder import evaluate
from pipeline.frames import Crop, iter_frames
from pipeline.media import probe
from pipeline.players import Player
from pipeline.scenes import detect_shots
from pipeline.segments import MergeOptions, Segment, format_time, merge_segments, snap_to_shots
from pipeline.service import AnalyzeOptions, RenderOptions, analyze, render
from pipeline.signals.commentary import transcribe
from pipeline.signals.scoreboard import parse_scan, scan_graphics

app = typer.Typer(help="VideoExtractor pipeline tools", no_args_is_help=True)

# Windows console (cp1252) Urdu/Hindi/arrows print nahi kar sakta
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
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


def _progress_task(label: str):
    """Rich progress bar context + 0–1 callback."""
    progress = Progress(console=console)
    task = progress.add_task(label, total=1.0)
    return progress, lambda f: progress.update(task, completed=f)


@app.command()
def graphics(video: Path, no_cache: bool = False) -> None:
    """Scoreboard / cards OCR — kaun kaun se naam aaye (batsmen, bowlers, cards)."""
    progress, cb = _progress_task("Scoreboard OCR")
    with progress:
        scan = scan_graphics(video, use_cache=not no_cache, on_progress=cb)
    st = scan["stats"]
    console.print(f"[green]{st['frames']} frames[/], OCR {st['ocr_runs']} dafa, {st['elapsed']}s")
    frames = parse_scan(scan)
    bats = Counter(b for g in frames for b in g.batsmen)
    bowl = Counter(g.bowler for g in frames if g.bowler)
    events = Counter(g.event for g in frames if g.event)
    with_sb = sum(g.has_scoreboard for g in frames)
    console.print(f"Scoreboard dikha: {with_sb}/{len(frames)} frames")
    console.print("[bold]Batsmen:[/]", ", ".join(f"{k} ({v})" for k, v in bats.most_common(25)))
    console.print("[bold]Bowlers:[/]", ", ".join(f"{k} ({v})" for k, v in bowl.most_common(15)))
    console.print("[bold]Events:[/]", ", ".join(f"{k} ({v})" for k, v in events.most_common()))


@app.command(name="transcribe")
def transcribe_cmd(
    video: Path,
    model: str = typer.Option("small", help="tiny / base / small / medium / large-v3"),
    player: Path = typer.Option(None, help="Player folder — naam Whisper ko hint ke taur par"),
    language: str = typer.Option(None, help="en / ur / hi (khali = auto-detect)"),
    fast: bool = typer.Option(False, help="~1.5x tez (beam 1), kuch naam chhoot sakte hain"),
    no_cache: bool = False,
) -> None:
    """Commentary → text (faster-whisper). Pehli dafa model download hoga."""
    hot = Player.load(player).all_names() if player else None
    progress, cb = _progress_task(f"Whisper ({model})")
    with progress:
        tr = transcribe(video, model_size=model, hotwords=hot, language=language,
                        beam_size=1 if fast else 5, use_cache=not no_cache, on_progress=cb)
    console.print(f"[green]{len(tr['words'])} words[/], language={tr['language']} "
                  f"({tr['language_probability']}), {tr['elapsed']}s on {tr['device']}")
    preview = " ".join(w[2] for w in tr["words"][:60])
    console.print(f"[dim]{preview} …[/]")


@app.command()
def find(
    video: Path,
    player: Path = typer.Option(..., help="Player folder, e.g. ../samples/players/abdul_qadir"),
    commentary: bool = typer.Option(True, help="Whisper commentary signal bhi use karo"),
    face: bool = typer.Option(True, help="Face recognition signal (player images se)"),
    scene_text: bool = typer.Option(False, help="Jersey ki peeth / frame mein likha naam"),
    model: str = "small",
    fast: bool = typer.Option(False, help="Commentary tez (beam 1)"),
    threshold: float = 5.0,
    cut_clips: bool = typer.Option(False, "--cut", help="Raw clips bhi bana do"),
    key: Path = typer.Option(None, help="Answer key — accuracy check ke liye"),
) -> None:
    """Video mein player ke moments dhoondo → table + results + draft answer key."""
    pl = Player.load(player)
    console.print(f"Player: [bold]{pl.name}[/] — names: {', '.join(pl.all_names())}")
    opts = AnalyzeOptions(commentary=commentary, whisper_model=model, fast_commentary=fast,
                          face=face, scene_text=scene_text, threshold=threshold)
    with Progress(console=console) as progress:
        task = progress.add_task("Analyze", total=1.0)

        def cb(stage: str, frac: float, msg: str) -> None:
            progress.update(task, completed=frac, description=f"{stage}: {msg}"[:40])

        res = analyze(video, pl, opts, cb)
    stats = res["stats"]
    console.print(f"Shots {stats['shots']}, commentary mentions {stats['commentary_mentions']}, "
                  f"face matches {stats['face_matches']}")
    segs = [Segment(s["start"], s["end"], s["score"], s["labels"]) for s in res["segments"]]

    t = Table(title=f"{pl.name} — {len(segs)} moments")
    for col in ("#", "start", "end", "sec", "score", "signals / events"):
        t.add_column(col)
    for i, s in enumerate(segs, 1):
        t.add_row(str(i), format_time(s.start), format_time(s.end), f"{s.duration:.0f}",
                  f"{s.score:.0f}", ", ".join(s.labels))
    console.print(t)
    console.print(f"Kul: {sum(s.duration for s in segs) / 60:.1f} min")

    res_file = _results_path(video, pl.id)
    res_file.parent.mkdir(parents=True, exist_ok=True)
    res_file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    draft = settings.samples_dir / "answer_keys" / f"{video.stem}__{pl.id}.draft.json"
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.write_text(json.dumps({
        "video": video.name, "player_id": pl.id, "status": "draft — check karke .draft hata dein",
        "segments": [{"start": format_time(s.start), "end": format_time(s.end),
                      "event": ",".join(s.labels)} for s in segs],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Draft answer key → {draft}")

    if key:
        truth = [Segment.from_dict(d)
                 for d in json.loads(key.read_text(encoding="utf-8"))["segments"]]
        r = evaluate(segs, truth)
        console.print(f"[bold]Accuracy:[/] precision {r.precision:.0%}, recall {r.recall:.0%}, "
                      f"F1 {r.f1:.0%}, moments pakre {r.hits}/{r.total}")

    if cut_clips and segs:
        out = settings.storage_dir / "clips" / f"{video.stem[:30]}__{pl.id}"
        with Progress(console=console) as progress:
            task = progress.add_task("Cutting", total=len(segs))
            cut_segments(video, segs, out, prefix=pl.id,
                         on_progress=lambda i, n: progress.update(task, completed=i))
        console.print(f"[green]{len(segs)} clips[/] → {out}")


def _results_path(video: Path, player_id: str) -> Path:
    return settings.storage_dir / "results" / f"{video.stem[:40]}__{player_id}.json"


@app.command()
def make(
    video: Path,
    player: Path = typer.Option(..., help="Player folder (pehle `find` chalayen)"),
    shorts: bool = typer.Option(True, help="Har moment ka 9:16 short"),
    compilation: bool = typer.Option(True, help="Sab moments ki ek final video"),
    vertical: bool = typer.Option(True, help="Compilation 9:16 (False = 16:9 YouTube)"),
    mode: str = typer.Option("blur", help="blur (poora frame + blurred bg) / crop (player par)"),
    music: str = typer.Option(None, help="assets/music mein file ka naam ('' = music nahi)"),
    music_volume: float = 0.35,
    transition: str = typer.Option("fade", help="fade / wipeleft / slideup / circleopen / ..."),
    min_score: float = typer.Option(0, help="Sirf is score se upar wale moments"),
    rerender: bool = typer.Option(False, help="Pehle se bane shorts bhi dobara banao"),
) -> None:
    """Phase 4: `find` ke moments se shorts + music wali compilation banao."""
    pl = Player.load(player)
    res_file = _results_path(video, pl.id)
    if not res_file.exists():
        console.print(f"[red]Results nahi mile ({res_file.name}). Pehle `find` chalayen.[/]")
        raise typer.Exit(1)
    res = json.loads(res_file.read_text(encoding="utf-8"))
    segs = [s for s in res["segments"] if s["score"] >= min_score]
    if not segs:
        console.print("[yellow]Koi moment nahi.[/]")
        raise typer.Exit(1)

    out_dir = settings.storage_dir / "output" / f"{video.stem[:30]}__{pl.id}"
    opts = RenderOptions(shorts=shorts, compilation=compilation, vertical=vertical, mode=mode,
                         music=music, music_volume=music_volume, transition=transition,
                         rerender=rerender)
    with Progress(console=console) as progress:
        task = progress.add_task("Render", total=1.0)

        def cb(stage: str, frac: float, msg: str) -> None:
            progress.update(task, completed=frac, description=f"{stage}: {msg}"[:40])

        r = render(video, pl.name, segs, out_dir, opts, cb)
    console.print(f"{pl.name}: {len(segs)} moments, music: {r.music or '—'}")
    if r.shorts:
        console.print(f"[green]{len(r.shorts)} shorts[/] → {out_dir / 'shorts'}")
    if r.compilation:
        console.print(f"[green]Compilation[/] → {r.compilation} "
                      f"({probe(Path(r.compilation)).duration / 60:.1f} min)")


if __name__ == "__main__":
    app()
