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
from pipeline.finder import evaluate, find_segments
from pipeline.frames import Crop, iter_frames
from pipeline.media import probe
from pipeline.players import Player
from pipeline.scenes import detect_shots
from pipeline.segments import MergeOptions, Segment, format_time, merge_segments, snap_to_shots
from pipeline.signals.commentary import find_mentions, transcribe, words_from_transcript
from pipeline.signals.face import face_hits, scan_faces
from pipeline.signals.scene_text import scan_scene_text
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
    scene_text: bool = typer.Option(True, help="Jersey ki peeth / frame mein likha naam"),
    model: str = "small",
    fast: bool = typer.Option(False, help="Commentary tez (beam 1)"),
    threshold: float = 5.0,
    cut_clips: bool = typer.Option(False, "--cut", help="Clips bhi bana do"),
    key: Path = typer.Option(None, help="Answer key — accuracy check ke liye"),
) -> None:
    """Video mein player ke moments dhoondo → segments + draft answer key (+ clips)."""
    from pipeline.finder import FindOptions

    pl = Player.load(player)
    vi = probe(video)
    console.print(f"Player: [bold]{pl.name}[/] — names: {', '.join(pl.all_names())}")

    with console.status("Camera cuts…"):
        shots_ = detect_shots(video)
    progress, cb = _progress_task("Scoreboard OCR")
    with progress:
        scan = scan_graphics(video, on_progress=cb)
    frames = parse_scan(scan)

    mentions = None
    if commentary:
        progress, cb = _progress_task(f"Commentary ({model})")
        with progress:
            tr = transcribe(video, model_size=model, hotwords=pl.all_names(),
                            beam_size=1 if fast else 5, on_progress=cb)
        mentions = find_mentions(words_from_transcript(tr), pl.all_names())
        console.print(f"Commentary mentions: {len(mentions)} (language={tr['language']})")

    hits = counts = None
    if face:
        progress, cb = _progress_task("Faces")
        with progress:
            fscan = scan_faces(video, pl, on_progress=cb)
        hits = face_hits(fscan)
        counts = {fr["t"]: len(fr["faces"]) for fr in fscan["frames"]}
        console.print(f"Face matches: {len(hits)} frames "
                      f"(reference: {', '.join(fscan['reference']['used'])})")

    stext = None
    if scene_text:
        progress, cb = _progress_task("Jersey / scene text")
        with progress:
            stext = scan_scene_text(video, shots_, on_progress=cb)

    segs, _ = find_segments(pl, frames, duration=vi.duration, shots=shots_, mentions=mentions,
                            faces=hits, face_counts=counts, scene_text=stext,
                            step=1 / scan["fps"], opts=FindOptions(threshold=threshold))

    t = Table(title=f"{pl.name} — {len(segs)} moments")
    for col in ("#", "start", "end", "sec", "score", "signals / events"):
        t.add_column(col)
    for i, s in enumerate(segs, 1):
        t.add_row(str(i), format_time(s.start), format_time(s.end), f"{s.duration:.0f}",
                  f"{s.score:.0f}", ", ".join(s.labels))
    console.print(t)
    console.print(f"Kul: {sum(s.duration for s in segs) / 60:.1f} min")

    draft = settings.samples_dir / "answer_keys" / f"{video.stem}__{pl.id}.draft.json"
    draft.write_text(json.dumps({
        "video": video.name, "player_id": pl.id, "status": "draft — check karke .draft hata dein",
        "segments": [{"start": format_time(s.start), "end": format_time(s.end),
                      "event": ",".join(s.labels)} for s in segs],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Draft answer key → {draft}")

    # Results (Phase 4 `make` command inhi se shorts banata hai)
    def focus_x(s: Segment) -> float | None:
        xs = [(h.box[0] + h.box[2] / 2) / vi.width for h in hits or []
              if s.start <= h.t <= s.end]
        return round(sorted(xs)[len(xs) // 2], 3) if xs else None

    res = _results_path(video, pl.id)
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps({
        "video": str(video.resolve()), "player_id": pl.id, "player_name": pl.name,
        "segments": [{**s.to_dict(), "focus_x": focus_x(s)} for s in segs],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if key:
        truth = [Segment.from_dict(d) for d in json.loads(key.read_text(encoding="utf-8"))["segments"]]
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
    music: Path = typer.Option(None, help="Background music (default: assets/music ki pehli file)"),
    music_volume: float = 0.35,
    transition: str = typer.Option("fade", help="fade / wipeleft / slideup / circleopen / ..."),
    min_score: float = typer.Option(0, help="Sirf is score se upar wale moments"),
    rerender: bool = typer.Option(False, help="Pehle se bane shorts bhi dobara banao"),
) -> None:
    """Phase 4: `find` ke moments se shorts + music wali compilation banao."""
    from pipeline.compose import (
        EVENT_TEXT,
        HORIZONTAL,
        VERTICAL,
        compile_video,
        render_clip,
        title_card,
    )

    pl = Player.load(player)
    res_file = _results_path(video, pl.id)
    if not res_file.exists():
        console.print(f"[red]Results nahi mile ({res_file.name}). Pehle `find` chalayen.[/]")
        raise typer.Exit(1)
    res = json.loads(res_file.read_text(encoding="utf-8"))
    res_mtime = res_file.stat().st_mtime
    segs = [s for s in res["segments"] if s["score"] >= min_score]
    if not segs:
        console.print("[yellow]Koi moment nahi.[/]")
        raise typer.Exit(1)

    if music is None:
        found = sorted(p for p in settings.music_dir.glob("*")
                       if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".mp4", ".webm"})
        music = found[0] if found else None
    console.print(f"{pl.name}: {len(segs)} moments, music: {music.name if music else '—'}")

    out_dir = settings.storage_dir / "output" / f"{video.stem[:30]}__{pl.id}"
    size = VERTICAL if vertical else HORIZONTAL

    def event_of(s: dict) -> str | None:
        return next((lbl for lbl in s["labels"] if lbl in EVENT_TEXT), None)

    rendered: list[Path] = []
    with Progress(console=console) as progress:
        if shorts:
            task = progress.add_task("Shorts (9:16)", total=len(segs))
            for i, s in enumerate(segs, 1):
                short = out_dir / "shorts" / f"{pl.id}_{i:02d}.mp4"
                # Results ke baad bana hua short dobara render nahi (mode badla ho to --rerender)
                if rerender or not short.exists() or short.stat().st_mtime < res_mtime:
                    render_clip(video, s["start"], s["end"], short,
                                size=VERTICAL, mode=mode, focus_x=s.get("focus_x"),
                                title=pl.name, event=event_of(s))
                progress.update(task, completed=i)

        if compilation:
            task = progress.add_task("Compilation clips", total=len(segs))
            parts = out_dir / "parts"
            # Shorts aur compilation ka size same ho to dobara render nahi
            for i, s in enumerate(segs, 1):
                p = out_dir / "shorts" / f"{pl.id}_{i:02d}.mp4"
                if not (shorts and vertical):
                    p = render_clip(video, s["start"], s["end"], parts / f"{i:02d}.mp4",
                                    size=size, mode=mode, focus_x=s.get("focus_x"),
                                    title=None, event=event_of(s))
                rendered.append(p)
                progress.update(task, completed=i)
            intro = title_card(parts / "intro.mp4", pl.name, "Best Moments", size=size)
            final = out_dir / f"{pl.id}_compilation{'' if vertical else '_16x9'}.mp4"
            with console.status("Final video (transitions + music)…"):
                compile_video([intro, *rendered], final, music=music,
                              music_volume=music_volume, transition=transition)
            console.print(f"[green]Compilation[/] → {final} "
                          f"({probe(final).duration / 60:.1f} min)")
    if shorts:
        console.print(f"[green]{len(segs)} shorts[/] → {out_dir / 'shorts'}")


if __name__ == "__main__":
    app()
