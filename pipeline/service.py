"""Pipeline ke do bare kaam — CLI aur API (worker) dono yahi functions use karte hain.

analyze(video, player, opts)  → moments (segments) dhoondo   [Phase 2 + 3]
render(video, player, segs)   → shorts + compilation banao    [Phase 4]

Progress callback: on_progress(stage: str, fraction: float 0–1, message: str).
Cancel: callback `CancelledError` raise kar sakta hai — beech mein ruk jayega.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import settings
from pipeline.compose import (
    EVENT_TEXT,
    HORIZONTAL,
    VERTICAL,
    compile_video,
    render_clip,
    title_card,
)
from pipeline.finder import FindOptions, find_segments
from pipeline.media import probe
from pipeline.players import Player
from pipeline.scenes import detect_shots
from pipeline.signals.commentary import find_mentions, transcribe, words_from_transcript
from pipeline.signals.face import face_hits, scan_faces
from pipeline.signals.scene_text import scan_scene_text
from pipeline.signals.scoreboard import parse_scan, scan_graphics

ProgressFn = Callable[[str, float, str], None]

MUSIC_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".mp4", ".webm"}


class CancelledError(Exception):
    pass


def _noop(stage: str, frac: float, msg: str) -> None:
    pass


@dataclass
class AnalyzeOptions:
    commentary: bool = True
    whisper_model: str = "small"
    fast_commentary: bool = False
    face: bool = True
    scene_text: bool = False  # ~10 min / 25 min video, chhota faida — default band
    threshold: float = 5.0


# Har stage ka overall progress mein hissa (CPU par andaze se waqt)
STAGE_WEIGHTS = {"shots": 5, "scoreboard": 25, "commentary": 45, "faces": 20, "scene_text": 10,
                 "matching": 1}


class _Stages:
    """Stage progress (0–1) ko overall progress (0–1) mein badalta hai."""

    def __init__(self, active: list[str], cb: ProgressFn) -> None:
        total = sum(STAGE_WEIGHTS[s] for s in active)
        self.offsets: dict[str, tuple[float, float]] = {}
        acc = 0.0
        for s in active:
            w = STAGE_WEIGHTS[s] / total
            self.offsets[s] = (acc, w)
            acc += w
        self.cb = cb

    def fn(self, stage: str, message: str = "") -> Callable[[float], None]:
        start, width = self.offsets[stage]

        def report(frac: float) -> None:
            self.cb(stage, start + width * min(max(frac, 0.0), 1.0), message)

        report(0.0)
        return report


def analyze(
    video: str | Path,
    player: Player,
    opts: AnalyzeOptions | None = None,
    on_progress: ProgressFn = _noop,
) -> dict:
    """Video mein player ke moments. Returns JSON-able dict (results)."""
    video = Path(video)
    opts = opts or AnalyzeOptions()
    vi = probe(video)
    active = ["shots", "scoreboard"]
    active += ["commentary"] if opts.commentary else []
    active += ["faces"] if opts.face else []
    active += ["scene_text"] if opts.scene_text else []
    active += ["matching"]
    st = _Stages(active, on_progress)

    p = st.fn("shots", "Camera cuts dhoondh rahe hain")
    shots = detect_shots(video)
    p(1.0)

    scan = scan_graphics(video, on_progress=st.fn("scoreboard", "Scoreboard OCR"))
    frames = parse_scan(scan)

    mentions = None
    language = None
    if opts.commentary:
        tr = transcribe(video, model_size=opts.whisper_model, hotwords=player.all_names(),
                        beam_size=1 if opts.fast_commentary else 5,
                        on_progress=st.fn("commentary", "Commentary sun rahe hain (Whisper)"))
        mentions = find_mentions(words_from_transcript(tr), player.all_names())
        language = tr["language"]

    hits = counts = None
    if opts.face:
        fscan = scan_faces(video, player, on_progress=st.fn("faces", "Chehre pehchan rahe hain"))
        hits = face_hits(fscan)
        counts = {fr["t"]: len(fr["faces"]) for fr in fscan["frames"]}

    stext = None
    if opts.scene_text:
        stext = scan_scene_text(video, shots, on_progress=st.fn("scene_text", "Jersey text"))

    p = st.fn("matching", "Moments jor rahe hain")
    segs, _ = find_segments(player, frames, duration=vi.duration, shots=shots, mentions=mentions,
                            faces=hits, face_counts=counts, scene_text=stext,
                            step=1 / scan["fps"], opts=FindOptions(threshold=opts.threshold))
    p(1.0)

    def focus_x(start: float, end: float) -> float | None:
        xs = sorted((h.box[0] + h.box[2] / 2) / vi.width for h in hits or []
                    if start <= h.t <= end)
        return round(xs[len(xs) // 2], 3) if xs else None

    return {
        "video": str(video.resolve()),
        "duration": vi.duration,
        "player_id": player.id,
        "player_name": player.name,
        "options": asdict(opts),
        "stats": {
            "shots": len(shots),
            "commentary_mentions": len(mentions) if mentions is not None else None,
            "commentary_language": language,
            "face_matches": len(hits) if hits is not None else None,
        },
        "segments": [
            {"id": i, **s.to_dict(), "focus_x": focus_x(s.start, s.end), "include": True}
            for i, s in enumerate(segs, 1)
        ],
    }


@dataclass
class RenderOptions:
    shorts: bool = True
    compilation: bool = True
    vertical: bool = True  # compilation 9:16 (False = 16:9)
    mode: str = "blur"  # blur / crop
    music: str | None = None  # file name in music_dir, "" = koi music nahi, None = pehli file
    music_volume: float = 0.35
    transition: str = "fade"
    title: str | None = None  # intro card (default: player ka naam)
    subtitle: str = "Best Moments"
    rerender: bool = False


def list_music() -> list[Path]:
    if not settings.music_dir.exists():
        return []
    return sorted(p for p in settings.music_dir.iterdir() if p.suffix.lower() in MUSIC_EXT)


def resolve_music(name: str | None) -> Path | None:
    if name == "":
        return None
    files = list_music()
    if name is None:
        return files[0] if files else None
    match = next((p for p in files if p.name == name), None)
    if match is None:
        raise FileNotFoundError(f"Music '{name}' assets/music mein nahi mila")
    return match


def event_of(seg: dict) -> str | None:
    return next((lbl for lbl in seg.get("labels", []) if lbl in EVENT_TEXT), None)


@dataclass
class RenderResult:
    shorts: list[str] = field(default_factory=list)
    compilation: str | None = None
    music: str | None = None


def render(
    video: str | Path,
    player_name: str,
    segments: list[dict],
    out_dir: str | Path,
    opts: RenderOptions | None = None,
    on_progress: ProgressFn = _noop,
) -> RenderResult:
    """Segments (dicts: start, end, labels, focus_x, include) → shorts + compilation."""
    video, out_dir = Path(video), Path(out_dir)
    opts = opts or RenderOptions()
    segs = [s for s in segments if s.get("include", True)]
    if not segs:
        raise ValueError("Koi moment select nahi")
    music = resolve_music(opts.music)
    result = RenderResult(music=music.name if music else None)

    # Final compile (sab clips dobara encode + transitions) sab se bhari qadam hai — uska
    # hissa clips ki tadaad jitna (warna progress 95% par minton ruka rehta tha)
    final_w = len(segs)
    n_steps = (len(segs) if opts.shorts else 0) + (len(segs) + 1 + final_w if opts.compilation
                                                    else 0)
    done = 0

    def step(stage: str, msg: str, weight: int = 1) -> None:
        nonlocal done
        done += weight
        on_progress(stage, done / max(n_steps, 1), msg)

    on_progress("shorts" if opts.shorts else "compilation", 0.0, "Shuru")
    shorts_dir = out_dir / "shorts"
    for i, s in enumerate(segs if opts.shorts else [], 1):
        short = shorts_dir / f"short_{i:02d}_{int(s['start']):05d}s.mp4"
        if opts.rerender or not short.exists():
            render_clip(video, s["start"], s["end"], short, size=VERTICAL, mode=opts.mode,
                        focus_x=s.get("focus_x"), title=player_name, event=event_of(s))
        result.shorts.append(str(short))
        step("shorts", f"Short {i}/{len(segs)}")

    if opts.compilation:
        size = VERTICAL if opts.vertical else HORIZONTAL
        parts_dir = out_dir / ("parts_v" if opts.vertical else "parts_h")
        clips: list[Path] = []
        for i, s in enumerate(segs, 1):
            if opts.shorts and opts.vertical:
                clips.append(Path(result.shorts[i - 1]))  # same format → reuse
            else:
                part = parts_dir / f"{i:02d}_{int(s['start']):05d}s.mp4"
                if opts.rerender or not part.exists():
                    render_clip(video, s["start"], s["end"], part, size=size, mode=opts.mode,
                                focus_x=s.get("focus_x"), title=None, event=event_of(s))
                clips.append(part)
            step("compilation", f"Clip {i}/{len(segs)}")
        intro = title_card(parts_dir / "intro.mp4", opts.title or player_name, opts.subtitle,
                           size=size)
        step("compilation", "Final video ban rahi hai (transitions + music)…")
        final = out_dir / f"compilation_{'9x16' if opts.vertical else '16x9'}.mp4"
        compile_video([intro, *clips], final, music=music, music_volume=opts.music_volume,
                      transition=opts.transition)
        result.compilation = str(final)
        step("compilation", "Final video", final_w)
    return result
