"""Phase 4 — shorts aur final compilation (sab ffmpeg filters se; MoviePy se kai guna tez).

render_clip   : source video ka ek segment → normalized clip (30fps, 48k stereo), 9:16 ya 16:9,
                text overlays ke saath. Yahi "short" hai.
compile_video : clips + intro card → crossfade transitions → background music (commentary ke
                waqt khud dheema = ducking) → fade out.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline.media import probe, run_ffmpeg

FPS = 30
SR = 48000

VERTICAL = (1080, 1920)
HORIZONTAL = (1920, 1080)

EVENT_TEXT = {
    "six": "SIX!", "four": "FOUR!", "wicket": "WICKET!", "fifty": "FIFTY!",
    "hundred": "HUNDRED!", "dropped": "DROPPED!",
}


def find_font() -> str | None:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for name in ("segoeuib.ttf", "arialbd.ttf", "bahnschrift.ttf", "calibrib.ttf"):
        p = Path(windir) / "Fonts" / name
        if p.exists():
            return str(p)
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        if Path(p).exists():
            return p
    return None


def _esc(path: str | Path) -> str:
    """ffmpeg filter argument ke andar path: backslash → slash, ':' aur "'" escape."""
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


@dataclass
class TextStyle:
    font: str | None = None
    size: int = 64
    color: str = "white"
    box_color: str = "black@0.55"


class _TextFiles:
    """drawtext ke liye text files (textfile= se escaping ke masail khatam)."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="ve_text_"))
        self.n = 0

    def add(self, text: str) -> Path:
        self.n += 1
        p = self.dir / f"t{self.n}.txt"
        p.write_text(text, encoding="utf-8")
        return p

    def cleanup(self) -> None:
        for p in self.dir.glob("*"):
            p.unlink(missing_ok=True)
        self.dir.rmdir()


def _drawtext(tf: _TextFiles, text: str, style: TextStyle, *, y: str,
              enable: str | None = None, size: int | None = None) -> str:
    parts = [
        f"textfile='{_esc(tf.add(text))}'",
        f"fontsize={size or style.size}",
        f"fontcolor={style.color}",
        "x=(w-text_w)/2",
        f"y={y}",
        "box=1", f"boxcolor={style.box_color}", "boxborderw=24",
    ]
    if style.font:
        parts.insert(0, f"fontfile='{_esc(style.font)}'")
    if enable:
        parts.append(f"enable='{enable}'")
    return "drawtext=" + ":".join(parts)


def _video_layout(size: tuple[int, int], mode: str, focus_x: float | None) -> str:
    """[0:v] se [base] banane wala filter. mode: blur (fit + blurred bg) / crop / fit."""
    W, H = size
    norm = f"fps={FPS},setsar=1"
    if W >= H or mode == "fit":  # horizontal output: seedha scale + pad
        return (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,{norm}[base]")
    if mode == "crop":
        fx = 0.5 if focus_x is None else min(max(focus_x, 0.0), 1.0)
        # crop width = ih*9/16; x ko frame ke andar rakho
        cw = "ih*9/16"
        x = f"max(0\\,min(iw-{cw}\\,iw*{fx:.3f}-{cw}/2))"
        return f"[0:v]crop={cw}:ih:{x}:0,scale={W}:{H},{norm}[base]"
    # blur: background = zoom + blur, foreground = poora frame width mein fit
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
        f"boxblur=luma_radius=40:luma_power=2,eq=brightness=-0.08[bgb];"
        f"[fg]scale={W}:-2[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,{norm}[base]"
    )


def render_clip(
    src: str | Path,
    start: float,
    end: float,
    out: str | Path,
    *,
    size: tuple[int, int] = VERTICAL,
    mode: str = "blur",
    focus_x: float | None = None,
    title: str | None = None,
    event: str | None = None,
    style: TextStyle | None = None,
    fade: float = 0.25,
    crf: int = 20,
) -> Path:
    """Ek segment ko formatted clip mein render karo (cut + layout + text + fade, ek hi pass)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    style = style or TextStyle(font=find_font())
    W, H = size
    dur = end - start
    tf = _TextFiles()
    try:
        chain = [_video_layout(size, mode, focus_x)]
        overlays = []
        if title:
            overlays.append(_drawtext(tf, title, style, y=f"{int(H * 0.08)}",
                                      size=int(style.size * (1.0 if W < H else 0.8))))
        if event and event in EVENT_TEXT:
            # Event text aakhri 60% hisse mein (jab shot lag chuka ho)
            overlays.append(_drawtext(tf, EVENT_TEXT[event], style, y=f"{int(H * 0.80)}",
                                      enable=f"gte(t,{dur * 0.4:.2f})",
                                      size=int(style.size * 1.5)))
        fades = [f"fade=t=in:st=0:d={fade}", f"fade=t=out:st={max(dur - fade, 0):.3f}:d={fade}"]
        chain.append(f"[base]{','.join([*overlays, *fades, 'format=yuv420p'])}[v]")

        has_audio = probe(src).has_audio
        args = ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
        if not has_audio:
            args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", f"anullsrc=r={SR}:cl=stereo"]
        a_in = "0:a" if has_audio else "1:a"
        chain.append(
            f"[{a_in}]aresample={SR},aformat=channel_layouts=stereo,"
            f"afade=t=in:st=0:d={fade},afade=t=out:st={max(dur - fade, 0):.3f}:d={fade}[a]"
        )
        run_ffmpeg([
            *args, "-filter_complex", ";".join(chain), "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out),
        ])
    finally:
        tf.cleanup()
    return out


def title_card(
    out: str | Path, title: str, subtitle: str | None = None, *,
    size: tuple[int, int] = VERTICAL, duration: float = 2.5,
    bg: str = "0x0b3d20", style: TextStyle | None = None,
) -> Path:
    """Intro / outro card: rangeen background + text, khamosh audio."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    style = style or TextStyle(font=find_font(), box_color="black@0.0")
    W, H = size
    tf = _TextFiles()
    try:
        filters = [_drawtext(tf, title, style, y="(h-text_h)/2-60", size=int(style.size * 1.4))]
        if subtitle:
            filters.append(_drawtext(tf, subtitle, style, y="(h-text_h)/2+80",
                                     size=int(style.size * 0.8)))
        filters += ["fade=t=in:st=0:d=0.4", f"fade=t=out:st={duration - 0.4}:d=0.4",
                    "format=yuv420p"]
        run_ffmpeg([
            "-f", "lavfi", "-i", f"color=c={bg}:s={W}x{H}:r={FPS}:d={duration}",
            "-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo",
            "-filter_complex", f"[0:v]{','.join(filters)}[v]",
            "-map", "[v]", "-map", "1:a", "-t", f"{duration}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", str(out),
        ])
    finally:
        tf.cleanup()
    return out


def compile_video(
    clips: list[Path],
    out: str | Path,
    *,
    music: Path | None = None,
    music_volume: float = 0.35,
    duck: bool = True,
    transition: str = "fade",
    transition_dur: float = 0.5,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Clips (same size / fps — render_clip / title_card se bane hue) ko jor do.

    - Video: xfade transitions (fade, wipeleft, slideup, circleopen, ...)
    - Audio: acrossfade; music loop + commentary par sidechain ducking + aakhir mein fade out
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not clips:
        raise ValueError("Koi clip nahi")
    durs = [probe(c).duration for c in clips]
    T = min(transition_dur, *(d / 3 for d in durs))

    args: list[str] = []
    for c in clips:
        args += ["-i", str(c)]
    n = len(clips)
    chain: list[str] = []
    for i in range(n):
        chain.append(f"[{i}:v]settb=AVTB,fps={FPS},format=yuv420p[v{i}]")
        chain.append(f"[{i}:a]aresample={SR},aformat=channel_layouts=stereo[a{i}]")

    v_prev, a_prev = "v0", "a0"
    offset = 0.0
    for i in range(1, n):
        offset += durs[i - 1] - T
        chain.append(f"[{v_prev}][v{i}]xfade=transition={transition}:duration={T:.3f}:"
                     f"offset={offset:.3f}[vx{i}]")
        chain.append(f"[{a_prev}][a{i}]acrossfade=d={T:.3f}[ax{i}]")
        v_prev, a_prev = f"vx{i}", f"ax{i}"
    total = sum(durs) - T * (n - 1)

    if music:
        args += ["-stream_loop", "-1", "-i", str(music)]
        mi = n
        chain.append(f"[{mi}:a]aresample={SR},aformat=channel_layouts=stereo,"
                     f"volume={music_volume},atrim=0:{total:.3f},"
                     f"afade=t=out:st={max(total - 2.0, 0):.3f}:d=2[mus]")
        if duck:
            # Commentary ki awaz music ko dabati hai (sidechain) — awaz ruke to music wapas
            chain.append(f"[{a_prev}]asplit=2[voice][key]")
            chain.append("[mus][key]sidechaincompress=threshold=0.03:ratio=8:attack=30:"
                         "release=600[mduck]")
            chain.append("[voice][mduck]amix=inputs=2:duration=first:normalize=0[aout]")
        else:
            chain.append(f"[{a_prev}][mus]amix=inputs=2:duration=first:normalize=0[aout]")
        a_prev = "aout"

    if on_progress:
        on_progress(f"Rendering {n} clips, {total:.0f}s")
    run_ffmpeg([
        *args, "-filter_complex", ";".join(chain),
        "-map", f"[{v_prev}]", "-map", f"[{a_prev}]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ])
    return out
