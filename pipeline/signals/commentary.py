"""Commentary → text (faster-whisper) aur player ke naam ke mentions.

- Transcript video ke cache mein save hota hai (dobara run par foran mil jata hai).
- Whisper ko player ke naam `hotwords` mein dete hain taake spelling sahi aaye.
- Urdu / Hindi / English — language khud detect hoti hai.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings
from pipeline.cache import cache_dir, load_json, save_json
from pipeline.media import FFmpegError
from pipeline.players import window_score

CACHE_VERSION = 1


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Mention:
    time: float
    text: str  # match hone wale words + thoda context
    score: float


def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """ffmpeg se mono 16 kHz float32 audio. (faster-whisper ka apna decoder PyAV use karta hai,
    jo PyAV 19 ke saath toot gaya — ffmpeg hamesha kaam karta hai.)"""
    proc = subprocess.run(
        [settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"Audio extraction failed: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, np.float32)


def _device() -> tuple[str, str]:
    if settings.device in ("cuda", "auto"):
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:  # noqa: BLE001
            pass
    return "cpu", "int8"


def transcribe(
    path: str | Path,
    *,
    model_size: str = "small",
    hotwords: list[str] | None = None,
    language: str | None = None,
    beam_size: int = 5,
    use_cache: bool = True,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    """Returns {"language", "duration", "model", "elapsed", "words": [[start, end, text], ...]}."""
    # beam 5 = zyada sahi naam (default); beam 1 = ~1.5x tez lekin kuch naam chhoot jaate hain
    suffix = "" if beam_size == 5 else f"_b{beam_size}"
    cache_file = cache_dir(path) / f"transcript_{model_size}{suffix}.json"
    if use_cache and isinstance(cached := load_json(cache_file), dict):
        if cached.get("version") == CACHE_VERSION:
            return cached

    from faster_whisper import WhisperModel

    device, compute_type = _device()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        load_audio(path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,  # khamoshi / sirf crowd noise skip
        hotwords=" ".join(hotwords) if hotwords else None,
        condition_on_previous_text=False,  # lambi audio mein repetition loops se bachao
    )
    words: list[list] = []
    for seg in segments:
        for w in seg.words or []:
            words.append([round(w.start, 2), round(w.end, 2), w.word.strip()])
        if on_progress and info.duration:
            on_progress(min(seg.end / info.duration, 1.0))

    result = {
        "version": CACHE_VERSION,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": info.duration,
        "model": model_size,
        "device": device,
        "elapsed": round(time.perf_counter() - t0, 1),
        "words": words,
    }
    save_json(cache_file, result)
    return result


def find_mentions(
    words: list[Word], names: list[str], *, threshold: float = 85, max_words: int = 3
) -> list[Mention]:
    """Transcript mein naam dhoondo (1–3 words ke windows). Qareebi duplicate hata dete hain."""
    mentions: list[Mention] = []
    for i in range(len(words)):
        best, best_k = 0.0, 0
        for k in range(1, max_words + 1):
            if i + k > len(words):
                break
            # Poora window naam se match ho (sirf andar kahin nahi) — taake time naam ke
            # pehle word ka aaye, "shot by Qadir" ka nahi
            s = window_score(" ".join(w.text for w in words[i : i + k]), names)
            if s > best:
                best, best_k = s, k
        if best >= threshold:
            if mentions and words[i].start - mentions[-1].time < 2.0:
                continue
            ctx = " ".join(w.text for w in words[max(0, i - 4) : i + best_k + 4])
            mentions.append(Mention(words[i].start, ctx, best))
    return mentions


def words_from_transcript(tr: dict) -> list[Word]:
    return [Word(*w) for w in tr["words"]]
