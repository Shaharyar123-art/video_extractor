"""Time segments: parse, merge, padding, shot boundaries par snap."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def parse_time(value: str | float) -> float:
    """'mm:ss', 'hh:mm:ss', 'ss.s' ya number → seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid time: {value!r}")
    secs = 0.0
    for part in parts:
        secs = secs * 60 + float(part)
    return secs


def format_time(seconds: float) -> str:
    m, s = divmod(max(seconds, 0.0), 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


@dataclass
class Segment:
    start: float
    end: float
    score: float = 0.0
    labels: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(
            start=parse_time(d["start"]),
            end=parse_time(d["end"]),
            score=float(d.get("score", 0.0)),
            labels=list(d.get("labels", [d["event"]] if "event" in d else [])),
        )


@dataclass
class MergeOptions:
    pad_before: float = 2.0
    pad_after: float = 2.0
    merge_gap: float = 1.5  # itne seconds se kam fasla ho to do segments jor do
    min_length: float = 3.0
    max_length: float = 60.0


def merge_segments(
    segments: list[Segment], duration: float, opts: MergeOptions | None = None
) -> list[Segment]:
    """Padding lagao, overlap/qareebi segments jor do, chhote nikal do, lambe tod do."""
    opts = opts or MergeOptions()
    padded = sorted(
        (
            Segment(
                max(0.0, s.start - opts.pad_before),
                min(duration, s.end + opts.pad_after),
                s.score,
                list(s.labels),
            )
            for s in segments
            if s.end > s.start
        ),
        key=lambda s: s.start,
    )

    merged: list[Segment] = []
    for s in padded:
        if merged and s.start - merged[-1].end <= opts.merge_gap:
            last = merged[-1]
            last.end = max(last.end, s.end)
            last.score = max(last.score, s.score)
            last.labels.extend(lbl for lbl in s.labels if lbl not in last.labels)
        else:
            merged.append(s)

    out: list[Segment] = []
    for s in merged:
        if s.duration < opts.min_length:
            continue
        # Bohat lamba segment barabar hisson mein tod do
        n = max(1, int(-(-s.duration // opts.max_length)))
        step = s.duration / n
        for i in range(n):
            out.append(Segment(s.start + i * step, s.start + (i + 1) * step, s.score, list(s.labels)))
    return out


def snap_to_shots(
    segments: list[Segment], shots: list[tuple[float, float]], tolerance: float = 1.0
) -> list[Segment]:
    """Segment ki start/end ko qareebi shot boundary par le jao (agar `tolerance` ke andar ho),
    taake clip beech-shot mein na kate."""
    bounds = sorted({b for shot in shots for b in shot})
    if not bounds:
        return segments

    def nearest(t: float) -> float:
        best = min(bounds, key=lambda b: abs(b - t))
        return best if abs(best - t) <= tolerance else t

    out = []
    for s in segments:
        start, end = nearest(s.start), nearest(s.end)
        if end - start <= 0.5:
            start, end = s.start, s.end
        out.append(Segment(start, end, s.score, list(s.labels)))
    return out
