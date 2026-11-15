"""Signals ko jor kar player ke moments (segments) dhoondhna — multi-signal scoring.

Har second (graphics scan ki fps par) ke liye har source ka score:

| Source       | Score | Kab                                                   |
|--------------|-------|-------------------------------------------------------|
| striker      | 5     | Scoreboard par strike par yahi player                 |
| bowler       | 5     | Scoreboard par bowler yahi player                     |
| card         | 5     | Name card / dismissal card ('c QADIR b ...') mein naam |
| commentary   | 3     | Commentator ne naam liya (window: -6s … +2s)          |
| non_striker  | 1     | Crease par hai lekin strike par nahi                  |
| info         | 1     | Center box mein naam ('LAST WICKET QADIR 45')          |

Kul score ≥ threshold (default 5) wale seconds → segments → padding/merge → camera cuts tak
phaila do (taake clip shot ke beech se shuru/khatam na ho).
Phase 3 mein face / tracking / jersey signals isi table mein add honge.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

from pipeline.players import Player, name_score
from pipeline.scenes import Shot
from pipeline.segments import MergeOptions, Segment, merge_segments
from pipeline.signals.commentary import Mention
from pipeline.signals.scoreboard import FrameGraphics

WEIGHTS = {
    "striker": 5.0,
    "bowler": 5.0,
    "card": 5.0,
    "commentary": 3.0,
    "non_striker": 1.0,
    "info": 1.0,
}

NAME_MATCH = 85.0  # name_score threshold (0–100)


@dataclass
class FindOptions:
    threshold: float = 5.0
    commentary_window: tuple[float, float] = (-6.0, 2.0)
    merge: MergeOptions = field(
        default_factory=lambda: MergeOptions(
            pad_before=1.0, pad_after=2.0, merge_gap=4.0, min_length=3.0, max_length=60.0
        )
    )
    max_shot_expand: float = 12.0  # sirf itne chhote shots tak clip phailao
    split_balls: bool = True  # lambe segments ko alag deliveries mein todo (shorts ke liye)
    ball_gap: float = 3.0  # scoreboard itne sec gayab rahe phir wapas aaye = nayi ball


@dataclass
class Timeline:
    """Har time-step par har source ka score + events."""

    step: float
    scores: dict[float, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    events: dict[float, str] = field(default_factory=dict)
    # Jin seconds mein screen par KISI AUR player ka card hai (e.g. 'USMAN KHAN 50*')
    other: set[float] = field(default_factory=set)

    def key(self, t: float) -> float:
        return round(round(t / self.step) * self.step, 2)

    def add(self, t: float, source: str) -> None:
        self.scores[self.key(t)][source] = WEIGHTS[source]

    def total(self, t: float) -> float:
        return sum(self.scores.get(t, {}).values())


def _is_person_card(line: str) -> bool:
    """Card ki pehli line kisi player ka naam lagti hai? ('HAYAM KHAN', 'USMAN KHAN 50*')
    — kam az kam do 3+ harf wale words. 'INDIA', 'ASIAN CRIC' jaise logos ko alag karna mushkil
    hai, is liye sirf do-word + upper-case lines."""
    words = [w for w in line.split() if w.isalpha() and len(w) >= 3]
    return len(words) >= 2 and line.upper() == line


def other_player_runs(other: set[float], step: float, min_len: float = 2.0) -> list[tuple]:
    """Doosre player ke card wale seconds ko lagataar runs mein jor do."""
    runs: list[list[float]] = []
    for t in sorted(other):
        if runs and t - runs[-1][1] <= step * 1.5:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return [(a, b + step) for a, b in runs if b + step - a >= min_len]


def graphics_signals(frames: list[FrameGraphics], player: Player, tl: Timeline) -> None:
    names = player.all_names()

    def match(text: str | None) -> bool:
        return bool(text) and name_score(text, names) >= NAME_MATCH

    for g in frames:
        if g.event:
            tl.events[round(g.t, 2)] = g.event
        if match(g.striker):
            tl.add(g.t, "striker")
        elif any(match(b) for b in g.batsmen):
            tl.add(g.t, "non_striker")
        if match(g.bowler):
            tl.add(g.t, "bowler")
        if any(match(c) for c in g.cards):
            tl.add(g.t, "card")
        elif g.cards and _is_person_card(g.cards[0]):
            tl.other.add(tl.key(g.t))
        if match(g.center_text):
            tl.add(g.t, "info")


def commentary_signals(mentions: list[Mention], tl: Timeline, opts: FindOptions) -> None:
    a, b = opts.commentary_window
    for m in mentions:
        t = m.time + a
        while t <= m.time + b:
            tl.add(t, "commentary")
            t += tl.step


def expand_to_shots(segments: list[Segment], shots: list[Shot], max_len: float) -> list[Segment]:
    """Segment ki start ko us shot ki shuruat tak, aur end ko us shot ke end tak le jao jis mein
    woh pare hain — agar woh shot `max_len` se chhota ho."""
    out = []
    for s in segments:
        start, end = s.start, s.end
        for a, b in shots:
            if a <= start < b and b - a <= max_len:
                start = a
            if a < end <= b and b - a <= max_len:
                end = b
        out.append(Segment(start, end, s.score, list(s.labels)))
    return out


def ball_starts(frames: list[FrameGraphics], min_gap: float) -> list[float]:
    """Nayi delivery kahan shuru hoti hai — do nishaniyan:

    1. Overs badle ('8.2' → '8.4'): highlights editor nayi ball par cut karta hai aur
       scoreboard pehle se updated hota hai, is liye change camera cut ke saath aata hai.
       OCR ki ek-frame ghalti se bachne ke liye naya value 2 frames tak tikna chahiye.
    2. Scoreboard ≥ `min_gap` sec chhupa raha (replay / reaction) phir wapas aaya.
    """
    starts: set[float] = set()

    hidden_since: float | None = None
    for g in frames:
        if g.has_scoreboard:
            if hidden_since is not None and g.t - hidden_since >= min_gap:
                starts.add(g.t)
            hidden_since = None
        elif hidden_since is None:
            hidden_since = g.t

    seen = [g for g in frames if g.overs]
    current: str | None = None
    for a, b in pairwise(seen):
        if a.overs == b.overs and a.overs != current and b.t - a.t <= 2.5:
            if current is not None:
                starts.add(a.t)
            current = a.overs
    return sorted(starts)


def split_segments(
    segments: list[Segment], cuts: list[float], shots: list[Shot] | None, min_length: float
) -> list[Segment]:
    """Segments ko `cuts` par todo. Har cut ko qareebi camera cut par le jao (pehle 2 sec ke
    andar — 1 fps sampling ki wajah se asal cut thoda pehle hota hai). Chhote tukre pichhle
    se jor do."""
    bounds = sorted({a for a, _ in shots}) if shots else []

    def snap(t: float) -> float:
        near = [b for b in bounds if t - 2.0 <= b <= t + 0.5]
        return near[-1] if near else t

    out: list[Segment] = []
    for s in segments:
        points = [snap(c) for c in cuts if s.start + min_length <= c <= s.end - min_length]
        edges = [s.start, *sorted(set(points)), s.end]
        pieces = [Segment(a, b, s.score, list(s.labels)) for a, b in pairwise(edges)]
        merged: list[Segment] = []
        for p in pieces:
            if merged and p.duration < min_length:
                merged[-1].end = p.end
            else:
                merged.append(p)
        out.extend(merged)
    return out


def find_segments(
    player: Player,
    frames: list[FrameGraphics],
    *,
    duration: float,
    shots: list[Shot] | None = None,
    mentions: list[Mention] | None = None,
    step: float = 1.0,
    opts: FindOptions | None = None,
) -> tuple[list[Segment], Timeline]:
    opts = opts or FindOptions()
    tl = Timeline(step=step)
    graphics_signals(frames, player, tl)
    if mentions:
        commentary_signals(mentions, tl, opts)

    raw: list[Segment] = []
    for t in sorted(tl.scores):
        total = tl.total(t)
        if total >= opts.threshold:
            raw.append(Segment(t - step / 2, t + step / 2, total, sorted(tl.scores[t])))

    merged = merge_segments(raw, duration, opts.merge)
    if shots:
        merged = expand_to_shots(merged, shots, opts.max_shot_expand)
        # phailane ke baad overlap ho sakta hai → dobara jor do (padding ke baghair)
        merged = merge_segments(
            merged, duration,
            MergeOptions(pad_before=0, pad_after=0, merge_gap=0.5,
                         min_length=opts.merge.min_length, max_length=1e9),
        )

    # Doosre player ke card (e.g. pichhle batsman ka out card) ke runs ki shuru/end par bhi todo
    other_runs = other_player_runs(tl.other, step)
    cuts = [t for run in other_runs for t in run]
    if opts.split_balls:
        cuts += ball_starts(frames, opts.ball_gap)
    if cuts:
        merged = split_segments(merged, sorted(cuts), shots, opts.merge.min_length)

    def other_fraction(seg: Segment) -> float:
        overlap = sum(max(0.0, min(b, seg.end) - max(a, seg.start)) for a, b in other_runs)
        return overlap / seg.duration if seg.duration else 0.0

    merged = [s for s in merged if other_fraction(s) < 0.5]

    # Har tukre ke apne signals + events (four/six/wicket) labels mein
    for seg in merged:
        srcs = {s for t, d in tl.scores.items() if seg.start <= t <= seg.end
                and sum(d.values()) >= opts.threshold for s in d}
        seg.labels = sorted(srcs) or seg.labels
        seg.score = max((tl.total(t) for t in tl.scores if seg.start <= t <= seg.end),
                        default=seg.score)
        for t, e in sorted(tl.events.items()):
            if seg.start <= t <= seg.end and e not in seg.labels:
                seg.labels.append(e)
    # Todne ke baad kuch tukron mein player asal mein nahi hota (sirf padding / shot tak
    # phailaya hua hissa). Tukre ke kam az kam 2 sec ya 25% mein player hona chahiye.
    def strong_seconds(seg: Segment) -> int:
        return sum(1 for t in tl.scores
                   if seg.start <= t < seg.end and tl.total(t) >= opts.threshold)

    merged = [s for s in merged
              if strong_seconds(s) * step >= max(2.0, 0.25 * s.duration)]
    return merged, tl


# ---------------------------------------------------------------- evaluation


@dataclass
class EvalResult:
    precision: float  # predicted seconds mein se kitne sahi
    recall: float  # sahi seconds mein se kitne pakre
    hits: int  # answer-key segments jo ≥50% pakre gaye
    total: int

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def _seconds(segs: list[Segment]) -> set[int]:
    out: set[int] = set()
    for s in segs:
        out.update(range(int(s.start), int(s.end + 0.999)))
    return out


def evaluate(predicted: list[Segment], truth: list[Segment]) -> EvalResult:
    p, t = _seconds(predicted), _seconds(truth)
    overlap = p & t
    hits = sum(1 for s in truth if len(_seconds([s]) & p) >= 0.5 * max(len(_seconds([s])), 1))
    return EvalResult(
        precision=len(overlap) / len(p) if p else 0.0,
        recall=len(overlap) / len(t) if t else 0.0,
        hits=hits,
        total=len(truth),
    )
