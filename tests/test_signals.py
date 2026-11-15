from pipeline.finder import FindOptions, evaluate, expand_to_shots, find_segments
from pipeline.players import Player
from pipeline.segments import Segment
from pipeline.signals.commentary import Mention, Word, find_mentions
from pipeline.signals.scoreboard import FrameGraphics, _event_from, parse_frame

W, H = 1280, 186


def box(text, x0, y0, x1, y1):
    return [text, 1.0, x0, y0, x1, y1]


def bar_frame(t, top, bottom, bowler, center="", icon=(0.1, 0.0)):
    boxes = [
        box("PAK", 157, 115, 208, 137),
        box(top, 327, 114, 402, 139),
        box(bottom, 329, 143, 404, 163),
        box(center, 660, 115, 800, 160),
        box(bowler, 846, 115, 921, 135),
        box("THIS OVER", 847, 146, 931, 163),
        box("3-56 8.5", 1056, 115, 1137, 137),
    ]
    return {"t": t, "boxes": boxes, "icon": [0, icon[0], icon[1], 0, 0, 0, 0]}


def test_parse_bar_striker_top():
    g = parse_frame(bar_frame(10, "NIQAB", "SEYAM", "KAVYA"), W, H)
    assert g.batsmen == ["NIQAB", "SEYAM"]
    assert g.striker == "NIQAB"
    assert g.bowler == "KAVYA"
    assert g.cards == []


def test_parse_bar_striker_bottom_and_event():
    g = parse_frame(bar_frame(10, "LAKSHYA", "AARYAVIR", "RAZA", "CFOUR", icon=(0, 0.12)), W, H)
    assert g.striker == "AARYAVIR"
    assert g.event == "four"


def test_parse_card_frame():
    rec = {"t": 5, "boxes": [box("HAMZA ZAHOOR", 421, 35, 708, 69),
                             box("c MOHIT b KAVYA", 429, 80, 611, 100),
                             box("FOURS: 0", 430, 109, 517, 127)]}
    g = parse_frame(rec, W, H)
    assert not g.has_scoreboard
    assert "HAMZA ZAHOOR" in g.cards


def test_event_parsing():
    assert _event_from("SIX") == "six"
    assert _event_from("LAST WICKET ZAHOOR 10 (10)") is None
    assert _event_from("NEED 86 OFF 52 BALLS") is None
    assert _event_from("ABOUT") is None
    assert _event_from("OUT") == "wicket"


PLAYER = Player(id="q", name="Abdul Qadir", jersey_number="17", aliases=["قادر"])


def g(t, **kw):
    return FrameGraphics(t=t, **kw)


def test_find_segments_striker_and_bowler():
    frames = [g(t, batsmen=["QADIR", "X"], striker="QADIR") for t in range(10, 16)]
    frames += [g(t, batsmen=["QADIR", "X"], striker="X") for t in range(16, 40)]  # non-striker
    frames += [g(t, bowler="QADIR", event="six" if t == 102 else None) for t in range(100, 106)]
    segs, _ = find_segments(PLAYER, frames, duration=200)
    assert len(segs) == 2
    assert segs[0].start <= 10 and segs[0].end < 25
    assert "striker" in segs[0].labels
    assert "bowler" in segs[1].labels and "six" in segs[1].labels


def test_commentary_alone_below_threshold_but_adds_up():
    frames = [g(t, batsmen=["QADIR", "X"], striker="X") for t in range(50, 60)]  # +1
    mentions = [Mention(55, "Qadir runs", 100)]  # +3 → 4 < 5
    segs, _ = find_segments(PLAYER, frames, duration=200, mentions=mentions)
    assert segs == []
    segs, _ = find_segments(PLAYER, frames, duration=200, mentions=mentions,
                            opts=FindOptions(threshold=4))
    assert len(segs) == 1


def test_expand_to_shots():
    shots = [(0, 10), (10, 14), (14, 60)]
    out = expand_to_shots([Segment(11, 13)], shots, max_len=12)
    assert (out[0].start, out[0].end) == (10, 14)
    out = expand_to_shots([Segment(20, 25)], shots, max_len=12)  # lamba shot → nahi phailao
    assert (out[0].start, out[0].end) == (20, 25)


def test_find_mentions_multiscript():
    words = [Word(1, 1.3, "what"), Word(1.3, 1.6, "a"), Word(1.6, 2.0, "shot"),
             Word(2.0, 2.4, "by"), Word(2.4, 2.9, "Qadir"), Word(10, 10.5, "قادر"),
             Word(10.5, 11, "نے")]
    ms = find_mentions(words, PLAYER.all_names())
    assert [round(m.time, 1) for m in ms] == [2.4, 10.0]


def test_evaluate():
    truth = [Segment(10, 20), Segment(50, 60)]
    pred = [Segment(12, 22)]
    r = evaluate(pred, truth)
    assert r.hits == 1 and r.total == 2
    assert round(r.precision, 2) == 0.8
    assert round(r.recall, 2) == 0.4


def test_bar_frame_with_logo_text_above_is_not_card():
    rec = bar_frame(10, "QADIR", "USMAN", "ARYAN")
    rec["boxes"].append(box("GATORA", 100, 20, 200, 40))  # video background mein logo
    rec["icon"].append(0)
    g = parse_frame(rec, W, H)
    assert g.cards == []
    assert g.striker == "QADIR"


def test_ball_splitting():
    from pipeline.finder import ball_starts, split_segments

    frames = []
    # ball 1: scoreboard 0-9, hidden 10-17 (replay); ball 2: scoreboard 18-27, hidden 28-35
    for t in range(36):
        sb = t < 10 or 18 <= t < 28
        frames.append(g(t, batsmen=["QADIR", "X"] if sb else [], striker="QADIR" if sb else None))
    assert ball_starts(frames, 3.0) == [18]
    shots = [(0, 10), (10, 17), (17, 28), (28, 36)]
    parts = split_segments([Segment(0, 36)], [18], shots, min_length=3)
    assert [(p.start, p.end) for p in parts] == [(0, 17), (17, 36)]


def test_ball_starts_from_overs_change():
    from pipeline.finder import ball_starts

    ov = {**dict.fromkeys(range(10), "8.2"), 10: "8.3",  # 10 = OCR glitch (1 frame)
          **dict.fromkeys(range(11, 14), "8.2"), **dict.fromkeys(range(14, 25), "8.4")}
    frames = [g(t, batsmen=["A", "B"], overs=ov[t]) for t in range(25)]
    assert ball_starts(frames, 3.0) == [14]


def test_parse_team_score_and_overs():
    rec = bar_frame(1, "A", "B", "C")
    rec["boxes"] += [box("319-7", 138, 133, 232, 165), box("47.5", 236, 140, 283, 162)]
    rec["icon"] += [0, 0]
    gr = parse_frame(rec, W, H)
    assert (gr.team_score, gr.overs) == ("319-7", "47.5")


def test_other_player_card_removed():
    # 0-7: doosre player (HAYAM KHAN) ka out card; 8-20: Qadir striker
    frames = [g(t, cards=["HAYAM KHAN", "21", "c ANVAY b KAVYA"]) for t in range(8)]
    frames += [g(t, batsmen=["QADIR", "X"], striker="QADIR", overs="9.1") for t in range(8, 21)]
    mentions = [Mention(6, "Qadir comes in", 100)]
    segs, _ = find_segments(PLAYER, frames, duration=60, mentions=mentions,
                            shots=[(0, 8), (8, 21), (21, 60)])
    assert len(segs) == 1
    assert segs[0].start >= 7.5
