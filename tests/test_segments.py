import pytest

from pipeline.segments import MergeOptions, Segment, merge_segments, parse_time, snap_to_shots


@pytest.mark.parametrize(
    ("value", "expected"),
    [("00:12", 12.0), ("1:05", 65.0), ("1:02:03", 3723.0), ("7.5", 7.5), (42, 42.0)],
)
def test_parse_time(value, expected):
    assert parse_time(value) == expected


def test_parse_time_invalid():
    with pytest.raises(ValueError):
        parse_time("1:2:3:4")


NO_PAD = MergeOptions(pad_before=0, pad_after=0, merge_gap=1.0, min_length=0, max_length=1000)


def test_merge_overlapping_and_close():
    segs = [Segment(10, 15), Segment(14, 20), Segment(20.5, 25), Segment(40, 45)]
    out = merge_segments(segs, duration=100, opts=NO_PAD)
    assert [(s.start, s.end) for s in out] == [(10, 25), (40, 45)]


def test_padding_clamped_to_video():
    out = merge_segments([Segment(1, 5), Segment(95, 99)], duration=100,
                         opts=MergeOptions(pad_before=2, pad_after=2, min_length=0))
    assert [(s.start, s.end) for s in out] == [(0, 7), (93, 100)]


def test_min_length_drops_short():
    opts = MergeOptions(pad_before=0, pad_after=0, min_length=3)
    out = merge_segments([Segment(10, 11), Segment(20, 25)], duration=100, opts=opts)
    assert [(s.start, s.end) for s in out] == [(20, 25)]


def test_max_length_splits_evenly():
    opts = MergeOptions(pad_before=0, pad_after=0, min_length=0, max_length=10)
    out = merge_segments([Segment(0, 25)], duration=100, opts=opts)
    assert len(out) == 3
    assert out[0].start == 0 and out[-1].end == 25
    assert all(s.duration <= 10 for s in out)


def test_labels_merged_without_duplicates():
    segs = [Segment(10, 15, 3, ["four"]), Segment(15, 20, 5, ["four", "six"])]
    out = merge_segments(segs, duration=100, opts=NO_PAD)
    assert out[0].labels == ["four", "six"]
    assert out[0].score == 5


def test_snap_to_shots():
    shots = [(0, 9.6), (9.6, 18.2), (18.2, 30)]
    out = snap_to_shots([Segment(10, 18)], shots, tolerance=1.0)
    assert (out[0].start, out[0].end) == (9.6, 18.2)


def test_snap_ignores_far_boundaries():
    shots = [(0, 5), (5, 30)]
    out = snap_to_shots([Segment(10, 20)], shots, tolerance=1.0)
    assert (out[0].start, out[0].end) == (10, 20)


def test_segment_from_answer_key_dict():
    s = Segment.from_dict({"start": "01:05", "end": "01:14", "event": "six"})
    assert (s.start, s.end, s.labels) == (65, 74, ["six"])
