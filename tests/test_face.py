import numpy as np

from pipeline.finder import find_segments
from pipeline.players import Player
from pipeline.signals.face import FaceHit, face_hits, hue_distance, kit_hue, kit_matches

PLAYER = Player(id="q", name="Abdul Qadir")


def test_hue_distance_wraps():
    assert hue_distance(5, 175) == 10
    assert hue_distance(51, 105) == 54


def test_kit_hue_green_vs_blue():
    img = np.zeros((300, 200, 3), np.uint8)
    img[:] = (40, 160, 40)  # BGR green
    face = np.array([75, 30, 50, 50] + [0] * 11, np.float32)
    g = kit_hue(img, face)
    img[:] = (180, 90, 20)  # BGR blue
    b = kit_hue(img, face)
    assert kit_matches(g, [51.0]) is True
    assert kit_matches(b, [51.0]) is False
    assert kit_matches(-1, [51.0]) is None


def test_face_hits_rejects_other_team():
    scan = {"reference": {"kit_hues": [51.0]}, "frames": [
        {"t": 1.0, "faces": [[0, 0, 40, 40, 0.9, 0.55, 105.0]]},  # blue kit → reject
        {"t": 2.0, "faces": [[0, 0, 40, 40, 0.9, 0.50, 51.0]]},  # green → ok
        {"t": 3.0, "faces": [[0, 0, 40, 40, 0.9, 0.50, -1.0]]},  # rang nahi pata → ok
        {"t": 4.0, "faces": [[0, 0, 40, 40, 0.9, 0.20, 51.0]]},  # kam similarity
    ]}
    assert [h.t for h in face_hits(scan)] == [2.0, 3.0]


def test_face_marks_whole_short_shot():
    hits = [FaceHit(12.0, 0.65, (0, 0, 50, 50))]  # very strong → 5
    segs, _ = find_segments(PLAYER, [], duration=60, faces=hits,
                            shots=[(0, 10), (10, 18), (18, 60)])
    assert len(segs) == 1
    assert segs[0].start == 10 and segs[0].end == 18
    assert "face_strong" in segs[0].labels


def test_face_ignored_in_crowded_graphic():
    hits = [FaceHit(12.0, 0.8, (0, 0, 50, 50))]
    segs, _ = find_segments(PLAYER, [], duration=60, faces=hits, face_counts={12.0: 11},
                            shots=[(0, 10), (10, 18), (18, 60)])
    assert segs == []


def test_back_name_plus_face():
    scene = {"frames": [{"t": 30.0, "boxes": [["QADIR", 0.9, 0, 0, 10, 10]]}]}
    hits = [FaceHit(31.0, 0.5, (0, 0, 50, 50))]  # face 3 + back_name 4 = 7
    segs, _ = find_segments(PLAYER, [], duration=60, faces=hits, scene_text=scene,
                            shots=[(0, 28), (28, 35), (35, 60)])
    assert len(segs) == 1 and {"face", "back_name"} <= set(segs[0].labels)
