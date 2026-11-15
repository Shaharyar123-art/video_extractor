"""Integration tests — ffmpeg se ek chhoti synthetic video banate hain (3 rang = 3 shots)."""

import pytest

from app.config import settings
from pipeline.cutter import cut_clip
from pipeline.frames import Crop, iter_frames
from pipeline.media import probe, run_ffmpeg
from pipeline.scenes import detect_shots


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid")
    out = d / "three_shots.mp4"
    # red 4s → green 4s → blue 4s, 25fps, 320x240, + sine audio
    run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=4:r=25",
        "-f", "lavfi", "-i", "color=c=green:s=320x240:d=4:r=25",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=4:r=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(out),
    ])
    return out


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")


def test_probe(synthetic_video):
    vi = probe(synthetic_video)
    assert (vi.width, vi.height) == (320, 240)
    assert vi.fps == 25
    assert abs(vi.duration - 12) < 0.2
    assert vi.has_audio


def test_detect_shots(synthetic_video):
    shots = detect_shots(synthetic_video)
    assert len(shots) == 3
    assert abs(shots[1][0] - 4) < 0.2
    assert abs(shots[2][0] - 8) < 0.2


def test_iter_frames_fps_and_timestamps(synthetic_video):
    frames = list(iter_frames(synthetic_video, fps=1, width=None))
    assert len(frames) == 12
    assert [round(t) for t, _ in frames[:3]] == [0, 1, 2]
    # second 5 → green shot (BGR)
    b, g, r = frames[5][1][120, 160]
    assert g > 100 and r < 50 and b < 50


def test_iter_frames_crop_and_resize(synthetic_video):
    _, frame = next(iter_frames(synthetic_video, fps=1, width=80, crop=Crop(0, 0.5, 1, 0.5)))
    assert frame.shape == (30, 80, 3)


def test_iter_frames_start_offset(synthetic_video):
    ts, frame = next(iter_frames(synthetic_video, fps=1, width=None, start=9))
    assert ts == 9
    b, _g, r = frame[120, 160]
    assert b > 100 and r < 50  # blue shot


def test_cut_clip(synthetic_video, tmp_path):
    out = cut_clip(synthetic_video, 3.5, 6.5, tmp_path / "c.mp4")
    vi = probe(out)
    assert abs(vi.duration - 3.0) < 0.15
    assert vi.has_audio


def test_find_cuts_non_max_suppression():
    import numpy as np

    from pipeline.scenes import ShotOptions, find_cuts

    t = np.arange(0, 5, 0.1)
    h = np.zeros_like(t)
    p = np.zeros_like(t)
    h[[10, 11, 12]] = [0.3, 0.6, 0.3]  # ek wipe transition, 3 frames mein
    h[40] = 0.5  # doosra cut
    h[25] = 0.1  # kamzor — ignore
    cuts = find_cuts(t, h, p, ShotOptions(min_shot=1.0))
    assert cuts == [t[11], t[40]]
