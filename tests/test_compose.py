import pytest

from pipeline.compose import HORIZONTAL, VERTICAL, compile_video, render_clip, title_card
from pipeline.media import probe, run_ffmpeg


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp("compose")
    out = d / "src.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=10",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
    ])
    return out


@pytest.fixture(scope="module")
def music(tmp_path_factory):
    out = tmp_path_factory.mktemp("music") / "m.mp3"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=500:duration=3", str(out)])
    return out


@pytest.mark.parametrize("mode", ["blur", "crop"])
def test_render_vertical_short(src, tmp_path, mode):
    out = render_clip(src, 2, 6, tmp_path / f"{mode}.mp4", mode=mode, focus_x=0.7,
                      title="Abdul Qadir", event="six")
    vi = probe(out)
    assert (vi.width, vi.height) == VERTICAL
    assert vi.fps == 30
    assert abs(vi.duration - 4) < 0.2
    assert vi.has_audio


def test_render_horizontal(src, tmp_path):
    vi = probe(render_clip(src, 0, 3, tmp_path / "h.mp4", size=HORIZONTAL))
    assert (vi.width, vi.height) == HORIZONTAL


def test_title_card_with_special_chars(tmp_path):
    out = title_card(tmp_path / "t.mp4", "Abdul Qadir: 80 (54)", "Asia Cup 'Final'")
    vi = probe(out)
    assert (vi.width, vi.height) == VERTICAL and abs(vi.duration - 2.5) < 0.2


def test_compile_with_music_and_transitions(src, music, tmp_path):
    c1 = render_clip(src, 0, 3, tmp_path / "1.mp4")
    c2 = render_clip(src, 4, 8, tmp_path / "2.mp4", event="four")
    intro = title_card(tmp_path / "i.mp4", "Highlights", duration=2.0)
    out = compile_video([intro, c1, c2], tmp_path / "final.mp4", music=music,
                        transition_dur=0.5)
    vi = probe(out)
    # 2 + 3 + 4 - 2 transitions * 0.5 = 8s (music 3s ka hai lekin loop hota hai)
    assert abs(vi.duration - 8.0) < 0.3
    assert (vi.width, vi.height) == VERTICAL
    assert vi.has_audio
