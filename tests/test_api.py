"""API end-to-end tests — alag storage folder, worker ko seedha (synchronously) chalate hain."""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.db import Job, JobKind, JobStatus, session
from pipeline.media import run_ffmpeg


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "embedded_worker", False)
    monkeypatch.setattr(settings, "music_dir", tmp_path / "music")
    monkeypatch.setattr(settings, "samples_dir", tmp_path / "samples")
    vids = tmp_path / "local"
    vids.mkdir()
    monkeypatch.setattr(settings, "local_video_roots", [vids])
    db.reset_engine()
    from app.main import create_app

    with TestClient(create_app()) as c:
        c.local_dir = vids
        yield c
    db.reset_engine()


@pytest.fixture
def video_file(client):
    p = client.local_dir / "match.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=12",
                "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)])
    return p


def _png() -> bytes:
    _, buf = cv2.imencode(".png", np.full((64, 64, 3), 128, np.uint8))
    return buf.tobytes()


def make_player(client, name="Abdul Qadir"):
    r = client.post("/api/players", data={"name": name, "jersey_number": "17",
                                          "aliases": "Qadir, قادر"},
                    files=[("images", ("a.png", _png(), "image/png"))])
    assert r.status_code == 201, r.text
    return r.json()


def test_health_and_music(client, tmp_path):
    assert client.get("/api/health").json()["worker"] == "external"
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "song.mp3").write_bytes(b"x")
    assert [m["name"] for m in client.get("/api/music").json()] == ["song.mp3"]


def test_player_crud(client):
    p = make_player(client)
    assert p["id"] == "abdul_qadir" and p["editable"]
    assert "قادر" in p["aliases"]
    assert client.get(p["images"][0]).status_code == 200
    assert make_player(client)["id"] == "abdul_qadir_2"  # same naam → naya id
    assert client.delete(f"/api/players/{p['id']}").status_code == 204
    ids = [x["id"] for x in client.get("/api/players").json()]
    assert "abdul_qadir" not in ids


def test_player_bad_image(client):
    r = client.post("/api/players", data={"name": "X Y"},
                    files=[("images", ("a.txt", b"hi", "text/plain"))])
    assert r.status_code == 422


def test_upload_and_local_video(client, video_file):
    with video_file.open("rb") as f:
        r = client.post("/api/videos", files={"file": ("m.mp4", f, "video/mp4")})
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["width"] == 320 and abs(v["duration"] - 12) < 0.3 and "path" not in v

    local = client.get("/api/videos/local-files").json()
    assert [x["name"] for x in local] == ["match.mp4"]
    r1 = client.post("/api/videos/local", json={"path": str(video_file)})
    r2 = client.post("/api/videos/local", json={"path": str(video_file)})
    assert r1.status_code == 201 and r1.json()["id"] == r2.json()["id"]  # dobara = wahi

    rng = client.get(f"/api/videos/{v['id']}/stream", headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100


def test_local_video_outside_roots_rejected(client, tmp_path):
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"x")
    assert client.post("/api/videos/local", json={"path": str(outside)}).status_code == 422


def test_bad_upload_type(client):
    r = client.post("/api/videos", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 422


def _setup_done_analyze(client, video_file):
    from app.worker import process_one

    p = make_player(client)
    v = client.post("/api/videos/local", json={"path": str(video_file)}).json()
    r = client.post("/api/jobs/analyze", json={
        "video_id": v["id"], "player_id": p["id"],
        "options": {"commentary": False, "face": False}})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["status"] == "queued"
    assert process_one() is True  # worker: synthetic video → koi scoreboard nahi
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done", done["error"]
    assert done["result"]["segments"] == []
    # Render test ke liye fake moments daal do
    with session() as s:
        j = s.get(Job, job["id"])
        j.result = {**j.result, "segments": [
            {"id": 1, "start": 1.0, "end": 4.0, "score": 8, "labels": ["striker", "six"],
             "focus_x": None, "include": True},
            {"id": 2, "start": 6.0, "end": 9.0, "score": 5, "labels": ["striker"],
             "focus_x": 0.6, "include": True},
        ]}
        s.add(j)
        s.commit()
    return job["id"]


def test_analyze_edit_render_flow(client, video_file):
    from app.worker import process_one

    jid = _setup_done_analyze(client, video_file)

    r = client.patch(f"/api/jobs/{jid}/segments",
                     json=[{"id": 2, "include": False}, {"id": 1, "start": 0.5}])
    assert r.status_code == 200, r.text
    segs = r.json()["result"]["segments"]
    assert segs[0]["start"] == 0.5 and segs[1]["include"] is False
    bad = client.patch(f"/api/jobs/{jid}/segments", json=[{"id": 1, "start": 5, "end": 2}])
    assert bad.status_code == 422

    r = client.post(f"/api/jobs/{jid}/render", json={"options": {"music": ""}})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    assert process_one() is True
    rj = client.get(f"/api/jobs/{rid}").json()
    assert rj["status"] == "done", rj["error"]
    assert len(rj["result"]["shorts"]) == 1  # sirf included moment
    url = rj["result"]["compilation"]
    assert url.startswith("/media/output/")
    assert client.get(url).status_code == 200
    assert client.get(rj["result"]["shorts"][0], headers={"Range": "bytes=0-9"}).status_code == 206

    # SSE: mukammal job → ek event aur stream band
    with client.stream("GET", f"/api/jobs/{rid}/events") as s:
        events = [json.loads(line[6:]) for line in s.iter_lines() if line.startswith("data: ")]
    assert events[-1]["status"] == "done"


def test_render_validation(client, video_file):
    jid = _setup_done_analyze(client, video_file)
    assert client.post(f"/api/jobs/{jid}/render",
                       json={"options": {"mode": "zoom"}}).status_code == 422
    assert client.post(f"/api/jobs/{jid}/render",
                       json={"options": {"music": "nahi.mp3"}}).status_code == 422
    assert client.post(f"/api/jobs/{jid}/render",
                       json={"options": {"unknown": 1}}).status_code == 422


def test_cancel_queued_job(client, video_file):
    p = make_player(client)
    v = client.post("/api/videos/local", json={"path": str(video_file)}).json()
    j = client.post("/api/jobs/analyze", json={"video_id": v["id"], "player_id": p["id"]}).json()
    assert client.post(f"/api/jobs/{j['id']}/cancel").json()["status"] == "cancelled"
    from app.worker import process_one

    assert process_one() is False  # cancelled job worker nahi uthata


def test_cancel_running_job_stops_worker(client, video_file):
    """Running job ko cancel → worker ka progress callback CancelledError deta hai."""
    from app.worker import _Reporter
    from pipeline.service import CancelledError

    p = make_player(client)
    v = client.post("/api/videos/local", json={"path": str(video_file)}).json()
    j = client.post("/api/jobs/analyze", json={"video_id": v["id"], "player_id": p["id"]}).json()
    with session() as s:
        job = s.get(Job, j["id"])
        job.status = JobStatus.running
        s.add(job)
        s.commit()
    assert client.post(f"/api/jobs/{j['id']}/cancel").json()["status"] == "cancelling"
    with pytest.raises(CancelledError):
        _Reporter(j["id"])("scoreboard", 0.5, "")


def test_job_404s(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/analyze",
                       json={"video_id": "x", "player_id": "nope"}).status_code == 404
    assert client.get("/api/jobs", params={"kind": JobKind.render}).json() == []
