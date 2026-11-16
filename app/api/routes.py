"""REST API — players, videos, jobs (analyze / render), music, live progress (SSE)."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import select

from app import players_store
from app.config import settings
from app.db import Job, JobKind, JobStatus, Video, session
from pipeline.media import FFmpegError, probe
from pipeline.service import AnalyzeOptions, RenderOptions, list_music

router = APIRouter(prefix="/api")

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
TERMINAL = {JobStatus.done, JobStatus.failed, JobStatus.cancelled}


# ------------------------------------------------------------------ helpers


def _404(what: str) -> HTTPException:
    return HTTPException(404, f"{what} nahi mila")


def _media_url(rel: str | None) -> str | None:
    """storage/output/... → /media/output/..."""
    return f"/media/{rel}" if rel else None


def player_out(p, d: Path) -> dict:
    return {
        "id": p.id, "name": p.name, "jersey_number": p.jersey_number, "aliases": p.aliases,
        "images": [f"/api/players/{p.id}/images/{i}" for i in range(len(p.images))],
        "editable": settings.players_dir in d.parents,
    }


def video_out(v: Video) -> dict:
    return {**v.model_dump(mode="json", exclude={"path"}),
            "stream_url": f"/api/videos/{v.id}/stream"}


def job_out(j: Job) -> dict:
    d = j.model_dump(mode="json")
    if j.kind == JobKind.render and j.result:
        d["result"] = {**j.result,
                       "shorts": [_media_url(p) for p in j.result.get("shorts", [])],
                       "compilation": _media_url(j.result.get("compilation"))}
    return d


# ------------------------------------------------------------------ health / music


@router.get("/health")
def health(request: Request) -> dict:
    proc = getattr(request.app.state, "worker_proc", None)
    return {"ok": True, "worker": "embedded" if proc and proc.poll() is None
            else ("stopped" if proc else "external")}


@router.get("/music")
def music() -> list[dict]:
    return [{"name": p.name, "url": f"/api/music/{p.name}", "size_bytes": p.stat().st_size}
            for p in list_music()]


@router.get("/music/{name}")
def music_file(name: str) -> FileResponse:
    p = next((m for m in list_music() if m.name == name), None)
    if p is None:
        raise _404("Music")
    return FileResponse(p)


# ------------------------------------------------------------------ players


@router.get("/players")
def players() -> list[dict]:
    return [player_out(p, d) for p, d in players_store.list_all()]


@router.post("/players", status_code=201)
async def create_player(
    name: Annotated[str, Form(min_length=2, max_length=80)],
    images: Annotated[list[UploadFile], File(description="1–5 tasveerein")],
    jersey_number: Annotated[str | None, Form(max_length=4)] = None,
    aliases: Annotated[str, Form(description="Comma se alag: Qadir, قادر, कादिर")] = "",
) -> dict:
    if not 1 <= len(images) <= 5:
        raise HTTPException(422, "1 se 5 images dein")
    data = [(f.filename or "img.jpg", await f.read()) for f in images]
    alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
    try:
        p = await run_in_threadpool(players_store.create, name.strip(), jersey_number or None,
                                    alias_list, data)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return player_out(p, players_store.find_dir(p.id))


@router.get("/players/{player_id}/images/{idx}")
def player_image(player_id: str, idx: int) -> FileResponse:
    p = players_store.load(player_id)
    if p is None or not 0 <= idx < len(p.images):
        raise _404("Image")
    return FileResponse(p.images[idx])


@router.delete("/players/{player_id}", status_code=204)
def delete_player(player_id: str) -> None:
    if not players_store.delete(player_id):
        raise HTTPException(400, "Player delete nahi ho sakta (sample player ya mojood nahi)")


# ------------------------------------------------------------------ videos


def _register(path: Path, filename: str, uploaded: bool) -> Video:
    try:
        vi = probe(path)
    except (FFmpegError, FileNotFoundError) as e:
        raise HTTPException(422, f"Video parh nahi saka: {e}") from e
    v = Video(filename=filename, path=str(path), uploaded=uploaded, duration=vi.duration,
              width=vi.width, height=vi.height, size_bytes=path.stat().st_size)
    with session() as s:
        s.add(v)
        s.commit()
    return v


@router.get("/videos")
def videos() -> list[dict]:
    with session() as s:
        return [video_out(v) for v in s.exec(select(Video).order_by(Video.created_at.desc()))]


@router.post("/videos", status_code=201)
async def upload_video(file: Annotated[UploadFile, File()]) -> dict:
    name = Path(file.filename or "video.mp4").name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(422, f"Video format support nahi: {name}")
    v_dir = settings.storage_dir / "uploads"
    v_dir.mkdir(parents=True, exist_ok=True)
    dest = v_dir / f"{Video().id}_{name}"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(4 * 1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"Video {settings.max_upload_mb} MB se bari hai")
                out.write(chunk)
        v = await run_in_threadpool(_register, dest, name, True)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return video_out(v)


class LocalVideoIn(BaseModel):
    path: str


def _allowed_local(p: Path) -> bool:
    p = p.resolve()
    return any(p.is_relative_to(r.resolve()) for r in settings.local_video_roots)


@router.get("/videos/local-files")
def local_files() -> list[dict]:
    """Local mode: allowed folders ki videos (bina upload register karne ke liye)."""
    out = []
    for root in settings.local_video_roots:
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.suffix.lower() in VIDEO_EXT:
                    out.append({"path": str(p), "name": p.name, "size_bytes": p.stat().st_size})
    return out


@router.post("/videos/local", status_code=201)
async def register_local(body: LocalVideoIn) -> dict:
    p = Path(body.path)
    if not p.is_file() or p.suffix.lower() not in VIDEO_EXT or not _allowed_local(p):
        raise HTTPException(422, "Yeh file allowed local folders mein nahi ya video nahi")
    with session() as s:
        existing = s.exec(select(Video).where(Video.path == str(p.resolve()))).first()
    if existing:
        return video_out(existing)
    return video_out(await run_in_threadpool(_register, p.resolve(), p.name, False))


@router.get("/videos/{video_id}/stream")
def stream_video(video_id: str) -> FileResponse:
    with session() as s:
        v = s.get(Video, video_id)
    if v is None:
        raise _404("Video")
    return FileResponse(v.path, media_type="video/mp4")  # Range requests support (seek)


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: str) -> None:
    with session() as s:
        v = s.get(Video, video_id)
        if v is None:
            raise _404("Video")
        jobs = s.exec(select(Job).where(Job.video_id == video_id)).all()
        if any(j.status not in TERMINAL for j in jobs):
            raise HTTPException(409, "Is video ki job chal rahi hai")
        for j in jobs:
            shutil.rmtree(settings.storage_dir / "output" / j.id, ignore_errors=True)
            s.delete(j)
        s.delete(v)
        s.commit()
    if v.uploaded:
        Path(v.path).unlink(missing_ok=True)


# ------------------------------------------------------------------ jobs


class AnalyzeIn(BaseModel):
    video_id: str
    player_id: str
    options: dict[str, Any] = Field(default_factory=dict)


def _check_opts(cls, opts: dict) -> dict:
    try:
        return vars(cls(**opts))
    except TypeError as e:
        raise HTTPException(422, f"Ghalat option: {e}") from e


@router.post("/jobs/analyze", status_code=201)
def create_analyze(body: AnalyzeIn) -> dict:
    if players_store.load(body.player_id) is None:
        raise _404("Player")
    with session() as s:
        if s.get(Video, body.video_id) is None:
            raise _404("Video")
        j = Job(kind=JobKind.analyze, video_id=body.video_id, player_id=body.player_id,
                options=_check_opts(AnalyzeOptions, body.options))
        s.add(j)
        s.commit()
    return job_out(j)


class RenderIn(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


@router.post("/jobs/{job_id}/render", status_code=201)
def create_render(job_id: str, body: RenderIn) -> dict:
    with session() as s:
        parent = s.get(Job, job_id)
        if parent is None or parent.kind != JobKind.analyze:
            raise _404("Analyze job")
        if parent.status != JobStatus.done:
            raise HTTPException(409, "Analyze job abhi mukammal nahi")
        if not any(seg.get("include", True) for seg in parent.result["segments"]):
            raise HTTPException(422, "Koi moment select nahi")
        opts = _check_opts(RenderOptions, body.options)
        if opts["mode"] not in {"blur", "crop"}:
            raise HTTPException(422, "mode: blur ya crop")
        if opts["music"] and opts["music"] not in {m.name for m in list_music()}:
            raise HTTPException(422, f"Music '{opts['music']}' nahi mila")
        j = Job(kind=JobKind.render, video_id=parent.video_id, player_id=parent.player_id,
                parent_id=parent.id, options=opts)
        s.add(j)
        s.commit()
    return job_out(j)


@router.get("/jobs")
def jobs(video_id: str | None = None, kind: JobKind | None = None,
         limit: Annotated[int, Query(le=200)] = 50) -> list[dict]:
    q = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if video_id:
        q = q.where(Job.video_id == video_id)
    if kind:
        q = q.where(Job.kind == kind)
    with session() as s:
        return [job_out(j) for j in s.exec(q)]


def _get_job(job_id: str) -> Job:
    with session() as s:
        j = s.get(Job, job_id)
    if j is None:
        raise _404("Job")
    return j


@router.get("/jobs/{job_id}")
def job(job_id: str) -> dict:
    return job_out(_get_job(job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    with session() as s:
        j = s.get(Job, job_id)
        if j is None:
            raise _404("Job")
        if j.status == JobStatus.queued:
            j.status, j.message = JobStatus.cancelled, "Cancel ho gaya"
        elif j.status == JobStatus.running:
            j.status, j.message = JobStatus.cancelling, "Cancel kar rahe hain…"
        s.add(j)
        s.commit()
    return job_out(j)


class SegmentPatch(BaseModel):
    id: int
    include: bool | None = None
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)


@router.patch("/jobs/{job_id}/segments")
def patch_segments(job_id: str, patches: list[SegmentPatch]) -> dict:
    """Review screen: moments include / exclude, start/end trim."""
    with session() as s:
        j = s.get(Job, job_id)
        if j is None or j.kind != JobKind.analyze:
            raise _404("Analyze job")
        if j.status != JobStatus.done:
            raise HTTPException(409, "Analyze job abhi mukammal nahi")
        result = dict(j.result)
        segs = {seg["id"]: dict(seg) for seg in result["segments"]}
        for p in patches:
            seg = segs.get(p.id)
            if seg is None:
                raise HTTPException(404, f"Segment {p.id} nahi mila")
            if p.include is not None:
                seg["include"] = p.include
            start = seg["start"] if p.start is None else p.start
            end = seg["end"] if p.end is None else p.end
            if not 0 <= start < end <= result["duration"] + 0.01 or end - start < 1:
                raise HTTPException(422, f"Segment {p.id}: ghalat start/end")
            seg["start"], seg["end"] = round(start, 2), round(end, 2)
        result["segments"] = sorted(segs.values(), key=lambda x: x["start"])
        j.result = result  # naya dict → SQLAlchemy change pakarta hai
        s.add(j)
        s.commit()
    return job_out(j)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events: job badalte hi `data: {...}`; mukammal / fail par stream band."""
    _get_job(job_id)

    async def gen():
        last = None
        while not await request.is_disconnected():
            j = await run_in_threadpool(_get_job, job_id)
            snap = (j.status, j.stage, j.progress, j.message)
            if snap != last:
                last = snap
                yield f"data: {json.dumps(job_out(j))}\n\n"
            if j.status in TERMINAL:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
