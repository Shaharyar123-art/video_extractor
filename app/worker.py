"""Background worker — database se queued jobs utha kar pipeline chalata hai.

Alag process mein chalta hai (API tez rahe):  uv run python -m app.worker
(API `embedded_worker=True` par isay khud start kar deti hai.)
Ek waqt mein ek job — CPU par bhari kaam hai, parallel karne se sab slow ho jata hai.
"""

from __future__ import annotations

import logging
import signal
import time
import traceback
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import update
from sqlmodel import select

from app import players_store
from app.config import settings
from app.db import Job, JobKind, JobStatus, Video, session
from pipeline.service import (
    AnalyzeOptions,
    CancelledError,
    RenderOptions,
    analyze,
    render,
)

log = logging.getLogger("worker")

_stop = False


def output_dir(job: Job) -> Path:
    return settings.storage_dir / "output" / job.id


def _opts(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


def claim_next() -> Job | None:
    """Sab se purani queued job → running (atomic: UPDATE ... WHERE status='queued')."""
    with session() as s:
        job = s.exec(select(Job).where(Job.status == JobStatus.queued)
                     .order_by(Job.created_at)).first()
        if job is None:
            return None
        n = s.exec(
            update(Job)
            .where(Job.id == job.id, Job.status == JobStatus.queued)
            .values(status=JobStatus.running, started_at=datetime.now(UTC))
        ).rowcount
        s.commit()
        if n != 1:
            return None  # kisi aur ne utha li
        s.refresh(job)
        return job


class _Reporter:
    """Progress DB mein likhta hai (zyada se zyada 2 dafa / sec) aur cancel check karta hai."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.last = 0.0

    def __call__(self, stage: str, frac: float, message: str) -> None:
        now = time.monotonic()
        if now - self.last < 0.5 and frac < 1.0:
            return
        self.last = now
        with session() as s:
            job = s.get(Job, self.job_id)
            if job is None or job.status == JobStatus.cancelling:
                raise CancelledError()
            job.stage, job.progress, job.message = stage, round(frac, 4), message
            s.add(job)
            s.commit()


def run_job(job: Job) -> None:
    report = _Reporter(job.id)
    with session() as s:
        video = s.get(Video, job.video_id)
        parent = s.get(Job, job.parent_id) if job.parent_id else None
    player = players_store.load(job.player_id)
    if video is None or player is None:
        raise ValueError("Video ya player nahi mila")

    if job.kind == JobKind.analyze:
        result = analyze(video.path, player, _opts(AnalyzeOptions, job.options), report)
    else:
        if parent is None or not parent.result:
            raise ValueError("Render ke liye analyze job ka result chahiye")
        r = render(video.path, player.name, parent.result["segments"], output_dir(job),
                   _opts(RenderOptions, job.options), report)
        root = settings.storage_dir
        result = {
            "shorts": [Path(p).relative_to(root).as_posix() for p in r.shorts],
            "compilation": Path(r.compilation).relative_to(root).as_posix()
            if r.compilation else None,
            "music": r.music,
        }

    with session() as s:
        j = s.get(Job, job.id)
        j.status, j.progress, j.result = JobStatus.done, 1.0, result
        j.message, j.finished_at = "Mukammal", datetime.now(UTC)
        s.add(j)
        s.commit()


def _finish(job_id: str, status: JobStatus, error: str | None = None) -> None:
    with session() as s:
        j = s.get(Job, job_id)
        if j is None:
            return
        j.status, j.error, j.finished_at = status, error, datetime.now(UTC)
        j.message = "Cancel ho gaya" if status == JobStatus.cancelled else "Fail"
        s.add(j)
        s.commit()


def recover_stale() -> None:
    """Pichhli dafa worker beech mein band hua ho to 'running' jobs dobara queue mein."""
    with session() as s:
        for j in s.exec(select(Job).where(Job.status.in_([JobStatus.running,
                                                           JobStatus.cancelling]))):
            j.status = JobStatus.queued if j.status == JobStatus.running else JobStatus.cancelled
            s.add(j)
        s.commit()


def process_one() -> bool:
    job = claim_next()
    if job is None:
        return False
    log.info("Job %s (%s) shuru", job.id, job.kind)
    try:
        run_job(job)
        log.info("Job %s mukammal", job.id)
    except CancelledError:
        _finish(job.id, JobStatus.cancelled)
        log.info("Job %s cancel", job.id)
    except Exception as e:  # noqa: BLE001
        _finish(job.id, JobStatus.failed, f"{type(e).__name__}: {e}")
        log.error("Job %s fail:\n%s", job.id, traceback.format_exc())
    return True


def main(poll: float = 1.0) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(message)s")

    def _sig(*_):
        global _stop
        _stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    recover_stale()
    log.info("Worker tayyar (storage: %s)", settings.storage_dir)
    while not _stop:
        if not process_one():
            time.sleep(poll)


if __name__ == "__main__":
    main()
