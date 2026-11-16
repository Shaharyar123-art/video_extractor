"""SQLite database (SQLModel). Videos aur jobs; players disk par (player.json + images)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import event
from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine

from app.config import settings


def _now() -> datetime:
    return datetime.now(UTC)


def _id() -> str:
    return uuid.uuid4().hex[:12]


class Video(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    filename: str
    path: str  # absolute path on disk
    uploaded: bool = True  # False = local file register kiya (delete par file nahi mitate)
    duration: float = 0.0
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=_now)


class JobKind(StrEnum):
    analyze = "analyze"
    render = "render"


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelling = "cancelling"
    cancelled = "cancelled"


class Job(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    kind: JobKind
    video_id: str = Field(foreign_key="video.id", index=True)
    player_id: str
    parent_id: str | None = None  # render job → kis analyze job ke segments
    options: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: JobStatus = Field(default=JobStatus.queued, index=True)
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


_engine = None


def engine():
    global _engine
    if _engine is None:
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        url = settings.database_url or f"sqlite:///{settings.storage_dir / 'app.db'}"
        _engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(_engine, "connect")
        def _pragma(conn, _):  # API + worker dono ek saath likhte hain → WAL mode
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        SQLModel.metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Tests ke liye (naya storage_dir)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def session() -> Session:
    return Session(engine(), expire_on_commit=False)
