"""FastAPI app.  Chalayen:  uv run uvicorn app.main:app --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.db import engine

BACKEND_DIR = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine()  # tables bana do
    proc = None
    if settings.embedded_worker:
        proc = subprocess.Popen([sys.executable, "-m", "app.worker"], cwd=BACKEND_DIR)  # noqa: ASYNC220
    app.state.worker_proc = proc
    try:
        yield
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def create_app() -> FastAPI:
    app = FastAPI(title="VideoExtractor API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    out = settings.storage_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    # Sirf output folder public (shorts / compilations) — cache, uploads, db nahi
    app.mount("/media/output", StaticFiles(directory=out), name="media")
    return app


app = create_app()
