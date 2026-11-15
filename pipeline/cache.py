"""Har video ka apna cache folder (scenes, frames, OCR results waghera dobara compute na hon)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import settings


def video_key(path: str | Path) -> str:
    """Path + size + modified-time se chhota stable id (poori file hash karna slow hai)."""
    p = Path(path).resolve()
    st = p.stat()
    raw = f"{p}|{st.st_size}|{int(st.st_mtime)}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def cache_dir(path: str | Path) -> Path:
    d = settings.storage_dir / "cache" / video_key(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_json(path: Path) -> Any | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
