"""Players disk par: storage/players/<id>/ (naye) + samples/players/<id>/ (examples).

Har folder: player.json + images — wahi format jo CLI use karta hai (`Player.load`).
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path

from app.config import settings
from pipeline.players import IMAGE_EXT, Player


def _roots() -> list[Path]:
    return [settings.players_dir, settings.samples_dir / "players"]


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "player"


def find_dir(player_id: str) -> Path | None:
    if not re.fullmatch(r"[a-z0-9_]+", player_id):
        return None
    for root in _roots():
        d = root / player_id
        if (d / "player.json").exists():
            return d
    return None


def load(player_id: str) -> Player | None:
    d = find_dir(player_id)
    return Player.load(d) if d else None


def list_all() -> list[tuple[Player, Path]]:
    seen: set[str] = set()
    out = []
    for root in _roots():
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if (d / "player.json").exists() and d.name not in seen:
                seen.add(d.name)
                out.append((Player.load(d), d))
    return out


def create(name: str, jersey_number: str | None, aliases: list[str],
           images: list[tuple[str, bytes]]) -> Player:
    base = slugify(name)
    pid, n = base, 1
    while find_dir(pid):
        n += 1
        pid = f"{base}_{n}"
    d = settings.players_dir / pid
    d.mkdir(parents=True)
    try:
        for i, (fname, data) in enumerate(images, 1):
            ext = Path(fname).suffix.lower()
            if ext not in IMAGE_EXT:
                raise ValueError(f"Image format support nahi: {fname}")
            (d / f"{i}{ext}").write_bytes(data)
        (d / "player.json").write_text(json.dumps({
            "player_id": pid, "player_name": name, "jersey_number": jersey_number,
            "aliases": aliases,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return Player.load(d)


def delete(player_id: str) -> bool:
    d = find_dir(player_id)
    if d is None or settings.players_dir not in d.parents:
        return False  # samples ke players delete nahi
    shutil.rmtree(d)
    return True
