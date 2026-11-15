"""Phase 0 environment check.

Chalayen:  cd backend && uv run python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}

ok = True


def report(passed: bool, label: str, detail: str = "") -> None:
    global ok
    ok &= passed
    mark = "OK  " if passed else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    report(v >= (3, 14), "Python", f"{v.major}.{v.minor}.{v.micro}")


def check_binary(name: str) -> None:
    path = shutil.which(name)
    if not path:
        report(False, Path(name).stem, "PATH mein nahi mila (naya terminal kholein)")
        return
    out = subprocess.run([name, "-version"], capture_output=True, text=True).stdout
    report(True, Path(name).stem, out.splitlines()[0] if out else path)


def check_imports() -> None:
    for mod in ("numpy", "cv2", "pydantic", "typer"):
        try:
            m = importlib.import_module(mod)
            report(True, f"import {mod}", getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            report(False, f"import {mod}", str(e))


def check_samples() -> None:
    videos = [p for p in (settings.samples_dir / "videos").glob("*") if p.suffix.lower() in VIDEO_EXT]
    players = [p for p in (settings.samples_dir / "players").glob("*") if p.is_dir()]
    keys = list((settings.samples_dir / "answer_keys").glob("*.json"))

    # Samples abhi optional hain — warning, fail nahi
    print(f"[INFO] Sample videos: {len(videos)}")
    for v in videos:
        info = probe(v)
        print(f"       - {v.name}: {info}")
    print(f"[INFO] Player folders: {len(players)}")
    for p in players:
        imgs = [i for i in p.iterdir() if i.suffix.lower() in IMAGE_EXT]
        has_meta = (p / "player.json").exists()
        print(f"       - {p.name}: {len(imgs)} images, player.json={'yes' if has_meta else 'NO'}")
    real_keys = [k for k in keys if not k.name.startswith("_")]
    print(f"[INFO] Answer keys: {len(real_keys)}")


def probe(path: Path) -> str:
    try:
        out = subprocess.run(
            [settings.ffprobe_bin, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        data = json.loads(out)
        vs = next(s for s in data["streams"] if s["codec_type"] == "video")
        dur = float(data["format"]["duration"])
        has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
        return f"{vs['width']}x{vs['height']}, {dur / 60:.1f} min, audio={'yes' if has_audio else 'NO'}"
    except Exception as e:  # noqa: BLE001
        return f"probe failed: {e}"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    check_python()
    check_binary(settings.ffmpeg_bin)
    check_binary(settings.ffprobe_bin)
    check_imports()
    check_samples()
    print("\nSab theek hai ✅" if ok else "\nKuch masle hain ❌")
    sys.exit(0 if ok else 1)
