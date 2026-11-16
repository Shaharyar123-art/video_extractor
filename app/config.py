"""Project-wide settings. Environment variables (prefix VE_) ya backend/.env se override ho sakti hain."""

import os
import shutil
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_binary(name: str) -> str:
    """PATH mein dhoondo; na mile to winget ki install location (purane terminals mein PATH
    update nahi hota). Kuch na mile to sirf naam wapas — error baad mein saaf aayega."""
    if found := shutil.which(name):
        return found
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = [local / "Microsoft" / "WinGet" / "Links" / f"{name}.exe"]
    candidates += sorted((local / "Microsoft" / "WinGet" / "Packages").glob(f"Gyan.FFmpeg*/*/bin/{name}.exe"))
    return next((str(c) for c in candidates if c.exists()), name)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VE_", env_file=".env", extra="ignore")

    storage_dir: Path = PROJECT_ROOT / "storage"
    samples_dir: Path = PROJECT_ROOT / "samples"
    music_dir: Path = PROJECT_ROOT / "assets" / "music"
    models_dir: Path = PROJECT_ROOT / "models"  # ONNX models (pehli dafa khud download)

    # "auto" = GPU ho to GPU, warna CPU
    device: str = "auto"

    # Frame sampling rate (frames per second) for analysis
    sample_fps: float = 2.0

    ffmpeg_bin: str = ""
    ffprobe_bin: str = ""

    @model_validator(mode="after")
    def _resolve_binaries(self) -> Settings:
        self.ffmpeg_bin = self.ffmpeg_bin or _find_binary("ffmpeg")
        self.ffprobe_bin = self.ffprobe_bin or _find_binary("ffprobe")
        return self


settings = Settings()
