"""ONNX model files — pehli dafa zaroorat par download, phir `models/` folder se.

Sab models commercial-use friendly licenses wale hain (Apache-2.0 / MIT).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from app.config import settings

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS: dict[str, str] = {
    # OpenCV Zoo — Apache-2.0
    "face_detection_yunet_2023mar.onnx": f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def model_path(name: str) -> Path:
    path = settings.models_dir / name
    if path.exists() and path.stat().st_size > 0:
        return path
    url = MODELS.get(name)
    if url is None:
        raise FileNotFoundError(f"Model {name} nahi mila aur download URL maloom nahi")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path
