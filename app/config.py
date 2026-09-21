"""Explicit, simple configuration for the detection service.

Values can be overridden with environment variables; defaults match the
WO-001 baseline. ``DEVICE`` is intentionally *not* configurable: CPU-only
execution is product behavior for this project version.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: CPU-only is a deliberate, non-configurable product decision (WO-001).
DEVICE = "cpu"

_DEFAULT_MODEL = "yolo26n.pt"
_DEFAULT_CONF = 0.25
_DEFAULT_IMGSZ = 640


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"Environment variable {name} must be a number, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Environment variable {name} must be an integer, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for one service process."""

    model_name: str = _DEFAULT_MODEL
    conf_threshold: float = _DEFAULT_CONF
    imgsz: int = _DEFAULT_IMGSZ
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_name=os.environ.get("YOLO_MODEL", _DEFAULT_MODEL),
            conf_threshold=_env_float("YOLO_CONF_THRESHOLD", _DEFAULT_CONF),
            imgsz=_env_int("YOLO_IMGSZ", _DEFAULT_IMGSZ),
            host=os.environ.get("DETECTOR_HOST", "0.0.0.0"),
            port=_env_int("DETECTOR_PORT", 8000),
        )
