"""Shared fixtures for the Delavnica test suite.

The expensive YOLO inference is stubbed at the ``ultralytics.YOLO`` package
boundary. Everything in ``app/detector.py`` (image decoding, box extraction,
schema mapping) and ``app/main.py`` (routing, error handling) runs for real,
and the stub records the ``predict()`` call kwargs so the CPU-only invariant
can be asserted.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

# Make the repository root importable regardless of how pytest is invoked.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_image_bytes(width: int = 320, height: int = 240, color=(120, 200, 90)) -> bytes:
    """Create an in-memory PNG (no binary test assets are committed)."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _StubBox:
    """Mimics the attribute surface of one Ultralytics box row."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = [list(xyxy)]
        self.conf = [conf]
        self.cls = [cls]


class _StubResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _StubYOLO:
    """Drop-in stand-in for ``ultralytics.YOLO`` used by monkeypatching."""

    instances: list[_StubYOLO] = []
    next_boxes: list[_StubBox] = []
    names: dict = {0: "person", 1: "bottle"}

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.predict_kwargs: dict | None = None
        _StubYOLO.instances.append(self)

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [_StubResult(self.names, self.next_boxes)]


@pytest.fixture
def stub_yolo(monkeypatch):
    """Patch ``ultralytics.YOLO`` with the recording stub."""
    import ultralytics

    _StubYOLO.instances = []
    _StubYOLO.next_boxes = []
    _StubYOLO.names = {0: "person", 1: "bottle"}
    monkeypatch.setattr(ultralytics, "YOLO", _StubYOLO)
    return _StubYOLO


@pytest.fixture
def test_image_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture
def client(stub_yolo):
    """FastAPI TestClient wired to a real Detector (with stubbed YOLO)."""
    from fastapi.testclient import TestClient

    from app.detector import Detector
    from app.main import create_app

    detector = Detector("yolo26n.pt")
    app = create_app(detector=detector)
    with TestClient(app) as test_client:
        yield test_client
