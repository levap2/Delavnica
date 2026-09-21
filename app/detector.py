"""CPU-only Ultralytics YOLO detection layer.

This module is the *only* place that talks to Ultralytics. The FastAPI layer
(``app/main.py``) depends on the small :class:`Detector` interface here rather
than calling Ultralytics directly, which keeps HTTP handling separate from
inference and makes the CPU-only invariant easy to verify in one spot.

Design notes (WO-001):
- The model is loaded once and the instance reused across requests.
- Inference is *always* forced to ``device="cpu"``. GPU selection is
  intentionally not configurable in this work order.
- No training, fine-tuning, or dataset code lives here.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, UnidentifiedImageError

from .config import DEVICE

logger = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Base error for detection-layer failures."""


class InvalidImageError(DetectorError):
    """The supplied bytes are not a decodable image."""


class InferenceError(DetectorError):
    """Ultralytics inference raised unexpectedly."""


@dataclass
class Detection:
    """A single bounding box in original-image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 2) for v in self.box],
        }


@dataclass
class DetectionResult:
    """Full result of one inference pass."""

    model_name: str
    device: str
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "device": self.device,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_ms": round(self.inference_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
        }


class Detector:
    """Wraps a single, reused Ultralytics YOLO model on CPU.

    Parameters
    ----------
    model_name:
        Pretrained weight name/path, e.g. ``yolo26n.pt``.
    conf_threshold:
        Minimum confidence to report a detection.
    imgsz:
        Inference image size passed to Ultralytics.
    """

    def __init__(
        self,
        model_name: str,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        # Import lazily so the rest of the app can be imported/tested even if
        # Ultralytics is slow to import.
        from ultralytics import YOLO

        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        logger.info("Loading model %s (device=%s)", model_name, DEVICE)
        try:
            self._model = YOLO(model_name)
        except Exception as exc:  # pragma: no cover - exercised by live run
            logger.exception("Failed to load model %s", model_name)
            raise DetectorError(f"Failed to load model {model_name!r}: {exc}") from exc
        logger.info("Model %s loaded; ready for CPU inference", model_name)

    @property
    def device(self) -> str:
        """The fixed inference device for this service. Always CPU."""
        return DEVICE

    def _decode(self, data: bytes) -> Image.Image:
        """Decode uploaded bytes into an RGB PIL image or raise InvalidImageError."""
        if not data:
            raise InvalidImageError("Uploaded image is empty.")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            return image.convert("RGB")
        except UnidentifiedImageError as exc:
            raise InvalidImageError("Uploaded data is not a readable image.") from exc
        except (OSError, ValueError) as exc:
            raise InvalidImageError(
                f"Uploaded image could not be decoded: {exc}"
            ) from exc

    def detect(self, data: bytes) -> DetectionResult:
        """Run one CPU inference over the raw image bytes."""
        import time

        image = self._decode(data)
        width, height = image.size

        start = time.perf_counter()
        try:
            # The CPU-only invariant: device is hardcoded to "cpu" here and is
            # not derived from any user input or environment variable.
            results = self._model.predict(
                source=image,
                device=DEVICE,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
        except InvalidImageError:
            raise
        except Exception as exc:
            logger.exception("Inference failed")
            raise InferenceError(f"Inference failed: {exc}") from exc
        inference_ms = (time.perf_counter() - start) * 1000.0

        detections = self._extract(results, width, height)
        return DetectionResult(
            model_name=self.model_name,
            device=self.device,
            image_width=width,
            image_height=height,
            inference_ms=inference_ms,
            detections=detections,
        )

    def _extract(
        self, results: list[Any], width: int, height: int
    ) -> list[Detection]:
        """Convert Ultralytics results into JSON-safe Detection objects.

        Ultralytics already returns boxes in the original image coordinate
        space, so no rescaling is needed; we only clamp to the image bounds.
        """
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = result.names or {}
            for box in boxes:
                try:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                except (AttributeError, IndexError, TypeError, ValueError) as exc:
                    logger.warning("Skipping malformed box: %s", exc)
                    continue

                class_name = names.get(cls_id, f"class_{cls_id}")
                if isinstance(class_name, bytes):
                    class_name = class_name.decode("utf-8", "replace")

                # Clamp to original image bounds for a well-formed overlay.
                x1 = max(0.0, min(x1, width))
                y1 = max(0.0, min(y1, height))
                x2 = max(0.0, min(x2, width))
                y2 = max(0.0, min(y2, height))
                if x2 < x1 or y2 < y1:
                    continue

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        box=(x1, y1, x2, y2),
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


def load_detector(
    model_name: str, conf_threshold: float = 0.25, imgsz: int = 640
) -> Detector:
    """Factory used once at application startup."""
    return Detector(model_name, conf_threshold=conf_threshold, imgsz=imgsz)
