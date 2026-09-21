"""FastAPI application: HTTP layer for the CPU-only YOLO detection demo.

The inference implementation lives in ``app/detector.py``; this module only
handles HTTP concerns (routing, validation errors, serving the browser demo).

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import DEVICE, Settings
from .detector import (
    Detector,
    DetectorError,
    InferenceError,
    InvalidImageError,
    load_detector,
)

logger = logging.getLogger(__name__)

#: Simple development logging so startup/informational messages are visible
#: when running under uvicorn (which only configures its own loggers).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


def _json_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def create_app(detector: Detector | None = None) -> FastAPI:
    """Build the FastAPI app.

    ``detector`` may be injected (used by tests). When omitted, a real
    :class:`~app.detector.Detector` is created once during application startup
    and reused for every request — the model is never reloaded per request.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = Settings.from_env()
        if detector is not None:
            app.state.detector = detector
        else:
            settings = app.state.settings
            try:
                app.state.detector = load_detector(
                    model_name=settings.model_name,
                    conf_threshold=settings.conf_threshold,
                    imgsz=settings.imgsz,
                )
            except DetectorError:
                # Fail startup loudly rather than serving a misleadingly
                # healthy endpoint.
                raise
        logger.info(
            "Server startup complete: model=%s device=%s",
            app.state.detector.model_name,
            DEVICE,
        )
        yield
        app.state.detector = None

    app = FastAPI(
        title="Delavnica YOLO detection",
        version="0.1.0",
        description="CPU-only Ultralytics YOLO image detection demo (WO-001).",
        lifespan=lifespan,
    )

    @app.exception_handler(InvalidImageError)
    async def invalid_image_handler(_: Request, exc: InvalidImageError):
        return _json_error(400, str(exc))

    @app.exception_handler(InferenceError)
    async def inference_error_handler(_: Request, exc: InferenceError):
        logger.error("Inference error: %s", exc)
        return _json_error(500, "Detection inference failed. See server logs.")

    @app.exception_handler(DetectorError)
    async def detector_error_handler(_: Request, exc: DetectorError):
        return _json_error(500, str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        return _json_error(422, "Invalid request: missing or malformed 'file' field.")

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException):
        # Keep Starlette's own error responses JSON-shaped (e.g. 404, 405).
        return _json_error(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return _json_error(500, "Internal server error. See server logs.")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the browser demo page."""
        return FileResponse(INDEX_HTML)

    @app.get("/health")
    async def health() -> dict:
        """Liveness plus basic runtime configuration."""
        det: Detector = app.state.detector
        return {
            "status": "ok",
            "model": det.model_name,
            "device": DEVICE,
        }

    @app.post("/detect")
    async def detect(file: UploadFile = File(...)) -> dict:
        """Run CPU inference on one uploaded image and return JSON detections."""
        det: Detector = app.state.detector
        data = await file.read()
        result = det.detect(data)
        return result.to_dict()

    return app


#: Application instance for ``uvicorn app.main:app``.
app = create_app()
