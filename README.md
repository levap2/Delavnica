# Delavnica

> CPU-only YOLO object detection in the browser — no GPU, no training, no fuss.

**Delavnica** (Slovenian for *"workshop"*) is a small, reliable demonstration system that runs
pretrained [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) object detection
**on CPU only**, exposes detections through a FastAPI web service, and provides a plain
HTML5/JavaScript browser client for image input.

The goal is a working, reproducible demo — not a production platform.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Project status](#project-status)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [API](#api)
- [Browser client](#browser-client)
- [Configuration](#configuration)
- [CPU-only constraint](#cpu-only-constraint)
- [Performance policy](#performance-policy)
- [Testing](#testing)
- [Known limitations (WO-001)](#known-limitations-wo-001)
- [Contributing & agent workflow](#contributing--agent-workflow)
- [License](#license)

---

## Highlights

- **100% CPU** — inference is explicitly pinned to `device="cpu"`; no CUDA, no NVIDIA driver, no GPU required.
- **Pretrained models only** — no training, fine-tuning, or custom datasets.
- **Ultralytics YOLO** — the official `ultralytics` package is the sole detector; the model is loaded once at startup and reused across requests.
- **Minimal stack** — Python 3, FastAPI, Uvicorn, Ultralytics; the frontend is dependency-light HTML5 + CSS + plain JavaScript (no React/Vue/Node build step).
- **Structured JSON output** — detections come back as class, confidence, and bounding box (not just annotated images), so the browser draws its own overlay.

## Architecture

```text
Browser / HTML5 client
        |
        | HTTP
        v
FastAPI service
        |
        v
Ultralytics YOLO (loaded once at startup)
        |
        v
CPU inference (device="cpu")
        |
        v
JSON detections
        |
        v
Browser overlay / result rendering
```

## Project status

| Component            | Status |
| -------------------- | ------ |
| Project specification | ✅ Done |
| Python environment     | ✅ Done |
| FastAPI service        | ✅ Done |
| YOLO CPU inference     | ✅ Done |
| HTML5 browser client   | ✅ Done (image upload) |
| Automated tests        | ✅ Done |

The full engineering specification, hard constraints, and definition of done live in
[AGENTS.md](AGENTS.md) — read it before contributing, whether you are a human or a coding agent.
The sections below describe the implemented WO-001 demo slice.

## Requirements

- **OS:** Linux, preferably Ubuntu under WSL2 (portable to normal Linux)
- **Python:** 3.10+ (verified on 3.14)
- **Hardware:** any modern CPU; **no GPU is required or used**
- **Packages:** `ultralytics`, `fastapi`, `uvicorn`, `python-multipart`, `Pillow` (recorded in `requirements.txt`)

## Quick start

Commands verified on Ubuntu 26.04 (Linux), Python 3.14.

```bash
# 1. Clone the repository
git clone https://github.com/levap2/Delavnica.git
cd Delavnica

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the service (model loads once at startup)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Open the browser client
#    http://localhost:8000
```

On first run the pretrained `yolo26n.pt` weights (~5 MB) are downloaded
automatically and cached. No GPU is required or used; inference is always
pinned to CPU (see [CPU-only constraint](#cpu-only-constraint)).

## API

### `GET /`

Serves the browser demo page.

### `GET /health`

Liveness plus basic runtime configuration:

```json
{
  "status": "ok",
  "model": "yolo26n.pt",
  "device": "cpu"
}
```

### `POST /detect`

Uploads one image as a multipart form field named `file` and returns structured
detections:

```bash
curl -F "file=@photo.jpg" http://localhost:8000/detect
```

```json
{
  "model": "yolo26n.pt",
  "device": "cpu",
  "image_width": 810,
  "image_height": 1080,
  "inference_ms": 30.1,
  "detections": [
    {
      "class_id": 5,
      "class_name": "bus",
      "confidence": 0.8806,
      "box": [0.0, 230.48, 803.2, 750.7]
    },
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.872,
      "box": [48.61, 397.7, 240.34, 902.41]
    }
  ]
}
```

- `box` is `[x1, y1, x2, y2]` in **original-image pixel coordinates**.
- An image with no detected objects is a success and returns `"detections": []`.
- Detections are sorted by confidence, descending.

The service fails clearly and predictably on invalid uploads, unreadable files, and
inference errors — never with a raw Python stack trace:

| Condition              | HTTP status | Body                                   |
| ---------------------- | ----------- | -------------------------------------- |
| Not a readable image   | 400         | `{"detail": "Uploaded data is not a readable image."}` |
| Empty upload           | 400         | `{"detail": "Uploaded image is empty."}` |
| Missing `file` field   | 422         | `{"detail": "Invalid request: ..."}`    |
| Inference failure      | 500         | `{"detail": "Detection inference failed. See server logs."}` |

The exact schema may evolve; changes are documented.

## Browser client

A simple HTML5 page (CSS + plain JavaScript, no frameworks) served at `GET /`, supporting:

- **Image upload** — pick a file, preview it, run detection, and see bounding boxes,
  class names, and confidence values drawn over the image from the JSON response
- **Webcam capture** — *not implemented in WO-001* (planned for a later work order)

Bounding boxes are returned in original-image pixel coordinates and are scaled to the
displayed size by the browser automatically (the overlay canvas is sized to the image's
natural dimensions).

## Configuration

Configuration is explicit and simple via environment variables (see `app/config.py`),
never hard-coded machine paths:

| Environment variable    | Purpose                                  | Default       |
| ----------------------- | ---------------------------------------- | ------------- |
| `YOLO_MODEL`            | Pretrained detection model name/path     | `yolo26n.pt`  |
| `YOLO_CONF_THRESHOLD`   | Minimum detection confidence to report   | `0.25`        |
| `YOLO_IMGSZ`            | Inference image size                     | `640`         |
| `DETECTOR_HOST`         | Bind address (passed to uvicorn directly)| `0.0.0.0`     |
| `DETECTOR_PORT`         | Bind port (passed to uvicorn directly)   | `8000`        |

> Binding to `0.0.0.0` makes the service reachable from other systems on your LAN per your
> host/network/firewall configuration. This is a demo service — keep that in mind.
> The inference device is **not** configurable: it is always `cpu` (WO-001 product behavior).

## CPU-only constraint

Inference is explicitly pinned to CPU in the detector layer
(`app/detector.py` calls `model.predict(..., device="cpu")`, where the constant
`DEVICE = "cpu"` comes from `app/config.py`). There is no CUDA/GPU code path, no
automatic device selection, and no device configuration. The service works on a
machine with no usable GPU; the verified environment had no NVIDIA driver installed.

## Performance policy

Delavnica targets **usable demo responsiveness on CPU** — a few inference frames per second
where the hardware allows — with correctness and stability ahead of raw throughput.

We measure before we claim: input dimensions, model, CPU inference time, and end-to-end
request time. No FPS figures are published without a corresponding measurement, and
optimization (e.g. OpenVINO/ONNX via Ultralytics export) is considered only after a verified
baseline.

Baseline measured during WO-001 (actual, not projected):

- **CPU:** AMD Ryzen AI 7 350 (8 cores / 16 threads)
- **Model:** `yolo26n.pt`, `imgsz=640`, `conf=0.25`, `device="cpu"`
- **Input:** 810×1080 (`bus.jpg`) — 5 detections; 400×300 blank image — 0 detections
- **Steady-state inference:** ~20–30 ms per image (first call after startup ~0.9 s,
  which includes warm-up)
- **End-to-end HTTP** (upload + inference + response): ~25–30 ms for the 810×1080 image
  over localhost

These numbers are from a single machine; expect different values on different hardware.

## Testing

Run the automated suite:

```bash
.venv/bin/python -m pytest tests/ -v
```

Coverage (per `AGENTS.md` and WO-001):

1. Application imports successfully
2. Server starts
3. `/health` returns success
4. `/detect` accepts a valid test image
5. Detector runs explicitly on CPU
6. JSON response schema is valid
7. Zero-detection results are handled
8. Invalid input is rejected cleanly
9. Browser page can call the backend
10. Documented startup commands actually work

A green test run is evidence, not proof — test coverage is always described alongside results.

## Known limitations (WO-001)

- Image upload only; webcam capture, video, and streaming are not implemented in this slice.
- One inference per request; no batching, no concurrent-inference workers.
- The `yolo26n.pt` weights are downloaded from the internet on first run.
- Single-process demo service: no authentication, persistence, or deployment tooling.
- Bounding boxes are not pixel-perfect for extreme aspect ratios at `imgsz=640`;
  this is expected YOLO behavior, not a rendering bug.

## Contributing & agent workflow

- **Specification & constraints:** see [AGENTS.md](AGENTS.md). It is the source of truth for
  architecture rules (Ultralytics only, CPU only, no training), scope control, dependency
  discipline, and the agent reporting format.
- **Scope:** MVP is the demo service above. Training, GPU support, auth, databases,
  deployment tooling, tracking, and similar features are explicitly out of scope unless
  separately ordered.
- **Style:** small, reviewable changes; no unrelated refactoring.

## License

Distributed under the [Apache License 2.0](LICENSE). See [LICENSE](LICENSE) for details.
