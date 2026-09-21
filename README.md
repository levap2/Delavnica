# Delavnica

> CPU-only YOLO object detection in the browser — no GPU, no training, no fuss.

**Delavnica** (Slovenian for *"workshop"*) is a small, reliable demonstration system that runs
pretrained [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) object detection
**on CPU only**, exposes detections through a FastAPI web service, and provides a plain
HTML5/JavaScript browser client for image and webcam input.

The goal is a working, reproducible demo — not a production platform.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Project status](#project-status)
- [Requirements](#requirements)
- [Planned quick start](#planned-quick-start)
- [Planned API](#planned-api)
- [Browser client](#browser-client)
- [Configuration](#configuration)
- [Performance policy](#performance-policy)
- [Testing](#testing)
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

| Component            | Status     |
| -------------------- | ---------- |
| Project specification | ✅ Done    |
| Python environment     | 🔜 Planned |
| FastAPI service        | 🔜 Planned |
| YOLO CPU inference     | 🔜 Planned |
| HTML5 browser client   | 🔜 Planned |
| Automated tests        | 🔜 Planned |

The repository currently contains the project specification and workflow rules.
**The full engineering specification, hard constraints, and definition of done live in
[AGENTS.md](AGENTS.md)** — read it before contributing, whether you are a human or a coding agent.
The sections below describe the *planned* design as specified there.

## Requirements

- **OS:** Linux, preferably Ubuntu under WSL2 (portable to normal Linux)
- **Python:** 3.10+
- **Hardware:** any modern CPU; **no GPU is required or used**
- **Packages:** `ultralytics`, `fastapi`, `uvicorn` (recorded in a dependency file once code lands)

## Planned quick start

> ⚠️ Commands below describe the intended setup and will work once the service is implemented.

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

## Planned API

### `GET /health`

Liveness check:

```json
{
  "status": "ok"
}
```

### `POST /detect`

Sends an image and returns structured detections:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.94,
      "box": [120.2, 80.1, 420.7, 610.5]
    }
  ],
  "inference_ms": 143.2
}
```

The exact schema may evolve; changes are documented. The service fails clearly and
predictably on invalid uploads, unreadable files, and inference errors — never with a raw
Python stack trace.

## Browser client

A simple HTML5 page (CSS + plain JavaScript, no frameworks) supporting:

- **Image upload** — pick a file and send it to `/detect`
- **Webcam capture** — grab a frame via `navigator.mediaDevices.getUserMedia()` and send it to `/detect`

Bounding boxes are drawn in the browser directly from the JSON response.

## Configuration

Configuration is explicit and simple (environment variables or a small config module),
never hard-coded machine paths:

| Setting                  | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| YOLO model name/path     | Pretrained detection model to load           |
| Confidence threshold     | Minimum detection confidence to report       |
| Inference image size     | Input size for YOLO                          |
| Server host / port       | Bind address (default `0.0.0.0:8000` in dev) |

> Binding to `0.0.0.0` makes the service reachable from other systems on your LAN per your
> host/network/firewall configuration. This is a demo service — keep that in mind.

## Performance policy

Delavnica targets **usable demo responsiveness on CPU** — a few inference frames per second
where the hardware allows — with correctness and stability ahead of raw throughput.

We measure before we claim: input dimensions, model, CPU inference time, and end-to-end
request time. No FPS figures are published without a corresponding measurement, and
optimization (e.g. OpenVINO/ONNX via Ultralytics export) is considered only after a verified
baseline.

## Testing

Verification coverage includes (per `AGENTS.md`):

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
