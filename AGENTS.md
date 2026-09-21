# AGENTS.md

## Project: CPU-Only Ultralytics YOLO Web Detection Service

### Purpose
Build a small, reliable demonstration system that runs pretrained Ultralytics YOLO object detection on CPU only, exposes detection through a web/API service, and provides a browser-based HTML5 client for image or webcam input.

The immediate goal is a working demonstration, not a production platform.

---

## 1. Governing Principles

This project follows an Orchestrated Agentic Programming (OAP) workflow.

- The human lead owns product intent, acceptance, risk, and release decisions.
- The strategic model owns architecture, task decomposition, review, evidence interrogation, and work-order design.
- The coding agent is the execution layer. It implements bounded tasks, runs commands, installs project-local dependencies, tests its work, and reports evidence.
- The coding agent must not redefine the product or silently expand scope.
- Repository state, commits, tests, documentation, and reports are the durable source of project truth.
- Claims of completion must be supported by reproducible evidence.

If instructions conflict, follow this order:
1. Explicit human instruction.
2. Current strategic-model work order.
3. This `AGENTS.md`.
4. Existing repository conventions.

---

## 2. Non-Negotiable Technical Constraints

### 2.1 Ultralytics only
The detector must use the official `ultralytics` Python package and Ultralytics YOLO model APIs.

Do not replace Ultralytics with:
- raw PyTorch model implementations,
- Detectron2,
- OpenCV DNN as the primary detector,
- TensorFlow implementations,
- custom YOLO forks,
- third-party wrappers that bypass the Ultralytics API.

### 2.2 CPU only
The application must run without a GPU.

All inference must explicitly use CPU, for example through Ultralytics configuration equivalent to:

```python
device="cpu"
```

Do not:
- require CUDA,
- require an NVIDIA driver,
- require GPU-specific packages,
- assume GPU availability,
- make GPU execution the default path.

If a dependency attempts to select a GPU automatically, force CPU execution.

### 2.3 No training
This project uses pretrained detection weights only.

Do not add:
- training pipelines,
- dataset preparation,
- annotation tools,
- fine-tuning,
- transfer learning,
- custom class training.

### 2.4 Model choice
Prefer the smallest current Ultralytics detection model suitable for CPU real-time or near-real-time inference.

The starting model should be the Nano-sized pretrained model selected by the strategic work order.

Do not silently switch to a larger model. Any model change must be reported with its expected CPU performance impact.

### 2.5 Target environment
Primary runtime:
- Linux, preferably Ubuntu under WSL2.

The code should remain reasonably portable to normal Linux.

Windows-native support is secondary unless explicitly requested.

---

## 3. MVP Architecture

The intended MVP architecture is:

```text
Browser / HTML5 client
        |
        | HTTP
        v
FastAPI service
        |
        v
Ultralytics YOLO
        |
        v
CPU inference
        |
        v
JSON detections
        |
        v
Browser overlay / result rendering
```

The service should load the YOLO model once during application startup and reuse it across inference requests.

Do not reload the model for every request.

---

## 4. Backend Requirements

Preferred backend stack:
- Python 3
- FastAPI
- Uvicorn
- Ultralytics
- OpenCV and/or Pillow only where useful for image handling

The backend should expose at minimum:

### `GET /health`
Returns a simple health response proving that the HTTP service is alive.

Recommended response:

```json
{
  "status": "ok"
}
```

### `POST /detect`
Accepts an image and performs CPU-only Ultralytics YOLO inference.

The response should contain structured JSON rather than only returning an annotated image.

Each detection should include at least:
- class ID,
- class name,
- confidence,
- bounding box coordinates.

Example shape:

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

Exact schema may be improved, but changes must be documented.

---

## 5. Browser Requirements

The MVP browser UI should be simple and dependency-light.

Preferred implementation:
- HTML5
- CSS
- plain JavaScript

Avoid introducing React, Vue, Angular, Node build pipelines, or other frontend frameworks unless explicitly requested.

The page should support at least one of the following in the first slice:
- upload an image and send it to `/detect`, or
- capture a frame from the browser webcam and send it to `/detect`.

If webcam support is implemented, use the standard browser media APIs such as `navigator.mediaDevices.getUserMedia()`.

Bounding boxes should be drawn in the browser from the JSON detection response.

Do not make continuous high-frame-rate streaming a requirement for the first MVP unless explicitly ordered.

---

## 6. Performance Goals

The project is designed for CPU execution.

Primary performance objective:
- usable demo responsiveness,
- approximately a few inference frames per second where hardware allows,
- correctness and stability before maximum throughput.

Measure actual performance instead of claiming expected performance.

Record at minimum:
- input image dimensions,
- model used,
- CPU inference time,
- approximate end-to-end request time where practical.

Do not claim a specific FPS unless it was measured on the target machine.

---

## 7. Optimization Policy

Start with direct Ultralytics inference using the pretrained model.

Only optimize after the baseline works and has measured performance.

Permitted later optimization paths may include Ultralytics-supported export/runtime paths such as:
- OpenVINO,
- ONNX Runtime.

Any optimization must preserve these project constraints:
- Ultralytics remains the model source and project-level inference framework,
- no GPU requirement,
- no training requirement.

Do not introduce optimization complexity before the baseline implementation is verified.

---

## 8. Scope Control

### In scope for MVP
- Python environment setup.
- Ultralytics installation.
- Pretrained YOLO model download/use.
- Explicit CPU inference.
- FastAPI HTTP service.
- Image detection endpoint.
- Simple HTML5 UI.
- Webcam or image input.
- Bounding-box rendering.
- Basic error handling.
- Basic tests.
- README/run instructions.

### Explicitly out of scope unless separately ordered
- model training,
- custom datasets,
- GPU/CUDA support,
- user authentication,
- database storage,
- cloud deployment,
- Kubernetes,
- Docker orchestration,
- message queues,
- multi-user scaling,
- production TLS termination,
- production security hardening,
- persistent video recording,
- RTSP ingest,
- WebRTC media servers,
- distributed inference,
- multi-camera orchestration,
- analytics dashboards,
- custom object tracking,
- face recognition.

Do not implement out-of-scope features merely because they seem useful.

---

## 9. Dependency Rules

Keep dependencies minimal.

Before adding a dependency, verify that it solves a concrete requirement.

The coding agent may install required packages inside the bounded development environment without asking the human to manually perform routine dependency work.

Durable dependencies must be recorded in the repository using an appropriate dependency file, such as:
- `requirements.txt`, or
- `pyproject.toml`.

Do not leave the environment reproducible only from shell history.

---

## 10. Configuration Rules

Configuration should be explicit and simple.

Useful configurable values may include:
- YOLO model name/path,
- confidence threshold,
- inference image size,
- server host,
- server port.

Prefer environment variables or a small configuration module.

Do not hard-code machine-specific absolute paths.

Default server binding for development may be:

```text
0.0.0.0:8000
```

when external access from the Windows host or LAN is required.

Security implications of network exposure must be documented.

---

## 11. Error Handling

The service must fail clearly and predictably.

Handle at least:
- invalid image uploads,
- unsupported or unreadable files,
- inference exceptions,
- model loading failures,
- missing model weights,
- empty detection results.

Do not return raw Python stack traces to normal API clients.

Useful internal errors may be logged to the console during development.

---

## 12. Testing Requirements

Every implementation task must include relevant verification.

At minimum, verify:
1. the application imports successfully;
2. the server starts;
3. `/health` returns success;
4. `/detect` accepts a valid test image;
5. the detector runs explicitly on CPU;
6. the JSON response schema is valid;
7. zero detections are handled correctly;
8. invalid input is rejected cleanly;
9. the browser page can call the backend;
10. documented startup commands actually work.

Use automated tests where practical.

A green test command is evidence, not proof by itself. The agent must also describe what the tests cover.

---

## 13. CPU Verification Requirement

CPU-only behavior is a release-critical requirement.

The coding agent must provide evidence that inference runs on CPU.

Acceptable evidence may include:
- explicit `device="cpu"` in the inference path,
- runtime log output showing CPU selection,
- absence of CUDA requirements,
- successful execution on a system with no usable GPU.

The coding agent must not report the project complete if CPU-only execution has not been demonstrated.

---

## 14. Security and Runtime Boundaries

This is a demo project, but basic discipline still applies.

Do not:
- add real secrets to the repository,
- commit passwords,
- expose unrelated host files,
- use production credentials,
- run destructive host-level commands unless explicitly required,
- alter Windows host configuration unnecessarily.

Prefer performing installation and runtime work inside WSL/Linux.

If network binding beyond localhost is enabled, document that the service becomes reachable from other systems according to host/network/firewall configuration.

---

## 15. Repository Discipline

Before changing files:
- inspect the repository,
- identify existing conventions,
- identify unrelated uncommitted changes,
- do not overwrite human work.

Keep each task narrow.

Do not perform unrelated refactoring while implementing a feature.

Do not rename files, reorganize directories, or change style globally unless required by the work order.

Prefer small, reviewable changes.

---

## 16. Documentation Requirements

The repository must contain concise run instructions.

The README should eventually explain:
- project purpose,
- requirements,
- WSL/Linux setup,
- Python environment creation,
- dependency installation,
- server startup,
- browser access,
- API usage,
- model used,
- explicit CPU-only behavior,
- known limitations.

Documentation must describe what has actually been verified.

Do not write "production ready", "real time", "high performance", or similar claims without evidence.

---

## 17. Definition of Done for a Work Order

A coding task is complete only when:
- requested behavior is implemented;
- scope stayed within the work order;
- required tests were run;
- test results are reported;
- CPU-only constraints remain intact;
- documentation is updated when behavior changed;
- known limitations are disclosed;
- no unrelated files were intentionally changed;
- the agent provides a concise evidence-backed report.

"Code written" is not equivalent to "done".

---

## 18. Required Agent Report

At the end of every task, report using this structure:

### Summary
What was implemented in 2-5 sentences.

### Files changed
List each changed file and why it changed.

### Verification
List every command or test executed and whether it passed or failed.

### CPU-only evidence
State exactly how CPU-only execution was verified.

### Runtime evidence
If the server was started, report:
- command used,
- bind address and port,
- tested endpoint(s),
- observed result.

### Performance evidence
If measured, report:
- model,
- input size,
- inference time,
- machine/CPU information when available.

Do not invent measurements.

### Deviations
Describe any deviation from the work order or this file.

### Known limitations
List remaining limitations, skipped tests, assumptions, and unresolved issues.

### Recommended next step
Recommend one bounded next task. Do not silently begin it unless it is part of the current work order.

---

## 19. Stop / Escalation Conditions

Stop and report instead of improvising when:
- a requested change violates CPU-only execution;
- fulfilling the task would require training a model;
- fulfilling the task would require replacing Ultralytics as the primary detector;
- requirements conflict materially;
- a destructive operation could affect valuable host data;
- production secrets or credentials appear in scope;
- the requested change would substantially expand the architecture beyond the work order;
- verification cannot be completed.

When blocked, explain the blocker and propose the smallest safe resolution.

Do not turn the human into a command relay for routine setup work that can safely be done inside the execution environment.

---

## 20. Current Product Direction

Unless superseded by a newer strategic work order, the current direction is:

> Build a minimal WSL2/Linux-hosted HTTP object-detection service using pretrained Ultralytics YOLO, running exclusively on CPU, with a simple FastAPI backend and HTML5 browser client capable of submitting images or webcam frames and displaying returned detections.

The first implementation should optimize for:
1. demo reliability,
2. simplicity,
3. CPU correctness,
4. reproducibility,
5. measurable performance,
6. clean evidence for strategic review.

