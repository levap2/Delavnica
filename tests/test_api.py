"""HTTP-layer tests for GET /, GET /health and POST /detect.

Inference is stubbed at the ultralytics boundary (see tests/conftest.py);
decoding, extraction, JSON schema, and error handling all run for real.
"""

from __future__ import annotations

from conftest import _StubBox


def _post_image(client, data: bytes, filename="photo.png", content_type="image/png"):
    return client.post(
        "/detect",
        files={"file": (filename, data, content_type)},
    )


# ---------------------------------------------------------------- health


def test_health_reports_ok_cpu_and_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["model"] == "yolo26n.pt"


# ---------------------------------------------------------------- index


def test_index_serves_browser_demo(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert 'type="file"' in html
    assert "/detect" in html


def test_index_includes_webcam_capture_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "getUserMedia" in html          # standard browser media API
    assert "<video" in html                # live camera preview
    assert "Capture &amp; detect" in html  # single-frame capture action


# ---------------------------------------------------------------- detect


def test_detect_valid_image_returns_schema(client, test_image_bytes, stub_yolo):
    stub_yolo.next_boxes = [
        _StubBox((10.5, 20.25, 110.0, 220.75), 0.93, 0),
        _StubBox((5.0, 5.0, 30.0, 40.0), 0.41, 1),
    ]
    response = _post_image(client, test_image_bytes)
    assert response.status_code == 200
    body = response.json()

    assert body["model"] == "yolo26n.pt"
    assert body["device"] == "cpu"
    assert body["image_width"] == 320
    assert body["image_height"] == 240
    assert isinstance(body["inference_ms"], (int, float))
    assert isinstance(body["detections"], list)
    assert len(body["detections"]) == 2

    first = body["detections"][0]
    for key in ("class_id", "class_name", "confidence", "box"):
        assert key in first
    assert first["class_id"] == 0
    assert first["class_name"] == "person"
    assert 0.0 <= first["confidence"] <= 1.0
    box = first["box"]
    assert len(box) == 4
    assert all(isinstance(v, (int, float)) for v in box)
    # Box must stay inside the original image bounds.
    assert 0 <= box[0] < 320
    assert 0 <= box[1] < 240
    assert 0 < box[2] <= 320
    assert 0 < box[3] <= 240
    assert box[2] > box[0] and box[3] > box[1]

    # Sorted by confidence, descending.
    confs = [d["confidence"] for d in body["detections"]]
    assert confs == sorted(confs, reverse=True)


def test_detect_accepts_jpeg_frame(client, test_jpeg_bytes, stub_yolo):
    """Webcam frames arrive as JPEG; the endpoint must accept them."""
    stub_yolo.next_boxes = []
    response = client.post(
        "/detect", files={"file": ("webcam_frame.jpg", test_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["image_width"] == 320
    assert body["image_height"] == 240
    assert body["detections"] == []


def test_detect_zero_detections_is_success(client, test_image_bytes, stub_yolo):
    stub_yolo.next_boxes = []
    response = _post_image(client, test_image_bytes)
    assert response.status_code == 200
    body = response.json()
    assert body["detections"] == []


def test_detect_uses_default_cpu_conf_and_imgsz(client, test_image_bytes, stub_yolo):
    stub_yolo.next_boxes = []
    assert _post_image(client, test_image_bytes).status_code == 200
    kwargs = stub_yolo.instances[-1].predict_kwargs
    assert kwargs["device"] == "cpu"
    assert kwargs["conf"] == 0.25
    assert kwargs["imgsz"] == 640


# ---------------------------------------------------------------- errors


def test_detect_invalid_bytes_rejected_cleanly(client):
    response = _post_image(client, b"this is definitely not an image")
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert "detail" in body
    assert "Traceback" not in response.text


def test_detect_empty_upload_rejected_cleanly(client):
    response = _post_image(client, b"")
    assert response.status_code == 400
    assert "Traceback" not in response.text


def test_detect_missing_file_field_rejected(client):
    response = client.post("/detect")
    assert response.status_code == 422
    assert "Traceback" not in response.text


def test_unknown_route_returns_json(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
