"""CPU-only invariant tests for the detector layer.

These run the *real* ``app/detector.py`` code path (decoding, extraction,
schema mapping) against a stubbed ``ultralytics.YOLO`` and assert that the
inference call explicitly requests ``device="cpu"``.
"""

from __future__ import annotations

import pytest

from conftest import _StubBox

from app.detector import Detector, InferenceError, InvalidImageError


def test_predict_explicitly_requests_cpu(stub_yolo, test_image_bytes):
    detector = Detector("yolo26n.pt")
    stub_yolo.next_boxes = []

    result = detector.detect(test_image_bytes)

    predictor = stub_yolo.instances[-1]
    assert predictor.predict_kwargs is not None
    assert predictor.predict_kwargs["device"] == "cpu"
    assert result.device == "cpu"


def test_result_maps_boxes_names_and_confidence(stub_yolo, test_image_bytes):
    stub_yolo.next_boxes = [
        _StubBox((1.0, 2.0, 3.0, 4.0), 0.8, 0),
        _StubBox((10.0, 20.0, 15.0, 25.0), 0.3, 1),
    ]
    detector = Detector("yolo26n.pt")
    result = detector.detect(test_image_bytes)

    assert len(result.detections) == 2
    assert result.detections[0].class_name == "person"
    assert result.detections[0].confidence == pytest.approx(0.8)
    assert result.detections[0].box == (1.0, 2.0, 3.0, 4.0)
    assert result.detections[1].class_name == "bottle"

    payload = result.to_dict()
    assert payload["model"] == "yolo26n.pt"
    assert payload["device"] == "cpu"
    assert payload["image_width"] == 320
    assert payload["image_height"] == 240
    assert payload["detections"][0]["box"] == [1.0, 2.0, 3.0, 4.0]


def test_boxes_are_clamped_to_image_bounds(stub_yolo, test_image_bytes):
    stub_yolo.next_boxes = [_StubBox((-50.0, -50.0, 500.0, 999.0), 0.9, 0)]
    detector = Detector("yolo26n.pt")
    result = detector.detect(test_image_bytes)

    assert len(result.detections) == 1
    x1, y1, x2, y2 = result.detections[0].box
    assert (x1, y1, x2, y2) == (0.0, 0.0, 320.0, 240.0)


def test_invalid_image_raises_invalid_image_error(stub_yolo):
    detector = Detector("yolo26n.pt")
    with pytest.raises(InvalidImageError):
        detector.detect(b"not an image at all")


def test_empty_image_raises_invalid_image_error(stub_yolo):
    detector = Detector("yolo26n.pt")
    with pytest.raises(InvalidImageError):
        detector.detect(b"")


def test_inference_exception_wrapped(stub_yolo, test_image_bytes):
    detector = Detector("yolo26n.pt")

    def boom(**kwargs):
        raise RuntimeError("kaboom")

    stub_yolo.instances[-1].predict = boom  # type: ignore[method-assign]
    with pytest.raises(InferenceError):
        detector.detect(test_image_bytes)
