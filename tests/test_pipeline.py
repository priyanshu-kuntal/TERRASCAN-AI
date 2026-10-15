"""
TERRASCAN-AI — Comprehensive Pipeline & API Test Suite
Validates unit models, data bounds, resolution invariances, and Flask endpoints.
"""

import io
import os
import json
import numpy as np
import pytest
from PIL import Image

from terrascan.utils import load_image, resize_image_for_inference, normalize_map
from terrascan.camouflage import CamouflageDetector
from terrascan.anomaly import AnomalyDetector
from terrascan.objects import ObjectDetector
from terrascan.fusion import FusionEngine
from terrascan.pipeline import TerraScanPipeline
from app import app


@pytest.fixture
def sample_rgb_image():
    """Create a dummy RGB terrain image for testing."""
    np.random.seed(42)
    img = np.random.randint(40, 180, (256, 256, 3), dtype=np.uint8)
    return img


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestImageUtils:
    def test_load_image_rgb(self, sample_rgb_image):
        loaded = load_image(sample_rgb_image)
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == (256, 256, 3)
        assert loaded.dtype == np.uint8

    def test_load_image_grayscale(self):
        gray = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        loaded = load_image(gray)
        assert loaded.shape == (128, 128, 3)
        assert loaded.dtype == np.uint8

    def test_load_image_rgba(self):
        rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        loaded = load_image(rgba)
        assert loaded.shape == (100, 100, 3)
        assert loaded.dtype == np.uint8

    def test_resize_huge_image(self):
        huge = np.zeros((1800, 2400, 3), dtype=np.uint8)
        resized, scale, (orig_h, orig_w) = resize_image_for_inference(huge, max_long_side=1280)
        assert max(resized.shape[:2]) <= 1280
        assert orig_h == 1800
        assert orig_w == 2400
        assert scale < 1.0

    def test_normalize_map_bounds(self):
        arr = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32)
        norm = normalize_map(arr)
        assert norm.min() == 0.0
        assert norm.max() == 1.0

    def test_normalize_uniform_array(self):
        arr = np.ones((10, 10), dtype=np.float32) * 42.0
        norm = normalize_map(arr)
        assert norm.shape == (10, 10)
        assert norm.max() == 0.0


class TestDetectors:
    def test_camouflage_detector(self, sample_rgb_image):
        detector = CamouflageDetector()
        res = detector.detect(sample_rgb_image, threshold=0.5)
        assert "score_map" in res
        assert "binary_mask" in res
        assert "contours" in res
        assert res["score_map"].shape == sample_rgb_image.shape[:2]
        assert 0.0 <= res["score_map"].min()
        assert res["score_map"].max() <= 1.0

    def test_anomaly_detector_self_reference(self, sample_rgb_image):
        detector = AnomalyDetector()
        res = detector.detect(sample_rgb_image)
        assert "score_map" in res
        assert res["score_map"].shape == sample_rgb_image.shape[:2]
        assert 0.0 <= res["score_map"].min()
        assert res["score_map"].max() <= 1.0
        assert "mode" in res

    def test_object_detector_graceful(self, sample_rgb_image):
        detector = ObjectDetector()
        dets = detector.detect(sample_rgb_image, conf_threshold=0.25)
        assert isinstance(dets, list)

    def test_fusion_engine(self, sample_rgb_image):
        engine = FusionEngine()
        h, w = sample_rgb_image.shape[:2]
        camo = np.random.uniform(0.0, 1.0, (h, w)).astype(np.float32)
        anom = np.random.uniform(0.0, 1.0, (h, w)).astype(np.float32)

        susp = engine.compute_suspicion_map(camo, anom, 0.6, 0.4)
        assert susp.shape == (h, w)
        assert 0.0 <= susp.min() <= susp.max() <= 1.0

        regions = engine.extract_top_regions(susp, threshold=0.3, max_regions=5)
        assert isinstance(regions, list)
        assert len(regions) <= 5

        threat = engine.compute_threat_index(susp, regions, [])
        assert 0 <= threat["index"] <= 100
        assert threat["level"] in ("LOW / CLEAR", "ELEVATED", "HIGH CONCERN", "CRITICAL ALERT")


class TestPipeline:
    def test_pipeline_execution(self, sample_rgb_image):
        pipeline = TerraScanPipeline()
        result = pipeline.run_analysis(
            sample_rgb_image,
            camo_threshold=0.5,
            camo_weight=0.6,
            anomaly_weight=0.4,
            object_conf=0.25,
            overlay_opacity=0.6,
            enable_camo=True,
            enable_anomaly=True,
            enable_objects=True,
        )

        assert result["success"] is True
        assert "overlays" in result
        for layer in ("original", "camouflage", "anomaly", "objects", "suspicion", "all_layers"):
            assert layer in result["overlays"]
            assert result["overlays"][layer].startswith("data:image/png;base64,")

        assert 0 <= result["threat_index"] <= 100
        assert "timings" in result
        assert "logs" in result
        assert len(result["logs"]) > 0


class TestFlaskAPI:
    def test_get_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_get_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "OPERATIONAL"
        assert "system" in data

    def test_get_sample(self, client):
        resp = client.get("/api/sample/1")
        assert resp.status_code == 200
        assert resp.mimetype == "image/png"

    def test_analyze_with_sample_id(self, client):
        resp = client.post("/api/analyze", data={"sample_id": 1, "camo_threshold": 0.5})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "threat_index" in data
        assert "overlays" in data

    def test_analyze_with_file_upload(self, client):
        # Create a simple test PNG in memory
        img = Image.new("RGB", (128, 128), color=(40, 80, 40))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/api/analyze",
            data={
                "image": (buf, "test_foliage.png"),
                "camo_threshold": "0.45",
                "camo_weight": "0.7",
                "anomaly_weight": "0.3",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "overlays" in data
        assert "all_layers" in data["overlays"]

    def test_analyze_invalid_payload(self, client):
        resp = client.post("/api/analyze", data={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
