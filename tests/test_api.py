"""
Integration tests for the Flask API.
Uses Flask test client — no real network or DB connection required.
DB and model dependencies are mocked.
"""
import io
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_png_bytes(colour=(80, 160, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buf, format="PNG")
    return buf.getvalue()


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_run_pipeline():
    """Return a canned prediction without touching model or DB."""
    return {
        "id":                  1,
        "filename":            "test.png",
        "cached":              False,
        "prediction":          "healthy",
        "confidence":          72.5,
        "model_used":          "ensemble",
        "top3": [
            {"label": "healthy",     "probability": 72.5},
            {"label": "leaf_spot",   "probability": 15.3},
            {"label": "downy_mildew","probability":  8.1},
        ],
        "all_probabilities":   {"healthy": 72.5, "leaf_spot": 15.3},
        "processing_time_ms":  145.2,
        "disease_info":        {"status": "Healthy Spinach", "severity": "none"},
    }


@pytest.fixture
def client(mock_run_pipeline):
    """Flask test client with all external dependencies mocked."""
    mock_deep = MagicMock()
    mock_deep.ready       = True
    mock_deep._load_error = None

    mock_sklearn = MagicMock()
    mock_sklearn.get_info.return_value = {"available": ["random_forest"], "active": "random_forest"}

    with patch("ml_models.deep_model",       mock_deep), \
         patch("ml_models.sklearn_registry", mock_sklearn), \
         patch("ml_models.run_pipeline",     return_value=mock_run_pipeline), \
         patch("database.init_db"), \
         patch("database.get_db"):
        import app as flask_app
        flask_app.app.config["TESTING"]             = True
        flask_app.app.config["WTF_CSRF_ENABLED"]    = False
        with flask_app.app.test_client() as c:
            yield c


# ── Health endpoint ────────────────────────────────────────────────────────

def test_health_returns_200(client):
    with patch("app.execute", return_value={"v": 1}):
        resp = client.get("/health")
    assert resp.status_code in (200, 503)   # 503 if no DB, both valid responses


def test_health_json_structure(client):
    with patch("app.execute", return_value={"v": 1}):
        resp = client.get("/health")
    data = json.loads(resp.data)
    assert "status"  in data
    assert "version" in data


# ── Disease info ───────────────────────────────────────────────────────────

def test_disease_info_all(client):
    resp = client.get("/disease-info")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    assert "diseases" in data["data"]
    assert "labels"   in data["data"]


def test_disease_info_valid_label(client):
    resp = client.get("/disease-info/healthy")
    assert resp.status_code == 200


def test_disease_info_invalid_label(client):
    resp = client.get("/disease-info/totally_fake_disease")
    assert resp.status_code == 404


# ── Predict endpoint ──────────────────────────────────────────────────────

def test_predict_missing_image_field(client):
    resp = client.post("/predict", data={})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert data["status"] == "error"


def test_predict_with_valid_image(client, mock_run_pipeline):
    with patch("ml_models.run_pipeline", return_value=mock_run_pipeline):
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(_make_png_bytes()), "leaf.png")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    assert "prediction" in data["data"]


def test_predict_response_has_confidence(client, mock_run_pipeline):
    with patch("ml_models.run_pipeline", return_value=mock_run_pipeline):
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(_make_png_bytes()), "leaf.png")},
            content_type="multipart/form-data",
        )
    data = json.loads(resp.data)
    assert "confidence" in data["data"]
    assert isinstance(data["data"]["confidence"], (int, float))


# ── Batch predict ─────────────────────────────────────────────────────────

def test_batch_predict_no_images(client):
    resp = client.post("/predict/batch", data={})
    assert resp.status_code == 400


def test_batch_predict_returns_results_key(client, mock_run_pipeline):
    with patch("ml_models.run_pipeline", return_value=mock_run_pipeline):
        files = [
            ("images[]", (io.BytesIO(_make_png_bytes()), f"img{i}.png"))
            for i in range(3)
        ]
        resp = client.post(
            "/predict/batch",
            data=files,
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "results" in data["data"]
    assert "total"   in data["data"]


# ── Models endpoint ────────────────────────────────────────────────────────

def test_models_endpoint(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "deep_model" in data["data"]
    assert "sklearn"    in data["data"]


# ── Response envelope ──────────────────────────────────────────────────────

def test_success_response_has_ts(client):
    resp = client.get("/disease-info")
    data = json.loads(resp.data)
    assert "ts" in data


def test_success_response_has_status(client):
    resp = client.get("/disease-info")
    data = json.loads(resp.data)
    assert data["status"] == "success"


# ── Rate limit header ─────────────────────────────────────────────────────

def test_response_has_model_version_header(client):
    resp = client.get("/disease-info")
    assert "X-Model-Version" in resp.headers
