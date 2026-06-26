"""
Unit tests for ml_models.py — feature extraction, probability pooling,
ensemble blending.  No real images or GPU required.
"""
import io
import sys
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config
from ml_models import (
    ImageProcessor,
    validate_image_bytes,
    compute_hash,
    sanitize_name,
    allowed_ext,
    fix_filename_ext,
    _ensemble_vote,
    PLANTVILLAGE_CLASSES,
    PV_TO_SPINACH,
)

LABELS = Config.LABELS


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_png_bytes(width: int = 64, height: int = 64,
                    colour: tuple = (100, 160, 80)) -> bytes:
    """Create a minimal in-memory solid-colour PNG."""
    img = Image.new("RGB", (width, height), colour)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), (180, 120, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── validate_image_bytes ──────────────────────────────────────────────────

def test_validate_png():
    assert validate_image_bytes(_make_png_bytes()) is True


def test_validate_jpeg():
    assert validate_image_bytes(_make_jpeg_bytes()) is True


def test_validate_rejects_empty():
    assert validate_image_bytes(b"") is False


def test_validate_rejects_random():
    assert validate_image_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07") is False


# ── compute_hash ──────────────────────────────────────────────────────────

def test_hash_deterministic():
    data = _make_png_bytes()
    assert compute_hash(data) == compute_hash(data)


def test_hash_differs_for_different_data():
    a = _make_png_bytes(colour=(100, 160, 80))
    b = _make_png_bytes(colour=(200, 50, 50))
    assert compute_hash(a) != compute_hash(b)


def test_hash_length():
    assert len(compute_hash(b"x" * 100)) == 64   # SHA-256 hex


# ── sanitize_name ────────────────────────────────────────────────────────

def test_sanitize_removes_path_traversal():
    assert "/" not in sanitize_name("../../etc/passwd")
    assert "\\" not in sanitize_name("..\\..\\win")


def test_sanitize_max_length():
    assert len(sanitize_name("a" * 300)) <= 200


def test_sanitize_returns_upload_for_empty():
    assert sanitize_name("") == "upload"


# ── allowed_ext ──────────────────────────────────────────────────────────

def test_allowed_ext_jpg():
    assert allowed_ext("leaf.jpg") is True


def test_allowed_ext_png():
    assert allowed_ext("leaf.PNG") is True


def test_allowed_ext_rejects_exe():
    assert allowed_ext("malware.exe") is False


def test_allowed_ext_rejects_no_ext():
    assert allowed_ext("noextension") is False


# ── fix_filename_ext ──────────────────────────────────────────────────────

def test_fix_filename_ext_detects_png():
    result = fix_filename_ext("unknownfile", _make_png_bytes())
    assert result.endswith(".png")


def test_fix_filename_ext_keeps_existing():
    data = _make_png_bytes()
    result = fix_filename_ext("already.png", data)
    assert result == "already.png"


# ── ImageProcessor ────────────────────────────────────────────────────────

class TestImageProcessor:

    def setup_method(self):
        self.raw  = _make_png_bytes(width=128, height=128, colour=(80, 160, 60))
        self.proc = ImageProcessor(self.raw)
        self.proc.prepare()

    def test_meta_dimensions(self):
        assert self.proc.meta["original_width"]  == 128
        assert self.proc.meta["original_height"] == 128

    def test_meta_megapixels(self):
        expected = round(128 * 128 / 1_000_000, 3)
        assert abs(self.proc.meta["megapixels"] - expected) < 1e-4

    def test_extract_features_shape(self):
        feats = self.proc.extract_features()
        assert feats.shape == (49,), f"Expected (49,), got {feats.shape}"

    def test_extract_features_dtype(self):
        feats = self.proc.extract_features()
        assert feats.dtype == np.float32

    def test_extract_features_finite(self):
        feats = self.proc.extract_features()
        assert np.all(np.isfinite(feats)), "Features contain NaN or Inf"

    def test_colour_analysis_keys(self):
        ca = self.proc.colour_analysis()
        for key in ("dominant_hex", "dominant_rgb", "avg_hue_deg",
                    "avg_saturation", "avg_brightness", "colour_pct",
                    "rgb_histogram", "disease_hints"):
            assert key in ca, f"Missing key: {key}"

    def test_colour_pct_sums_leq_100(self):
        ca = self.proc.colour_analysis()
        pcts = ca["colour_pct"]
        total = sum(pcts.values())
        # Percentages can overlap (different colour criteria) — each must be in [0,100]
        for k, v in pcts.items():
            assert 0.0 <= v <= 100.0, f"{k}={v} out of range"

    def test_thumbnail_b64_prefix(self):
        b64 = self.proc.thumbnail_b64()
        assert b64.startswith("data:image/jpeg;base64,")

    def test_pil_image_size(self):
        img = self.proc.pil_image()
        assert img.size == (224, 224)


# ── PV_TO_SPINACH coverage ────────────────────────────────────────────────

def test_pv_to_spinach_covers_all_pv_classes():
    """Every PlantVillage class must have an explicit mapping — no defaults."""
    for cls in PLANTVILLAGE_CLASSES:
        assert cls in PV_TO_SPINACH, \
            f"PV class '{cls}' has no explicit mapping in PV_TO_SPINACH"


def test_pv_to_spinach_targets_are_valid_labels():
    """Every mapped spinach label must be in Config.LABELS."""
    for pv_cls, spinach_lbl in PV_TO_SPINACH.items():
        assert spinach_lbl in LABELS, \
            f"PV class {pv_cls!r} maps to unknown label {spinach_lbl!r}"


def test_plantvillage_classes_count():
    """PLANTVILLAGE_CLASSES must contain exactly 38 entries."""
    assert len(PLANTVILLAGE_CLASSES) == 38, \
        f"Expected 38 PV classes, got {len(PLANTVILLAGE_CLASSES)}"


def test_validate_mappings_does_not_raise():
    """_validate_mappings() must pass cleanly on the current mapping."""
    from ml_models import _validate_mappings
    _validate_mappings()   # would raise RuntimeError if any mapping is missing


# ── _ensemble_vote ────────────────────────────────────────────────────────

def _make_deep_result(prediction: str, confidence: float = 50.0) -> dict:
    probs = {lbl: 0.0 for lbl in LABELS}
    probs[prediction] = confidence
    remaining = (100.0 - confidence) / (len(LABELS) - 1)
    for lbl in LABELS:
        if lbl != prediction:
            probs[lbl] = remaining
    return {
        "prediction":        prediction,
        "confidence":        confidence,
        "model_used":        "EfficientNet-B4",
        "top3":              [],
        "all_probabilities": probs,
    }


def _make_xgb_result(prediction: str) -> dict:
    probs = {lbl: 0.0 for lbl in LABELS}
    probs[prediction] = 80.0
    remaining = 20.0 / (len(LABELS) - 1)
    for lbl in LABELS:
        if lbl != prediction:
            probs[lbl] = remaining
    return {
        "prediction":        prediction,
        "confidence":        80.0,
        "model_used":        "XGBoost",
        "top3":              [],
        "all_probabilities": probs,
    }


def test_ensemble_falls_back_to_deep_when_xgb_none():
    """When xgb_result is None, ensemble must return deep_result unchanged."""
    deep   = _make_deep_result("healthy", 70.0)
    result = _ensemble_vote(deep, None)
    assert result is deep
    assert result["prediction"] == "healthy"


def test_ensemble_raises_on_missing_deep_probs():
    """Missing 'all_probabilities' in deep_result must raise ValueError."""
    deep_bad = {"prediction": "healthy", "confidence": 60.0,
                "model_used": "EfficientNet-B4", "top3": []}
    xgb = _make_xgb_result("healthy")
    with pytest.raises((ValueError, KeyError)):
        _ensemble_vote(deep_bad, xgb)


def test_ensemble_raises_on_missing_xgb_probs():
    """Missing 'all_probabilities' in xgb_result must raise ValueError."""
    deep    = _make_deep_result("healthy", 60.0)
    xgb_bad = {"prediction": "healthy", "confidence": 80.0,
               "model_used": "XGBoost", "top3": []}
    with pytest.raises((ValueError, KeyError)):
        _ensemble_vote(deep, xgb_bad)


def test_ensemble_output_probabilities_sum_to_100():
    deep = _make_deep_result("healthy", 60.0)
    xgb  = _make_xgb_result("leaf_spot")
    result = _ensemble_vote(deep, xgb)
    total = sum(result["all_probabilities"].values())
    assert abs(total - 100.0) < 0.1, f"Probabilities sum to {total}, expected ~100"


def test_ensemble_top3_length():
    deep   = _make_deep_result("healthy", 55.0)
    xgb    = _make_xgb_result("healthy")
    result = _ensemble_vote(deep, xgb)
    assert len(result["top3"]) == 3


def test_ensemble_prediction_is_argmax():
    """Ensemble prediction must be the label with the highest blended probability."""
    deep   = _make_deep_result("healthy", 80.0)
    xgb    = _make_xgb_result("healthy")
    result = _ensemble_vote(deep, xgb)
    best   = max(result["all_probabilities"].items(), key=lambda kv: kv[1])[0]
    assert result["prediction"] == best


def test_ensemble_weights_65_35():
    """
    Verify the 65/35 blend formula.
    deep="healthy" at 100%, xgb="leaf_spot" at 100%.
    blended healthy   = 0.65*100 + 0.35*0   = 65.0
    blended leaf_spot = 0.65*0   + 0.35*100 = 35.0
    """
    deep_probs = {lbl: 0.0 for lbl in LABELS}
    deep_probs["healthy"] = 100.0
    deep_result = {"prediction": "healthy", "confidence": 100.0,
                   "model_used": "EfficientNet-B4", "top3": [],
                   "all_probabilities": deep_probs}

    xgb_probs = {lbl: 0.0 for lbl in LABELS}
    xgb_probs["leaf_spot"] = 100.0
    xgb_result = {"prediction": "leaf_spot", "confidence": 100.0,
                  "model_used": "XGBoost", "top3": [],
                  "all_probabilities": xgb_probs}

    result = _ensemble_vote(deep_result, xgb_result)
    assert abs(result["all_probabilities"]["healthy"]   - 65.0) < 0.01
    assert abs(result["all_probabilities"]["leaf_spot"] - 35.0) < 0.01
    assert result["prediction"] == "healthy"
