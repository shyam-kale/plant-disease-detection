"""
Unit tests for metric computation functions in evaluate.py.
All tests use known inputs with analytically derived expected outputs.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate import compute_metrics, bootstrap_ci


LABELS = [
    "healthy", "downy_mildew", "leaf_spot", "damping_off", "white_rust",
    "anthracnose", "mosaic_virus", "nutrient_deficiency", "pest_damage",
]


# ── Perfect prediction ────────────────────────────────────────────────────────

def test_perfect_accuracy():
    y = np.array(["healthy", "leaf_spot", "downy_mildew"])
    m = compute_metrics(y, y, None, LABELS)
    assert m["accuracy"] == 1.0, f"Expected 1.0, got {m['accuracy']}"


def test_perfect_f1_macro():
    """Perfect prediction across all 9 classes → macro F1 = 1.0."""
    # Use all 9 labels so every class has support
    y = np.array(LABELS)
    m = compute_metrics(y, y, None, LABELS)
    assert m["f1_macro"] == 1.0, f"Expected 1.0, got {m['f1_macro']}"


# ── Trivial classifier (all-same prediction) ─────────────────────────────────

def test_trivial_classifier_accuracy():
    """A classifier that always predicts 'healthy' on balanced 9-class data."""
    n = 9
    y_true = np.array(LABELS)                      # one of each
    y_pred = np.array(["healthy"] * n)
    m = compute_metrics(y_true, y_pred, None, LABELS)
    # Only 1 correct out of 9
    expected_acc = round(1 / 9, 4)
    assert abs(m["accuracy"] - expected_acc) < 1e-4, \
        f"Expected accuracy≈{expected_acc}, got {m['accuracy']}"


def test_trivial_classifier_f1_macro_zero_for_non_majority():
    """Macro F1 for a trivial classifier on balanced data should be low."""
    y_true = np.array(LABELS)
    y_pred = np.array(["healthy"] * len(LABELS))
    m = compute_metrics(y_true, y_pred, None, LABELS)
    # F1-macro must be < 0.3 for a trivial classifier on 9 balanced classes
    assert m["f1_macro"] < 0.3, \
        f"Trivial classifier should have low F1-macro, got {m['f1_macro']}"


# ── Known confusion matrix ────────────────────────────────────────────────────

def test_known_accuracy():
    """4 correct out of 5 → accuracy = 0.8."""
    y_true = np.array(["healthy", "leaf_spot", "downy_mildew", "white_rust", "anthracnose"])
    y_pred = np.array(["healthy", "leaf_spot", "downy_mildew", "white_rust", "healthy"])
    m = compute_metrics(y_true, y_pred, None, LABELS)
    assert abs(m["accuracy"] - 0.8) < 1e-4, f"Expected 0.8, got {m['accuracy']}"


# ── Probability-based metrics ─────────────────────────────────────────────────

def test_auc_perfect():
    """Perfect probability oracle should produce AUC = 1.0 for all classes."""
    from sklearn.preprocessing import label_binarize

    y_true  = np.array(LABELS)                                         # 9 samples
    y_proba = label_binarize(y_true, classes=LABELS).astype(float)    # (9, 9) identity
    # y_proba row i = 1.0 for class i and 0.0 elsewhere — perfect oracle
    m = compute_metrics(y_true, y_true, y_proba, LABELS)
    assert m["auc_macro_ovr"] == 1.0, f"Expected AUC=1.0, got {m['auc_macro_ovr']}"


def test_auc_none_when_proba_none():
    """AUC should be None when no probability estimates provided."""
    y = np.array(LABELS)
    m = compute_metrics(y, y, None, LABELS)
    assert m["auc_macro_ovr"] is None


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def test_bootstrap_ci_width_positive():
    """CI upper bound must be >= lower bound."""
    rng    = np.random.default_rng(0)
    y_true = rng.choice(LABELS, size=100)
    y_pred = rng.choice(LABELS, size=100)
    from sklearn.metrics import accuracy_score
    lo, hi = bootstrap_ci(y_true, y_pred, accuracy_score, n_boot=200)
    assert hi >= lo, f"CI bounds inverted: [{lo}, {hi}]"


def test_bootstrap_ci_perfect_predictor():
    """Perfect predictor: CI should be [1.0, 1.0]."""
    y = np.array(LABELS)
    from sklearn.metrics import accuracy_score
    lo, hi = bootstrap_ci(y, y, accuracy_score, n_boot=200)
    assert lo == 1.0 and hi == 1.0, f"Expected [1.0, 1.0], got [{lo}, {hi}]"


# ── F1 per class ──────────────────────────────────────────────────────────────

def test_f1_per_class_keys():
    """f1_per_class must contain exactly LABELS as keys."""
    y = np.array(LABELS)
    m = compute_metrics(y, y, None, LABELS)
    assert set(m["f1_per_class"].keys()) == set(LABELS)


def test_f1_per_class_perfect():
    """Perfect prediction → all per-class F1 = 1.0."""
    y = np.array(LABELS)
    m = compute_metrics(y, y, None, LABELS)
    for lbl, val in m["f1_per_class"].items():
        assert val == 1.0, f"F1 for {lbl} should be 1.0, got {val}"
