"""
Unit tests for statistical_tests.py.
All tests use analytically derived expected outcomes.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from statistical_tests import (
    mcnemar_test,
    wilcoxon_test,
    bonferroni_correction,
    pairwise_wilcoxon,
)


# ── McNemar ───────────────────────────────────────────────────────────────

def test_mcnemar_identical_models():
    """Identical predictions → p=1.0, do not reject H0."""
    y_true = np.array(["a", "b", "c", "a", "b"])
    y_pred = np.array(["a", "b", "c", "a", "b"])
    result = mcnemar_test(y_true, y_pred, y_pred)
    assert result["reject_h0"] is False


def test_mcnemar_completely_different_models():
    """One model perfect, other completely wrong → should be significant."""
    y_true = np.array(["a"] * 50 + ["b"] * 50)
    y_good = np.array(["a"] * 50 + ["b"] * 50)   # perfect
    y_bad  = np.array(["b"] * 50 + ["a"] * 50)   # always wrong
    result = mcnemar_test(y_true, y_good, y_bad)
    assert result["reject_h0"] is True
    assert result["p_value"] < 0.05


def test_mcnemar_b_c_values():
    """b = cases where A right B wrong; c = cases where A wrong B right."""
    y_true = np.array(["a", "a", "b", "b"])
    y_a    = np.array(["a", "a", "a", "b"])   # correct on 1,2,4; wrong on 3
    y_b    = np.array(["b", "a", "b", "b"])   # correct on 2,3,4; wrong on 1
    result = mcnemar_test(y_true, y_a, y_b)
    # A right B wrong: sample 1 (a==a, b!=a) → b=1
    # A wrong B right: sample 3 (a!=b, b==b) → c=1
    assert result["b"] == 1
    assert result["c"] == 1


def test_mcnemar_statistic_nonnegative():
    y_true = np.array(["x", "y", "x", "y", "x"])
    y_a    = np.array(["x", "y", "x", "x", "y"])
    y_b    = np.array(["x", "x", "y", "y", "x"])
    result = mcnemar_test(y_true, y_a, y_b)
    assert result["statistic"] >= 0.0


# ── Wilcoxon ──────────────────────────────────────────────────────────────

def test_wilcoxon_identical_scores():
    """Identical fold scores → p=1.0, do not reject H0."""
    scores = [0.85, 0.87, 0.83, 0.86, 0.84]
    result = wilcoxon_test(scores, scores)
    assert result["reject_h0"] is False


def test_wilcoxon_clearly_different():
    """
    Consistent large gap across 10 folds → should be significant.
    Using n=10 ensures the Wilcoxon test has sufficient power.
    """
    a = [0.95, 0.94, 0.96, 0.93, 0.97, 0.95, 0.94, 0.96, 0.93, 0.97]
    b = [0.48, 0.51, 0.47, 0.52, 0.49, 0.50, 0.46, 0.53, 0.48, 0.51]
    result = wilcoxon_test(a, b)
    assert result["reject_h0"] is True, \
        f"Expected significant result (p<0.05), got p={result.get('p_value')}"
    assert result["p_value"] < 0.05


def test_wilcoxon_mean_diff_sign():
    """mean_diff should be positive when a > b consistently."""
    a = [0.90, 0.91, 0.89, 0.90, 0.92]
    b = [0.70, 0.71, 0.69, 0.70, 0.72]
    result = wilcoxon_test(a, b)
    assert result["mean_diff"] > 0.0


def test_wilcoxon_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        wilcoxon_test([0.8, 0.9], [0.7, 0.6, 0.5])


# ── Bonferroni ────────────────────────────────────────────────────────────

def test_bonferroni_corrects_alpha():
    """With 5 comparisons, corrected α = 0.05/5 = 0.01."""
    p_vals = [0.001, 0.03, 0.04, 0.06, 0.10]
    results = bonferroni_correction(p_vals, alpha=0.05)
    for r in results:
        assert abs(r["corrected_alpha"] - 0.01) < 1e-9


def test_bonferroni_reject_below_corrected_alpha():
    p_vals = [0.005, 0.02]   # corrected α = 0.025
    results = bonferroni_correction(p_vals, alpha=0.05)
    assert results[0]["reject_h0"] is True    # 0.005 < 0.025
    assert results[1]["reject_h0"] is True    # 0.020 < 0.025


def test_bonferroni_do_not_reject_above():
    p_vals = [0.04, 0.08]   # corrected α = 0.025
    results = bonferroni_correction(p_vals, alpha=0.05)
    # 0.04 > 0.025 → do not reject
    assert results[0]["reject_h0"] is False


def test_bonferroni_single_comparison_no_change():
    """Single comparison → corrected α = original α."""
    results = bonferroni_correction([0.03], alpha=0.05)
    assert abs(results[0]["corrected_alpha"] - 0.05) < 1e-9


# ── Pairwise Wilcoxon ─────────────────────────────────────────────────────

def test_pairwise_wilcoxon_pair_count():
    """n models → n*(n-1)/2 pairs."""
    cv_scores = {
        "rf":  [0.90, 0.91, 0.89, 0.90, 0.92],
        "svm": [0.85, 0.86, 0.84, 0.85, 0.87],
        "knn": [0.80, 0.81, 0.79, 0.80, 0.82],
    }
    results = pairwise_wilcoxon(cv_scores)
    assert len(results) == 3   # C(3,2) = 3


def test_pairwise_wilcoxon_has_bonferroni_key():
    cv_scores = {
        "a": [0.9, 0.91, 0.89, 0.90, 0.92],
        "b": [0.7, 0.71, 0.69, 0.70, 0.72],
    }
    results = pairwise_wilcoxon(cv_scores)
    assert "bonferroni_corrected_alpha" in results[0]
    assert "reject_h0_bonferroni"       in results[0]
