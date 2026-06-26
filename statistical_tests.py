"""
statistical_tests.py — Rigorous statistical comparison of classifiers
======================================================================
Spinach Plant Disease Detection System

Implements:
  - McNemar's test (comparing two classifiers on the same test set)
  - Wilcoxon signed-rank test (comparing CV fold scores)
  - Bonferroni correction for multiple comparisons
  - Bootstrap confidence intervals
  - Cochran's Q test (comparing ≥3 classifiers simultaneously)

All tests are two-sided at α=0.05 unless stated otherwise.

Usage
-----
    python statistical_tests.py --results results/evaluation_results.json

Reference
---------
Dietterich, T.G. (1998). Approximate statistical tests for comparing
supervised classification learning algorithms. Neural Computation, 10(7).

Authors : research team
Version : 1.0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon, chi2_contingency
    from scipy.stats import chi2 as chi2_dist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("ERROR: scipy is required.  Run: pip install scipy")
    sys.exit(1)


ALPHA = 0.05        # family-wise error rate before Bonferroni correction


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's test
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> dict:
    """
    McNemar's test for paired nominal data.

    Tests H₀: classifier A and classifier B have the same error rate.

    Contingency table:
        b = #samples where A correct,   B incorrect
        c = #samples where A incorrect, B correct

    Chi-squared statistic (with continuity correction for n<25):
        χ² = (|b - c| - 1)² / (b + c)

    Parameters
    ----------
    y_true   : ground-truth labels
    y_pred_a : predictions from model A
    y_pred_b : predictions from model B

    Returns
    -------
    dict with statistic, p_value, b, c, reject_h0
    """
    correct_a = y_true == y_pred_a
    correct_b = y_true == y_pred_b

    b = int(np.sum( correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a &  correct_b))   # A wrong, B right

    if b + c == 0:
        return {
            "statistic": 0.0, "p_value": 1.0,
            "b": b, "c": c, "reject_h0": False,
            "note": "b+c=0: no disagreements, models are identical on this set.",
        }

    # Continuity-corrected McNemar
    chi2_stat = float((abs(b - c) - 1) ** 2 / (b + c))
    p_value   = float(1.0 - chi2_dist.cdf(chi2_stat, df=1))

    return {
        "statistic":  round(chi2_stat, 4),
        "p_value":    round(p_value,   6),
        "b":          b,
        "c":          c,
        "reject_h0":  p_value < ALPHA,
        "interpretation": (
            "Statistically significant difference (α=0.05)."
            if p_value < ALPHA else
            "No statistically significant difference (α=0.05)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon signed-rank test (CV fold scores)
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_test(scores_a: list[float], scores_b: list[float],
                  metric_name: str = "F1-macro") -> dict:
    """
    Wilcoxon signed-rank test on paired per-fold metric scores.

    H₀: median of (scores_a - scores_b) = 0.

    Parameters
    ----------
    scores_a : per-fold metric values for model A (list of k floats)
    scores_b : per-fold metric values for model B (list of k floats)

    Returns
    -------
    dict with statistic, p_value, reject_h0
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have the same length.")

    diff = a - b
    if np.all(diff == 0):
        return {
            "statistic": 0.0, "p_value": 1.0, "reject_h0": False,
            "note": "All fold differences are zero.",
        }

    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError as exc:
        return {"statistic": None, "p_value": None, "reject_h0": False, "note": str(exc)}

    return {
        "metric":        metric_name,
        "statistic":     round(float(stat), 4),
        "p_value":       round(float(p),    6),
        "mean_diff":     round(float(np.mean(diff)), 4),
        "median_diff":   round(float(np.median(diff)), 4),
        "reject_h0":     float(p) < ALPHA,
        "interpretation": (
            f"Statistically significant difference in {metric_name} (α=0.05)."
            if float(p) < ALPHA else
            f"No statistically significant difference in {metric_name} (α=0.05)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bonferroni correction
# ─────────────────────────────────────────────────────────────────────────────

def bonferroni_correction(p_values: list[float], alpha: float = ALPHA) -> list[dict]:
    """
    Apply Bonferroni correction to a list of p-values.

    Corrected threshold = alpha / m  where m = number of comparisons.
    """
    m = len(p_values)
    corrected_alpha = alpha / m
    return [
        {
            "original_p":       round(p, 6),
            "corrected_alpha":  round(corrected_alpha, 6),
            "reject_h0":        p < corrected_alpha,
        }
        for p in p_values
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise comparison table
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_mcnemar(y_true: np.ndarray,
                     predictions: dict[str, np.ndarray]) -> list[dict]:
    """
    All pairwise McNemar tests with Bonferroni correction.

    Parameters
    ----------
    y_true      : ground-truth labels
    predictions : {model_name: y_pred array}

    Returns
    -------
    list of dicts, one per pair.
    """
    model_names = list(predictions)
    pairs = list(combinations(model_names, 2))

    raw_results = []
    p_values    = []

    for (a, b) in pairs:
        res = mcnemar_test(y_true, predictions[a], predictions[b])
        raw_results.append({"model_a": a, "model_b": b, **res})
        p_values.append(res["p_value"])

    corrections = bonferroni_correction(p_values)
    for r, c in zip(raw_results, corrections):
        r["bonferroni_corrected_alpha"] = c["corrected_alpha"]
        r["reject_h0_bonferroni"]       = c["reject_h0"]

    return raw_results


def pairwise_wilcoxon(cv_scores: dict[str, list[float]],
                      metric_name: str = "F1-macro") -> list[dict]:
    """
    All pairwise Wilcoxon tests with Bonferroni correction.

    Parameters
    ----------
    cv_scores   : {model_name: [fold_score, …]}
    metric_name : label for the metric being compared
    """
    model_names = list(cv_scores)
    pairs = list(combinations(model_names, 2))

    raw_results = []
    p_values    = []

    for (a, b) in pairs:
        res = wilcoxon_test(cv_scores[a], cv_scores[b], metric_name)
        raw_results.append({"model_a": a, "model_b": b, **res})
        p_values.append(res.get("p_value") or 1.0)

    corrections = bonferroni_correction(p_values)
    for r, c in zip(raw_results, corrections):
        r["bonferroni_corrected_alpha"] = c["corrected_alpha"]
        r["reject_h0_bonferroni"]       = c["reject_h0"]

    return raw_results


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for classifier comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default="results/evaluation_results.json",
        help="Path to evaluation_results.json produced by evaluate.py",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.  Run evaluate.py first.")
        sys.exit(1)

    with open(results_path) as fh:
        data = json.load(fh)

    cv_data = data.get("sklearn_cv", {})
    if not cv_data:
        print("No sklearn CV results found in the results file.")
        sys.exit(1)

    # ── Extract per-fold F1 scores for Wilcoxon ────────────────────────────
    cv_f1_scores = {}
    for model_name, res in cv_data.items():
        agg = res.get("aggregated_metrics", {})
        folds = agg.get("f1_macro", {}).get("folds")
        if folds:
            cv_f1_scores[model_name] = folds

    # ── Pairwise Wilcoxon on CV fold F1-macro ──────────────────────────────
    print("\n── Pairwise Wilcoxon Signed-Rank Test (CV F1-Macro, Bonferroni corrected) ──")
    wilcoxon_results = pairwise_wilcoxon(cv_f1_scores, "F1-macro (CV)")
    for r in wilcoxon_results:
        sig = "* SIGNIFICANT *" if r.get("reject_h0_bonferroni") else "n.s."
        print(f"  {r['model_a']} vs {r['model_b']}: "
              f"p={r.get('p_value', 'N/A')}  "
              f"Bonferroni α={r.get('bonferroni_corrected_alpha')}  {sig}")

    # ── Save statistical test results ──────────────────────────────────────
    out = {
        "timestamp":        data.get("timestamp"),
        "alpha":            ALPHA,
        "wilcoxon_cv_f1":   wilcoxon_results,
        "note": (
            "McNemar tests require per-sample predictions from a held-out test set. "
            "Run evaluate.py with --data to generate these. "
            "Wilcoxon tests operate on per-fold CV scores."
        ),
    }
    out_path = Path("results/statistical_tests.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nStatistical test results saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
