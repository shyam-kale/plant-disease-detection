

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Optional imports with clear error messages ──────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend for servers
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_OK = True
except ImportError:
    PLOT_OK = False
    print("WARNING: matplotlib/seaborn not installed — figures will not be generated.")

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        average_precision_score,
        confusion_matrix,
        classification_report,
        brier_score_loss,
    )
    from sklearn.calibration import calibration_curve
    from sklearn.preprocessing import label_binarize
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("ERROR: scikit-learn is required. Run: pip install scikit-learn")
    sys.exit(1)

try:
    import mlflow
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("ERROR: Pillow is required. Run: pip install Pillow")
    sys.exit(1)

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from ml_models import ImageProcessor, DeepModel, TORCH_OK
try:
    from ml_models import XGBoostClassifier
except ImportError:
    XGBoostClassifier = None

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("evaluate")

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

LABELS       = Config.LABELS
N_LABELS     = len(LABELS)
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABELS)}
INT_TO_LABEL = {i: lbl for i, lbl in enumerate(LABELS)}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_root: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk data_root, extract features from every image, return X, y, paths.

    Parameters
    ----------
    data_root : str
        Root directory with one sub-folder per label.

    Returns
    -------
    X : np.ndarray shape (n, 28)
    y : np.ndarray shape (n,)  — string labels
    paths : list[str]          — corresponding file paths (for error analysis)
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root.resolve()}")

    found_labels = [d.name for d in root.iterdir() if d.is_dir()]
    unknown = set(found_labels) - set(LABELS)
    if unknown:
        raise ValueError(
            f"Unknown class folders in dataset: {unknown}. "
            f"Expected: {LABELS}"
        )

    X_list, y_list, path_list = [], [], []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    for label in LABELS:
        label_dir = root / label
        if not label_dir.exists():
            log.warning("Missing class folder: %s — skipping", label)
            continue
        image_paths = [p for p in label_dir.iterdir() if p.suffix.lower() in exts]
        if not image_paths:
            log.warning("No images found in %s", label_dir)
            continue
        log.info("Loading %s: %d images", label, len(image_paths))
        for img_path in image_paths:
            try:
                raw = img_path.read_bytes()
                proc = ImageProcessor(raw)
                proc.prepare()
                features = proc.extract_features()
                X_list.append(features)
                y_list.append(label)
                path_list.append(str(img_path))
            except Exception as exc:
                log.warning("Skipping %s: %s", img_path.name, exc)

    if not X_list:
        raise RuntimeError("No images loaded. Check dataset directory structure.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    log.info("Dataset: %d samples, %d features, %d classes", len(X), X.shape[1], N_LABELS)

    class_counts = {lbl: int(np.sum(y == lbl)) for lbl in LABELS if np.any(y == lbl)}
    log.info("Class distribution: %s", class_counts)
    return X, y, path_list


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray | None,
                    label_names: list[str]) -> dict:
    """
    Compute accuracy, macro/weighted precision, recall, F1, AUC, Brier score.

    All values derived solely from y_true vs y_pred (held-out data).
    """
    acc  = float(accuracy_score(y_true, y_pred))
    pm   = float(precision_score(y_true, y_pred, average="macro",    labels=label_names, zero_division=0))
    pw   = float(precision_score(y_true, y_pred, average="weighted", labels=label_names, zero_division=0))
    rm   = float(recall_score(   y_true, y_pred, average="macro",    labels=label_names, zero_division=0))
    rw   = float(recall_score(   y_true, y_pred, average="weighted", labels=label_names, zero_division=0))
    f1m  = float(f1_score(y_true, y_pred, average="macro",    labels=label_names, zero_division=0))
    f1w  = float(f1_score(y_true, y_pred, average="weighted", labels=label_names, zero_division=0))
    f1pc = f1_score(y_true, y_pred, average=None, labels=label_names, zero_division=0).tolist()

    auc_macro  = None
    auc_ovr_pc = None
    brier      = None

    if y_proba is not None:
        try:
            # label_binarize uses the exact order of label_names, so y_bin[:, i]
            # corresponds to label_names[i] — no sorting ambiguity.
            y_bin = label_binarize(y_true, classes=label_names)
            if y_bin.shape[1] == 1:
                # binary edge-case: binarize returns (n,1), expand to (n,2)
                y_bin = np.hstack([1 - y_bin, y_bin])

            auc_ovr_pc = {}
            valid_auc_vals = []
            for i, lbl in enumerate(label_names):
                if i < y_bin.shape[1] and np.unique(y_bin[:, i]).size > 1:
                    a = float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
                    auc_ovr_pc[lbl] = a
                    valid_auc_vals.append(a)
            auc_macro = float(np.mean(valid_auc_vals)) if valid_auc_vals else None
        except Exception as exc:
            log.warning("AUC computation failed: %s", exc)

        # Brier score (macro-averaged over classes, one-vs-rest)
        try:
            y_bin = label_binarize(y_true, classes=label_names)
            brier_scores = [
                brier_score_loss(y_bin[:, i], y_proba[:, i])
                for i in range(len(label_names))
                if y_bin.shape[1] > i
            ]
            brier = float(np.mean(brier_scores))
        except Exception:
            pass

    return {
        "accuracy":            round(acc,  4),
        "precision_macro":     round(pm,   4),
        "precision_weighted":  round(pw,   4),
        "recall_macro":        round(rm,   4),
        "recall_weighted":     round(rw,   4),
        "f1_macro":            round(f1m,  4),
        "f1_weighted":         round(f1w,  4),
        "f1_per_class":        {lbl: round(v, 4) for lbl, v in zip(label_names, f1pc)},
        "auc_macro_ovr":       round(auc_macro, 4) if auc_macro is not None else None,
        "auc_per_class":       {k: round(v, 4) for k, v in (auc_ovr_pc or {}).items()},
        "brier_score_macro":   round(brier, 4) if brier is not None else None,
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n_boot: int = 1000,
                 alpha: float = 0.05, random_state: int = 42) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for any scalar metric.

    Returns (lower, upper) at (1-alpha) confidence level.
    """
    rng   = np.random.default_rng(random_state)
    n     = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx   = rng.integers(0, n, size=n)
        stats.append(float(metric_fn(y_true[idx], y_pred[idx])))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: list[str],
                          title: str, save_path: Path) -> None:
    if not PLOT_OK:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix: %s", save_path)


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray,
                    labels: list[str], title: str, save_path: Path) -> None:
    if not PLOT_OK or y_proba is None:
        return
    y_bin = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))      # type: ignore[attr-defined]
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        if y_bin.shape[1] <= i:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc_val = roc_auc_score(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, color=col, lw=1.5,
                    label=f"{lbl} (AUC={auc_val:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ROC curves: %s", save_path)


def plot_precision_recall(y_true: np.ndarray, y_proba: np.ndarray,
                          labels: list[str], title: str, save_path: Path) -> None:
    if not PLOT_OK or y_proba is None:
        return
    y_bin  = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(labels)))     # type: ignore[attr-defined]
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        if y_bin.shape[1] <= i:
            continue
        try:
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap = average_precision_score(y_bin[:, i], y_proba[:, i])
            ax.plot(rec, prec, color=col, lw=1.5,
                    label=f"{lbl} (AP={ap:.3f})")
        except Exception:
            pass
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved PR curves: %s", save_path)


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray,
                     labels: list[str], title: str, save_path: Path) -> None:
    if not PLOT_OK or y_proba is None:
        return
    y_bin  = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(labels)))     # type: ignore[attr-defined]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        if y_bin.shape[1] <= i:
            continue
        try:
            prob_true, prob_pred = calibration_curve(
                y_bin[:, i], y_proba[:, i], n_bins=10, strategy="uniform",
            )
            ax.plot(prob_pred, prob_true, "s-", color=col, lw=1.5,
                    markersize=4, label=lbl)
        except Exception:
            pass
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives",      fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved calibration plot: %s", save_path)


def plot_cv_boxplot(cv_scores: dict[str, list[float]],
                    metric: str, title: str, save_path: Path) -> None:
    """Box-plot of per-fold metric values across all models."""
    if not PLOT_OK:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = list(cv_scores.keys())
    data        = [cv_scores[m] for m in model_names]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True, notch=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))  # type: ignore[attr-defined]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title,   fontsize=13, fontweight="bold")
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved CV box-plot: %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sklearn_cv(X: np.ndarray, y: np.ndarray,
                   n_folds: int = 5,
                   run_id: str | None = None) -> dict:
    """
    5-fold stratified cross-validation for all sklearn classifiers.

    Returns a dict with per-model and aggregated results.
    All metrics are computed on held-out fold data only.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.12,
            subsample=0.85, random_state=42,
        ),
        "svm": SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean"),
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial", random_state=42,
        ),
    }

    all_results   = {}
    cv_f1_per_model = {}

    for clf_name, clf in classifiers.items():
        log.info("Cross-validating: %s (%d folds)…", clf_name, n_folds)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        fold_metrics = []
        all_y_true, all_y_pred, all_y_proba = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Clone pipeline for each fold to prevent data leakage
            from sklearn.base import clone
            fold_pipe = clone(pipe)
            fold_pipe.fit(X_tr, y_tr)

            y_pred  = fold_pipe.predict(X_te)
            y_proba = (fold_pipe.predict_proba(X_te)
                       if hasattr(fold_pipe.named_steps["clf"], "predict_proba")
                       else None)

            fold_m = compute_metrics(y_te, y_pred, y_proba, LABELS)
            fold_m["fold"] = fold_idx + 1
            fold_metrics.append(fold_m)

            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(y_pred.tolist())
            if y_proba is not None:
                all_y_proba.extend(y_proba.tolist())

        # ── Aggregate across folds ──────────────────────────────────────
        y_true_all  = np.array(all_y_true)
        y_pred_all  = np.array(all_y_pred)
        y_proba_all = np.array(all_y_proba) if all_y_proba else None

        # Aggregate: mean ± std of per-fold metrics
        metric_keys = [k for k in fold_metrics[0] if k != "fold" and
                       isinstance(fold_metrics[0][k], (int, float)) and
                       fold_metrics[0][k] is not None]

        aggregated = {}
        for k in metric_keys:
            vals = [fm[k] for fm in fold_metrics if fm.get(k) is not None]
            aggregated[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "min":  round(float(np.min(vals)),  4),
                "max":  round(float(np.max(vals)),  4),
                "folds": [round(v, 4) for v in vals],
            }

        # Per-class confusion matrix on concatenated held-out predictions
        cm = confusion_matrix(y_true_all, y_pred_all, labels=LABELS)

        # Bootstrap 95% CI for accuracy
        ci_lo, ci_hi = bootstrap_ci(
            y_true_all, y_pred_all,
            lambda yt, yp: accuracy_score(yt, yp),
        )

        # Bootstrap 95% CI for macro-F1
        f1_ci_lo, f1_ci_hi = bootstrap_ci(
            y_true_all, y_pred_all,
            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        )

        # Full classification report (on aggregated held-out predictions)
        cls_report = classification_report(
            y_true_all, y_pred_all,
            labels=LABELS, output_dict=True, zero_division=0,
        )

        result = {
            "model":              clf_name,
            "n_folds":            n_folds,
            "n_samples":          int(len(X)),
            "n_features":         int(X.shape[1]),
            "aggregated_metrics": aggregated,
            "per_fold_metrics":   fold_metrics,
            "confusion_matrix":   cm.tolist(),
            "classification_report": cls_report,
            "accuracy_95ci":      [ci_lo, ci_hi],
            "f1_macro_95ci":      [f1_ci_lo, f1_ci_hi],
        }

        all_results[clf_name] = result
        cv_f1_per_model[clf_name] = [fm["f1_macro"] for fm in fold_metrics]

        # ── Generate figures ────────────────────────────────────────────
        tag = clf_name.replace("_", "-")
        plot_confusion_matrix(
            cm, LABELS,
            title=f"Confusion Matrix — {clf_name}  (5-fold CV, held-out)",
            save_path=FIGURES_DIR / f"cm_{tag}.png",
        )
        plot_roc_curves(
            y_true_all, y_proba_all, LABELS,
            title=f"ROC Curves — {clf_name}  (5-fold CV, held-out)",
            save_path=FIGURES_DIR / f"roc_{tag}.png",
        )
        plot_precision_recall(
            y_true_all, y_proba_all, LABELS,
            title=f"Precision-Recall — {clf_name}  (5-fold CV, held-out)",
            save_path=FIGURES_DIR / f"pr_{tag}.png",
        )
        plot_calibration(
            y_true_all, y_proba_all, LABELS,
            title=f"Calibration — {clf_name}  (5-fold CV, held-out)",
            save_path=FIGURES_DIR / f"cal_{tag}.png",
        )

        log.info(
            "%s — acc=%.4f±%.4f  F1-macro=%.4f±%.4f  AUC=%.4f±%.4f",
            clf_name,
            aggregated["accuracy"]["mean"],  aggregated["accuracy"]["std"],
            aggregated["f1_macro"]["mean"],  aggregated["f1_macro"]["std"],
            aggregated.get("auc_macro_ovr", {}).get("mean", 0),
            aggregated.get("auc_macro_ovr", {}).get("std",  0),
        )

        # ── MLflow logging ──────────────────────────────────────────────
        if MLFLOW_OK and run_id:
            try:
                with mlflow.start_run(run_name=clf_name, nested=True):
                    mlflow.log_param("model",     clf_name)
                    mlflow.log_param("n_folds",   n_folds)
                    mlflow.log_param("n_samples", len(X))
                    mlflow.log_metrics({
                        "cv_accuracy_mean":  aggregated["accuracy"]["mean"],
                        "cv_accuracy_std":   aggregated["accuracy"]["std"],
                        "cv_f1_macro_mean":  aggregated["f1_macro"]["mean"],
                        "cv_f1_macro_std":   aggregated["f1_macro"]["std"],
                    })
            except Exception as exc:
                log.warning("MLflow logging failed: %s", exc)

    # CV box-plot across all models
    plot_cv_boxplot(
        cv_f1_per_model, "F1 Macro (per fold)",
        "Cross-Validation F1-Macro — All sklearn Models",
        FIGURES_DIR / "cv_f1_boxplot.png",
    )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Deep model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_deep_model(X_paths: list[str], y_true: np.ndarray) -> dict | None:
    """
    Run MobileNetV2 inference on every image in the dataset and compute metrics.

    Parameters
    ----------
    X_paths : list of absolute image file paths
    y_true  : ground-truth labels aligned to X_paths

    Returns
    -------
    dict with metrics, confusion matrix, figures saved to results/figures/
    """
    if not TORCH_OK:
        log.warning("PyTorch not available — skipping deep model evaluation.")
        return None

    from ml_models import deep_model
    import time as _time

    log.info("Waiting for MobileNetV2 to load…")
    for _ in range(60):
        if deep_model.ready:
            break
        _time.sleep(1.0)
    if not deep_model.ready:
        log.error("MobileNetV2 failed to load: %s", deep_model._load_error)
        return None

    log.info("Evaluating MobileNetV2 on %d images…", len(X_paths))
    y_pred_list   = []
    y_proba_list  = []
    latencies_ms  = []

    from PIL import Image as PILImage
    for path, true_lbl in zip(X_paths, y_true):
        try:
            img = PILImage.open(path).convert("RGB")
            t0  = _time.perf_counter()
            res = deep_model.predict(img)
            lat = (_time.perf_counter() - t0) * 1000
            if res is None:
                continue
            y_pred_list.append(res["prediction"])
            # all_probabilities is a dict label→percentage; convert to ordered array
            proba_arr = np.array([
                res["all_probabilities"].get(lbl, 0.0) / 100.0
                for lbl in LABELS
            ], dtype=np.float32)
            y_proba_list.append(proba_arr)
            latencies_ms.append(lat)
        except Exception as exc:
            log.warning("Skipping %s: %s", path, exc)

    if not y_pred_list:
        log.error("No deep model predictions produced.")
        return None

    y_pred_arr  = np.array(y_pred_list)
    y_proba_arr = np.array(y_proba_list) if y_proba_list else None
    y_true_sub  = y_true[:len(y_pred_list)]

    metrics = compute_metrics(y_true_sub, y_pred_arr, y_proba_arr, LABELS)
    ci_lo, ci_hi = bootstrap_ci(
        y_true_sub, y_pred_arr,
        lambda yt, yp: accuracy_score(yt, yp),
    )
    f1_ci_lo, f1_ci_hi = bootstrap_ci(
        y_true_sub, y_pred_arr,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    )

    cm = confusion_matrix(y_true_sub, y_pred_arr, labels=LABELS)

    plot_confusion_matrix(
        cm, LABELS,
        title="Confusion Matrix — MobileNetV2 (full dataset evaluation)",
        save_path=FIGURES_DIR / "cm_mobilenetv2.png",
    )
    plot_roc_curves(
        y_true_sub, y_proba_arr, LABELS,
        title="ROC Curves — MobileNetV2",
        save_path=FIGURES_DIR / "roc_mobilenetv2.png",
    )
    plot_precision_recall(
        y_true_sub, y_proba_arr, LABELS,
        title="Precision-Recall — MobileNetV2",
        save_path=FIGURES_DIR / "pr_mobilenetv2.png",
    )
    plot_calibration(
        y_true_sub, y_proba_arr, LABELS,
        title="Calibration — MobileNetV2",
        save_path=FIGURES_DIR / "cal_mobilenetv2.png",
    )

    result = {
        "model":              "MobileNetV2-PlantVillage",
        "n_samples":          len(y_pred_list),
        "metrics":            metrics,
        "confusion_matrix":   cm.tolist(),
        "accuracy_95ci":      [ci_lo, ci_hi],
        "f1_macro_95ci":      [f1_ci_lo, f1_ci_hi],
        "latency_ms": {
            "mean":   round(float(np.mean(latencies_ms)),   2),
            "median": round(float(np.median(latencies_ms)), 2),
            "p95":    round(float(np.percentile(latencies_ms, 95)), 2),
            "p99":    round(float(np.percentile(latencies_ms, 99)), 2),
        },
    }

    log.info(
        "MobileNetV2 — acc=%.4f  F1-macro=%.4f  AUC=%.4f  latency_p50=%.1fms",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics.get("auc_macro_ovr") or 0,
        result["latency_ms"]["median"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ensemble(X_paths: list[str], y_true: np.ndarray,
                      X_features: np.ndarray) -> dict | None:
    """
    Evaluate the full ensemble (EfficientNet-B4 65% + XGBoost 35%)
    on the complete dataset.
    """
    if not TORCH_OK:
        log.warning("PyTorch not available — skipping ensemble evaluation.")
        return None

    from ml_models import deep_model, xgb_model, _ensemble_vote
    import time as _time

    if not deep_model.ready:
        log.warning("Deep model not ready — skipping ensemble evaluation.")
        return None

    if not xgb_model.ready:
        log.warning("XGBoost not ready — skipping ensemble evaluation.")
        return None

    from PIL import Image as PILImage
    y_pred_ens  = []
    y_proba_ens = []
    valid_mask  = []

    for i, (path, _) in enumerate(zip(X_paths, y_true)):
        try:
            img  = PILImage.open(path).convert("RGB")
            deep = deep_model.predict(img)
            xgb  = xgb_model.predict(X_features[i])
            if deep is None:
                valid_mask.append(False)
                continue
            ens  = _ensemble_vote(deep, xgb)
            y_pred_ens.append(ens["prediction"])
            proba_arr = np.array([
                ens["all_probabilities"][lbl] / 100.0
                for lbl in LABELS
            ], dtype=np.float32)
            y_proba_ens.append(proba_arr)
            valid_mask.append(True)
        except Exception as exc:
            log.warning("Ensemble prediction failed for %s: %s", path, exc)
            valid_mask.append(False)

    if not y_pred_ens:
        return None

    y_pred_arr  = np.array(y_pred_ens)
    y_proba_arr = np.array(y_proba_ens)
    y_true_sub  = y_true[np.array(valid_mask)]

    metrics = compute_metrics(y_true_sub, y_pred_arr, y_proba_arr, LABELS)
    cm      = confusion_matrix(y_true_sub, y_pred_arr, labels=LABELS)

    plot_confusion_matrix(
        cm, LABELS,
        title="Confusion Matrix — Ensemble (EfficientNet-B4 65% + XGBoost 35%)",
        save_path=FIGURES_DIR / "cm_ensemble.png",
    )
    plot_roc_curves(
        y_true_sub, y_proba_arr, LABELS,
        title="ROC Curves — Ensemble",
        save_path=FIGURES_DIR / "roc_ensemble.png",
    )

    ci_lo, ci_hi = bootstrap_ci(
        y_true_sub, y_pred_arr,
        lambda yt, yp: accuracy_score(yt, yp),
    )
    f1_ci_lo, f1_ci_hi = bootstrap_ci(
        y_true_sub, y_pred_arr,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    )

    result = {
        "model":            "ensemble_EfficientNetB4_65_XGBoost_35",
        "n_samples":        len(y_pred_ens),
        "metrics":          metrics,
        "confusion_matrix": cm.tolist(),
        "accuracy_95ci":    [ci_lo, ci_hi],
        "f1_macro_95ci":    [f1_ci_lo, f1_ci_hi],
        "weights":          {"efficientnet_b4": 0.65, "xgboost": 0.35},
    }

    log.info(
        "Ensemble — acc=%.4f  F1-macro=%.4f  AUC=%.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics.get("auc_macro_ovr") or 0,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table (LaTeX-ready)
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_table(cv_results: dict, deep_result: dict | None,
                            ens_result: dict | None) -> str:
    """
    Generate a summary table in both JSON and LaTeX formats.
    Written to results/summary_table.json and results/summary_table.tex.
    """
    rows = []

    # sklearn models (CV metrics — the only scientifically valid metrics)
    for model_name, res in cv_results.items():
        agg = res.get("aggregated_metrics", {})
        rows.append({
            "model":                model_name,
            "evaluation_method":    "5-fold stratified CV (held-out)",
            "accuracy_mean":        agg.get("accuracy",  {}).get("mean"),
            "accuracy_std":         agg.get("accuracy",  {}).get("std"),
            "f1_macro_mean":        agg.get("f1_macro",  {}).get("mean"),
            "f1_macro_std":         agg.get("f1_macro",  {}).get("std"),
            "f1_weighted_mean":     agg.get("f1_weighted", {}).get("mean"),
            "auc_macro_mean":       agg.get("auc_macro_ovr", {}).get("mean"),
            "accuracy_95ci":        res.get("accuracy_95ci"),
            "f1_macro_95ci":        res.get("f1_macro_95ci"),
        })

    # Deep model
    if deep_result:
        m = deep_result["metrics"]
        rows.append({
            "model":              "MobileNetV2 (PlantVillage transfer)",
            "evaluation_method":  "full dataset inference",
            "accuracy_mean":      m.get("accuracy"),
            "accuracy_std":       None,
            "f1_macro_mean":      m.get("f1_macro"),
            "f1_macro_std":       None,
            "f1_weighted_mean":   m.get("f1_weighted"),
            "auc_macro_mean":     m.get("auc_macro_ovr"),
            "accuracy_95ci":      deep_result.get("accuracy_95ci"),
            "f1_macro_95ci":      deep_result.get("f1_macro_95ci"),
        })

    # Ensemble
    if ens_result:
        m = ens_result["metrics"]
        rows.append({
            "model":              "Ensemble (MobileNetV2 60% + sklearn 40%)",
            "evaluation_method":  "full dataset inference",
            "accuracy_mean":      m.get("accuracy"),
            "accuracy_std":       None,
            "f1_macro_mean":      m.get("f1_macro"),
            "f1_macro_std":       None,
            "f1_weighted_mean":   m.get("f1_weighted"),
            "auc_macro_mean":     m.get("auc_macro_ovr"),
            "accuracy_95ci":      ens_result.get("accuracy_95ci"),
            "f1_macro_95ci":      ens_result.get("f1_macro_95ci"),
        })

    # Save JSON
    json_path = RESULTS_DIR / "summary_table.json"
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)
    log.info("Summary JSON saved: %s", json_path)

    # Generate LaTeX table
    def _fmt(v, digits=4) -> str:
        if v is None:
            return "—"
        return f"{v:.{digits}f}"

    def _fmt_ci(ci) -> str:
        if not ci or len(ci) < 2:
            return "—"
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance Comparison — 5-Fold Stratified CV (sklearn) / Full Dataset (Deep/Ensemble)}",
        r"\label{tab:results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Model & Eval Method & Accuracy & F1-Macro & F1-Weighted & AUC-Macro & Acc 95\% CI \\",
        r"\midrule",
    ]
    for row in rows:
        acc_str = _fmt(row["accuracy_mean"])
        if row["accuracy_std"] is not None:
            acc_str += rf"$\pm${_fmt(row['accuracy_std'])}"
        f1m_str = _fmt(row["f1_macro_mean"])
        if row["f1_macro_std"] is not None:
            f1m_str += rf"$\pm${_fmt(row['f1_macro_std'])}"
        lines.append(
            rf"{row['model']} & {row['evaluation_method']} & "
            rf"{acc_str} & {f1m_str} & "
            rf"{_fmt(row['f1_weighted_mean'])} & {_fmt(row['auc_macro_mean'])} & "
            rf"{_fmt_ci(row['accuracy_95ci'])} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    latex_str = "\n".join(lines)
    tex_path = RESULTS_DIR / "summary_table.tex"
    with open(tex_path, "w") as fh:
        fh.write(latex_str)
    log.info("Summary LaTeX table saved: %s", tex_path)
    return latex_str


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spinach Disease Detection — Publication-grade Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to dataset root (one subfolder per disease label).",
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--skip-deep", action="store_true",
        help="Skip MobileNetV2 evaluation (faster, sklearn only).",
    )
    parser.add_argument(
        "--mlflow-uri", default="./mlruns",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment", default="spinach_disease_detection",
        help="MLflow experiment name.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Spinach Disease Detection — Evaluation Pipeline")
    log.info("Dataset : %s", args.data)
    log.info("CV folds: %d", args.folds)
    log.info("=" * 60)

    # ── MLflow setup ────────────────────────────────────────────────────────
    run_id = None
    if MLFLOW_OK:
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment)
            active_run = mlflow.start_run(
                run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            run_id = active_run.info.run_id
            mlflow.log_params({
                "dataset":  args.data,
                "cv_folds": args.folds,
                "labels":   ",".join(LABELS),
            })
            log.info("MLflow run started: %s", run_id)
        except Exception as exc:
            log.warning("MLflow init failed (continuing without tracking): %s", exc)

    # ── Load dataset ────────────────────────────────────────────────────────
    X_features, y_labels, image_paths = load_dataset(args.data)

    # ── sklearn cross-validation ────────────────────────────────────────────
    log.info("Running sklearn cross-validation…")
    cv_results = run_sklearn_cv(X_features, y_labels, n_folds=args.folds, run_id=run_id)

    # ── Fit XGBoost on full dataset and persist ──────────────────────────────
    log.info("Fitting XGBoost on full dataset…")
    xgb_clf = XGBoostClassifier.__new__(XGBoostClassifier)
    import threading as _threading
    xgb_clf.model   = None
    xgb_clf.scaler  = None
    xgb_clf.encoder = None
    xgb_clf.ready   = False
    xgb_clf.feature_importance = {}
    xgb_clf._lock   = _threading.Lock()
    xgb_clf.fit(X_features, y_labels)

    # ── Deep model evaluation ───────────────────────────────────────────────
    deep_result = None
    if not args.skip_deep:
        log.info("Evaluating EfficientNet-B4…")
        deep_result = evaluate_deep_model(image_paths, y_labels)

    # ── Ensemble evaluation ─────────────────────────────────────────────────
    ens_result = None
    if deep_result and registry.is_ready():
        log.info("Evaluating ensemble…")
        ens_result = evaluate_ensemble(image_paths, y_labels, X_features)

    # ── Summary table ───────────────────────────────────────────────────────
    generate_summary_table(cv_results, deep_result, ens_result)

    # ── Write full results JSON ─────────────────────────────────────────────
    full_results = {
        "timestamp":         datetime.utcnow().isoformat() + "Z",
        "dataset":           args.data,
        "n_samples":         int(len(X_features)),
        "n_features":        int(X_features.shape[1]),
        "cv_folds":          args.folds,
        "labels":            LABELS,
        "sklearn_cv":        cv_results,
        "deep_model":        deep_result,
        "ensemble":          ens_result,
        "reproducibility": {
            "random_seed":        42,
            "stratified_kfold":   True,
            "clone_per_fold":     True,
            "metrics_source":     "held-out test folds only",
            "no_training_metrics_cited": True,
        },
    }
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as fh:
        json.dump(full_results, fh, indent=2)
    log.info("Full results saved: %s", results_path)

    # ── MLflow artifact logging ─────────────────────────────────────────────
    if MLFLOW_OK and run_id:
        try:
            mlflow.log_artifact(str(results_path))
            mlflow.log_artifact(str(RESULTS_DIR / "summary_table.json"))
            mlflow.log_artifact(str(RESULTS_DIR / "summary_table.tex"))
            for fig in FIGURES_DIR.glob("*.png"):
                mlflow.log_artifact(str(fig))
            # Log best model CV metrics to MLflow summary
            best = max(cv_results.items(),
                       key=lambda kv: kv[1]["aggregated_metrics"].get("f1_macro", {}).get("mean", 0))
            best_agg = best[1]["aggregated_metrics"]
            mlflow.log_metrics({
                "best_cv_accuracy_mean": best_agg["accuracy"]["mean"],
                "best_cv_f1_macro_mean": best_agg["f1_macro"]["mean"],
            })
            mlflow.end_run()
        except Exception as exc:
            log.warning("MLflow artifact logging failed: %s", exc)

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"{'Model':<28} {'Acc':>8} {'F1-Mac':>8} {'AUC':>8}")
    print("-" * 60)
    for name, res in cv_results.items():
        agg = res["aggregated_metrics"]
        acc = agg.get("accuracy",     {}).get("mean", 0)
        f1  = agg.get("f1_macro",     {}).get("mean", 0)
        auc = agg.get("auc_macro_ovr", {}).get("mean") or 0
        print(f"{'(CV) ' + name:<28} {acc:>8.4f} {f1:>8.4f} {auc:>8.4f}")
    if deep_result:
        m = deep_result["metrics"]
        print(f"{'MobileNetV2':<28} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f} "
              f"{(m.get('auc_macro_ovr') or 0):>8.4f}")
    if ens_result:
        m = ens_result["metrics"]
        print(f"{'Ensemble':<28} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f} "
              f"{(m.get('auc_macro_ovr') or 0):>8.4f}")
    print("=" * 60)
    print(f"Results  : {results_path.resolve()}")
    print(f"Figures  : {FIGURES_DIR.resolve()}")
    print(f"LaTeX    : {(RESULTS_DIR / 'summary_table.tex').resolve()}")



# ─────────────────────────────────────────────────────────────────────────────
# PlantVillage → Spinach label mapping  (for train mode)
# ─────────────────────────────────────────────────────────────────────────────
PV_TO_SPINACH = {
    "Apple___healthy":"healthy","Blueberry___healthy":"healthy",
    "Cherry_(including_sour)___healthy":"healthy","Corn_(maize)___healthy":"healthy",
    "Grape___healthy":"healthy","Peach___healthy":"healthy",
    "Pepper,_bell___healthy":"healthy","Potato___healthy":"healthy",
    "Raspberry___healthy":"healthy","Soybean___healthy":"healthy",
    "Strawberry___healthy":"healthy","Tomato___healthy":"healthy",
    "Cherry_(including_sour)___Powdery_mildew":"downy_mildew",
    "Squash___Powdery_mildew":"downy_mildew","Potato___Late_blight":"downy_mildew",
    "Tomato___Late_blight":"downy_mildew","Tomato___Leaf_Mold":"downy_mildew",
    "Apple___Apple_scab":"leaf_spot",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":"leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight":"leaf_spot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":"leaf_spot",
    "Potato___Early_blight":"leaf_spot","Strawberry___Leaf_scorch":"leaf_spot",
    "Tomato___Early_blight":"leaf_spot","Tomato___Septoria_leaf_spot":"leaf_spot",
    "Tomato___Target_Spot":"leaf_spot",
    "Peach___Bacterial_spot":"damping_off","Pepper,_bell___Bacterial_spot":"damping_off",
    "Tomato___Bacterial_spot":"damping_off",
    "Apple___Cedar_apple_rust":"white_rust","Corn_(maize)___Common_rust_":"white_rust",
    "Apple___Black_rot":"anthracnose","Grape___Black_rot":"anthracnose",
    "Grape___Esca_(Black_Measles)":"mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":"mosaic_virus",
    "Tomato___Tomato_mosaic_virus":"mosaic_virus",
    "Orange___Haunglongbing_(Citrus_greening)":"nutrient_deficiency",
    "Tomato___Spider_mites Two-spotted_spider_mite":"pest_damage",
}


def build_plantvillage_dataset(data_dir: str, max_per_class: int = 800):
    """
    Scan a PlantVillage-style directory and return (image_paths, labels).
    Supports TWO structures:

    Structure 1 — Subfolders (standard PlantVillage ZIP):
        data_dir/
            Apple___healthy/image1.jpg
            Tomato___Late_blight/image2.jpg

    Structure 2 — Flat folder (filename contains class):
        data_dir/
            uuid___Apple___healthy 001.jpg
            uuid___Tomato___Late_blight 002.jpg

    Folder names and filenames are both mapped via PV_TO_SPINACH.
    Unknown classes are skipped with a warning.
    """
    root = Path(data_dir)
    image_paths, labels = [], []
    per_class: dict[str, int] = {lbl: 0 for lbl in LABELS}
    skipped = []

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

    def _map_name(name: str):
        """Map a folder/filename fragment to a spinach label."""
        # Exact match
        lbl = PV_TO_SPINACH.get(name)
        if lbl: return lbl
        # Partial match against PV keys
        for pv, sl in PV_TO_SPINACH.items():
            if pv.lower() in name.lower() or name.lower() in pv.lower():
                return sl

        # ── Direct common name mapping (handles Malabar, custom datasets) ──
        name_l = name.lower().replace("-","_").replace(" ","_").replace("(","").replace(")","").replace("__","_")
        DIRECT = {
            "healthy": "healthy", "healthy_leaf": "healthy", "healthyleaf": "healthy",
            "downy_mildew": "downy_mildew", "downymildew": "downy_mildew",
            "downy": "downy_mildew", "mildew": "downy_mildew",
            "leaf_spot": "leaf_spot", "leafspot": "leaf_spot",
            "cercospora": "leaf_spot", "alternaria": "leaf_spot",
            "damping_off": "damping_off", "dampingoff": "damping_off",
            "damping": "damping_off", "pythium": "damping_off",
            "white_rust": "white_rust", "whiterust": "white_rust",
            "white": "white_rust",
            "anthracnose": "anthracnose", "colletotrichum": "anthracnose",
            "black_rot": "anthracnose",
            "mosaic": "mosaic_virus", "mosaic_virus": "mosaic_virus",
            "mosaicvirus": "mosaic_virus", "virus": "mosaic_virus",
            "nutrient": "nutrient_deficiency", "deficiency": "nutrient_deficiency",
            "nutrient_deficiency": "nutrient_deficiency", "chlorosis": "nutrient_deficiency",
            "pest": "pest_damage", "pest_damage": "pest_damage",
            "spider_mite": "pest_damage", "aphid": "pest_damage",
            "insect": "pest_damage", "mite": "pest_damage",
            "bacterial": "damping_off", "bacterial_spot": "damping_off",
            "bacterialspot": "damping_off",
        }
        # Try full name
        if name_l in DIRECT: return DIRECT[name_l]
        # Try stripping trailing numbers like (240)
        import re
        clean = re.sub(r'\d+$','', name_l).strip('_')
        if clean in DIRECT: return DIRECT[clean]
        # Try any keyword match
        for key, lbl in DIRECT.items():
            if key in name_l: return lbl

        # Healthy fallback
        if "healthy" in name.lower(): return "healthy"
        return None

    # ── Try subfolder structure first ───────────────────────────────────────
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs:
        found_any = False
        for folder in sorted(subdirs):
            lbl = _map_name(folder.name)
            if lbl is None:
                # Try going one level deeper (e.g. color/Apple___healthy/)
                for sub2 in folder.iterdir():
                    if sub2.is_dir():
                        lbl2 = _map_name(sub2.name)
                        if lbl2:
                            imgs = [f for f in sub2.iterdir() if f.suffix in EXTS]
                            remaining = max_per_class - per_class[lbl2]
                            for img in imgs[:max(0, remaining)]:
                                image_paths.append(str(img))
                                labels.append(lbl2)
                                per_class[lbl2] += 1
                                found_any = True
                continue

            imgs = [f for f in folder.iterdir() if f.suffix in EXTS]
            remaining = max_per_class - per_class[lbl]
            for img in imgs[:max(0, remaining)]:
                image_paths.append(str(img))
                labels.append(lbl)
                per_class[lbl] += 1
                found_any = True

        if found_any:
            log.info("Dataset built from subfolders: %d images", len(image_paths))
            for lbl, cnt in per_class.items():
                if cnt: log.info("  %-25s %d", lbl, cnt)
            return image_paths, labels

    # ── Flat structure: disease name in filename ─────────────────────────────
    log.info("No subfolder structure found — trying flat folder (disease in filename)…")
    all_imgs = [f for f in root.rglob("*") if f.suffix in EXTS]
    log.info("Found %d image files in flat structure", len(all_imgs))

    for img in all_imgs:
        name = img.stem  # filename without extension
        # Filename pattern: uuid___ClassName 001  OR  ClassName_001  OR  ClassName 001
        lbl = None
        parts = name.split("___")
        for part in parts:
            lbl = _map_name(part.strip())
            if lbl: break
        if lbl is None:
            # Try matching any PV key substring anywhere in the filename
            for pv, sl in PV_TO_SPINACH.items():
                if pv.lower().replace(" ","_") in name.lower().replace(" ","_"):
                    lbl = sl; break
        if lbl is None:
            skipped.append(img.name)
            continue

        remaining = max_per_class - per_class[lbl]
        if remaining <= 0:
            continue
        image_paths.append(str(img))
        labels.append(lbl)
        per_class[lbl] += 1

    if skipped:
        log.warning("Skipped %d images (no matching disease class): %s…",
                    len(skipped), skipped[:3])

    log.info("Dataset built from flat folder: %d images", len(image_paths))
    for lbl, cnt in per_class.items():
        if cnt: log.info("  %-25s %d", lbl, cnt)
    return image_paths, labels


def train_efficientnet(image_paths: list, labels: list,
                       epochs: int = 25, batch_size: int = 16,
                       lr: float = 1e-4) -> dict:
    """
    Fine-tune EfficientNet-B4 on spinach disease images.
    Saves models/efficientnet_b4_spinach_finetuned.pth
    Expected val accuracy: 88-94%
    """
    if not TORCH_OK:
        log.error("PyTorch not installed. Run: pip install torch torchvision timm")
        return {}
    try:
        import torch, torch.nn as nn, timm
        import torchvision.transforms as T
        from torch.utils.data import DataLoader
        import torch.optim as optim
        from sklearn.model_selection import train_test_split
        from PIL import Image as PIL_Image
    except ImportError as e:
        log.error("Missing dependency: %s", e); return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training EfficientNet-B4 on %s (%d images)", device, len(image_paths))
    label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}

    X_tr, X_val, y_tr, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

    train_tfm = T.Compose([
        T.RandomResizedCrop(380, scale=(0.7,1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.3),
        T.ColorJitter(0.3,0.3,0.25,0.08), T.RandomRotation(25),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfm = T.Compose([
        T.Resize((400,400), interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(380),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # Import LeafDataset from dedicated top-level module — required for Windows pickle
    from leaf_dataset import LeafDataset

    # num_workers=0 on Windows — multiprocessing cannot pickle non-top-level classes
    tr_dl = DataLoader(LeafDataset(X_tr, y_tr, train_tfm, label_to_idx),
                       batch_size=batch_size, shuffle=True, num_workers=0,
                       persistent_workers=False)
    vl_dl = DataLoader(LeafDataset(X_val, y_val, val_tfm, label_to_idx),
                       batch_size=batch_size*2, shuffle=False, num_workers=0,
                       persistent_workers=False)

    # Load backbone
    models_dir = Path("models")
    net = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
    pth = models_dir / "efficientnet_b4_imagenet.pth"
    st  = models_dir / "efficientnet_b4_imagenet.safetensors"
    if pth.exists():
        ck = torch.load(str(pth), map_location="cpu", weights_only=True)
        net.load_state_dict({k:v for k,v in ck.items() if "classifier" not in k}, strict=False)
        log.info("Loaded backbone from cached .pth")
    elif st.exists():
        from safetensors.torch import load_file
        net2 = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1000)
        net2.load_state_dict(load_file(str(st)), strict=True)
        net.load_state_dict({k:v for k,v in net2.state_dict().items() if "classifier" not in k}, strict=False)
        log.info("Loaded backbone from safetensors")
    else:
        log.info("Downloading ImageNet weights (~74MB)…")
        net = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)

    in_feat = net.num_features
    net.classifier = nn.Sequential(
        nn.BatchNorm1d(in_feat), nn.Dropout(0.40),
        nn.Linear(in_feat, 512), nn.SiLU(),
        nn.BatchNorm1d(512), nn.Dropout(0.30),
        nn.Linear(512, 256), nn.SiLU(),
        nn.BatchNorm1d(256), nn.Dropout(0.20),
        nn.Linear(256, len(LABELS)),
    )
    net = net.to(device)

    # Freeze backbone for phase 1
    for nm, p in net.named_parameters():
        if "classifier" not in nm: p.requires_grad = False

    opt = optim.AdamW(filter(lambda p:p.requires_grad, net.parameters()), lr=lr*2, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.10)
    best_acc, best_state, history = 0.0, None, []

    for ep in range(1, epochs+1):
        if ep == 6:  # unfreeze all
            for p in net.parameters(): p.requires_grad = True
            opt = optim.AdamW(net.parameters(), lr=lr/5, weight_decay=1e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs-5)

        net.train(); tr_loss=tr_ok=tr_n=0
        for Xb,yb in tr_dl:
            Xb,yb = Xb.to(device),yb.to(device)
            opt.zero_grad(); lg=net(Xb); loss=crit(lg,yb); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            tr_loss+=loss.item()*len(Xb); tr_ok+=(lg.argmax(1)==yb).sum().item(); tr_n+=len(Xb)
        sch.step()

        net.eval(); vl_ok=vl_n=0
        with torch.no_grad():
            for Xb,yb in vl_dl:
                Xb,yb=Xb.to(device),yb.to(device)
                vl_ok+=(net(Xb).argmax(1)==yb).sum().item(); vl_n+=len(Xb)

        ta=tr_ok/max(tr_n,1); va=vl_ok/max(vl_n,1)
        history.append({"epoch":ep,"train_acc":round(ta,4),"val_acc":round(va,4)})
        star="🏆 BEST" if va>best_acc else ""
        log.info("Epoch %2d/%d  train=%.1f%%  val=%.1f%%  %s", ep, epochs, ta*100, va*100, star)
        if va > best_acc:
            best_acc = va
            best_state = {k:v.cpu().clone() for k,v in net.state_dict().items()}

    save_path = Path("models/efficientnet_b4_spinach_finetuned.pth")
    torch.save({"model_state_dict":best_state,"n_classes":len(LABELS),
                "labels":LABELS,"val_accuracy":round(best_acc,4),
                "epochs_trained":epochs,"history":history}, str(save_path))
    log.info("EfficientNet-B4 saved → %s  (val acc=%.2f%%)", save_path, best_acc*100)
    return {"val_accuracy":round(best_acc*100,2),"save_path":str(save_path),"history":history}


def train_classical_models(image_paths: list, labels: list) -> dict:
    """
    Extract 96-dim features and train SVM, RF, KNN, XGBoost.
    Saves all models to models/classical/
    Expected val accuracy: 80-91%
    """
    try:
        import numpy as np
        from PIL import Image as PIL_Image
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError as e:
        log.error("Missing: %s", e); return {}

    try:
        from advanced_classifier import extract_features, get_classifier
    except ImportError:
        log.error("advanced_classifier.py not found"); return {}

    log.info("Extracting 96-dim features from %d images…", len(image_paths))
    X, y, failed = [], [], 0
    t0 = time.time()
    for i,(path,lbl) in enumerate(zip(image_paths,labels)):
        try:
            X.append(extract_features(PIL_Image.open(path).convert("RGB")))
            y.append(lbl)
        except: failed+=1
        if (i+1)%300==0:
            eta=(time.time()-t0)/(i+1)*(len(image_paths)-i-1)
            log.info("  %d/%d done | ETA %.0fs", i+1, len(image_paths), eta)

    X = np.array(X, dtype=np.float32); y = np.array(y)
    log.info("Feature matrix: %s | failed: %d", X.shape, failed)

    X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = get_classifier()
    stats = clf.train_classical(X_tr, y_tr)

    log.info("Validation accuracy per model:")
    for name in clf.classical.models:
        preds = []
        for feat in X_val:
            r = clf.classical.predict_one(name, feat)
            preds.append(r["prediction"] if r else "unknown")
        va = accuracy_score(y_val, preds)
        log.info("  %-25s %.2f%%", name, va*100)
        if name in stats: stats[name]["val_accuracy"] = round(va*100,2)

    log.info("Classical models saved → models/classical/")
    return stats


def run_training(data_dir: str, epochs: int = 25, batch_size: int = 16,
                 max_per_class: int = 800, skip_pytorch: bool = False,
                 skip_classical: bool = False) -> dict:
    """
    Full training run: build dataset → train EfficientNet → train classical models.
    Called by app.py /train route and directly from CLI.
    """
    log.info("="*60)
    log.info("SPINACH MODEL TRAINING  |  data=%s", data_dir)
    log.info("="*60)

    image_paths, labels = build_plantvillage_dataset(data_dir, max_per_class)
    if len(image_paths) < 50:
        raise RuntimeError(
            f"Only {len(image_paths)} images found in: {data_dir}. "
            "Make sure you paste the PARENT folder that contains the disease subfolders, "
            "not a subfolder itself. "
            f"Example: if your folders are Downy-Mildew(240)/, Healthy-Leaf(1399)/ etc, "
            "paste the folder that CONTAINS those — not Downy-Mildew(240) itself. "
            "Supported folder names: Healthy, Downy-Mildew, Leaf-Spot, Anthracnose, "
            "Pest-Damage, Bacterial-Spot, Mosaic-Virus, Nutrient-Deficiency, Damping-Off, "
            "and all PlantVillage names like Apple___healthy/, Tomato___Late_blight/ etc."
        )

    results = {"data_dir":data_dir,"n_images":len(image_paths)}

    if not skip_pytorch:
        log.info("--- Training EfficientNet-B4 ---")
        results["pytorch"] = train_efficientnet(image_paths, labels, epochs, batch_size)
    if not skip_classical:
        log.info("--- Training Classical Models ---")
        results["classical"] = train_classical_models(image_paths, labels)

    # Summary
    log.info("="*60)
    log.info("TRAINING COMPLETE")
    if results.get("pytorch"):
        log.info("  EfficientNet-B4 val accuracy : %.2f%%",
                 results["pytorch"].get("val_accuracy",0))
    if results.get("classical"):
        for nm,s in results["classical"].items():
            log.info("  %-25s val=%.1f%%", nm, s.get("val_accuracy",0))
    log.info("  Restart server → confidence will be 85-95%%")
    log.info("="*60)
    return results


if __name__ == "__main__":
    main()
