"""
ablation.py — Ensemble weight ablation study
============================================
Spinach Plant Disease Detection System

Sweeps the MobileNetV2 weight α ∈ {0.0, 0.1, …, 1.0}
(sklearn weight = 1 - α) and records accuracy, F1-macro, AUC for
each combination.  Identifies the empirically optimal α and writes
results to results/ablation_weights.json and a figure to
results/figures/ablation_weights.png.

Usage
-----
    python ablation.py --data <dataset_root>

Authors : research team
Version : 1.0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOT_OK = True
except ImportError:
    PLOT_OK = False

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

LABELS      = Config.LABELS
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_predictions(data_root: str) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load ground-truth labels, MobileNetV2 probability arrays, and
    sklearn-ensemble probability arrays from the dataset.

    Returns
    -------
    y_true          : (n,)   ground-truth labels (strings)
    deep_probas     : (n, C) MobileNetV2 probability arrays
    sklearn_probas  : (n, C) sklearn ensemble (mean of all models) probability arrays
    """
    from PIL import Image as PILImage
    from ml_models import (
        ImageProcessor, DeepModel, XGBoostClassifier, TORCH_OK,
        deep_model, xgb_model,
    )
    import time as _time

    root  = Path(data_root)
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    # Wait for models
    if TORCH_OK:
        for _ in range(60):
            if deep_model.ready:
                break
            _time.sleep(1.0)

    y_true_list, deep_list, sk_list = [], [], []

    for label in LABELS:
        label_dir = root / label
        if not label_dir.exists():
            continue
        for img_path in label_dir.iterdir():
            if img_path.suffix.lower() not in exts:
                continue
            try:
                raw  = img_path.read_bytes()
                proc = ImageProcessor(raw)
                proc.prepare()
                feats = proc.extract_features()

                # Deep model
                deep_res = deep_model.predict(proc.pil_image()) if deep_model.ready else None

                # XGBoost prediction
                xgb_res = xgb_model.predict(feats) if xgb_model.ready else None
                sk_all  = {"xgboost": xgb_res} if xgb_res else {}
                if not deep_res or not sk_all:
                    continue

                deep_arr = np.array(
                    [deep_res["all_probabilities"][lbl] / 100.0 for lbl in LABELS],
                    dtype=np.float32,
                )
                # Average sklearn model probabilities
                sk_avg = np.zeros(len(LABELS), dtype=np.float32)
                for m_res in sk_all.values():
                    for j, lbl in enumerate(LABELS):
                        sk_avg[j] += m_res["all_probabilities"][lbl] / 100.0
                sk_avg /= max(len(sk_all), 1)

                y_true_list.append(label)
                deep_list.append(deep_arr)
                sk_list.append(sk_avg)
            except Exception:
                continue

    return (
        np.array(y_true_list),
        np.array(deep_list,  dtype=np.float32),
        np.array(sk_list, dtype=np.float32),
    )


def run_ablation(y_true: np.ndarray,
                 deep_probas: np.ndarray,
                 sklearn_probas: np.ndarray,
                 steps: int = 11) -> list[dict]:
    """
    Sweep α ∈ linspace(0, 1, steps) and record metrics.

    Ensemble probability: P = α * P_deep + (1-α) * P_sklearn
    """
    alphas  = np.linspace(0.0, 1.0, steps)
    results = []
    y_bin   = label_binarize(y_true, classes=LABELS)

    for alpha in alphas:
        beta = 1.0 - alpha
        blended = alpha * deep_probas + beta * sklearn_probas

        # Normalise each row (guard against fp drift)
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        blended /= row_sums

        y_pred = np.array([LABELS[i] for i in np.argmax(blended, axis=1)])

        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        try:
            auc = float(roc_auc_score(
                y_true, blended, multi_class="ovr", average="macro", labels=LABELS,
            ))
        except Exception:
            auc = None

        results.append({
            "alpha_deep":    round(float(alpha), 2),
            "alpha_sklearn": round(float(beta),  2),
            "accuracy":      round(acc, 4),
            "f1_macro":      round(f1m, 4),
            "auc_macro":     round(auc, 4) if auc is not None else None,
        })

    return results


def plot_ablation(results: list[dict], save_path: Path) -> None:
    if not PLOT_OK:
        return
    alphas = [r["alpha_deep"]  for r in results]
    accs   = [r["accuracy"]    for r in results]
    f1s    = [r["f1_macro"]    for r in results]
    aucs   = [r["auc_macro"]   for r in results if r["auc_macro"] is not None]
    auc_x  = [r["alpha_deep"]  for r in results if r["auc_macro"] is not None]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, accs,  "o-", label="Accuracy",  color="steelblue",  lw=2)
    ax.plot(alphas, f1s,   "s-", label="F1-Macro",  color="darkorange", lw=2)
    if aucs:
        ax.plot(auc_x, aucs, "^-", label="AUC-Macro", color="green",      lw=2)

    # Mark optimal α for F1-macro
    best_idx = int(np.argmax(f1s))
    ax.axvline(alphas[best_idx], color="red", lw=1.5, ls="--",
               label=f"Optimal α={alphas[best_idx]:.2f}  F1={f1s[best_idx]:.4f}")

    ax.set_xlabel("α  (MobileNetV2 weight;  sklearn weight = 1 − α)", fontsize=12)
    ax.set_ylabel("Metric value", fontsize=12)
    ax.set_title("Ensemble Weight Ablation Study", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation figure saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble weight ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="Dataset root (one folder per label).")
    parser.add_argument("--steps", type=int, default=11,
                        help="Number of α values to sweep (linspace 0→1).")
    args = parser.parse_args()

    print("Loading predictions from dataset…")
    y_true, deep_probas, sklearn_probas = load_predictions(args.data)

    if len(y_true) == 0:
        print("ERROR: No predictions loaded. Check dataset and model readiness.")
        sys.exit(1)

    print(f"Running ablation over {args.steps} α values on {len(y_true)} samples…")
    results = run_ablation(y_true, deep_probas, sklearn_probas, steps=args.steps)

    best = max(results, key=lambda r: r["f1_macro"])
    print(f"\nOptimal α (by F1-Macro): {best['alpha_deep']:.2f}")
    print(f"  MobileNetV2 weight: {best['alpha_deep']:.2f}")
    print(f"  sklearn weight    : {best['alpha_sklearn']:.2f}")
    print(f"  Accuracy          : {best['accuracy']:.4f}")
    print(f"  F1-Macro          : {best['f1_macro']:.4f}")
    print(f"  AUC-Macro         : {best.get('auc_macro', 'N/A')}")

    print("\nFull ablation table:")
    print(f"  {'α_deep':>6}  {'α_sk':>6}  {'Acc':>8}  {'F1-Mac':>8}  {'AUC':>8}")
    for r in results:
        print(f"  {r['alpha_deep']:>6.2f}  {r['alpha_sklearn']:>6.2f}  "
              f"{r['accuracy']:>8.4f}  {r['f1_macro']:>8.4f}  "
              f"{(r['auc_macro'] or 0):>8.4f}")

    # Save
    out_path = RESULTS_DIR / "ablation_weights.json"
    with open(out_path, "w") as fh:
        json.dump({"results": results, "optimal": best}, fh, indent=2)
    print(f"\nAblation results saved: {out_path.resolve()}")

    # Plot
    plot_ablation(results, FIGURES_DIR / "ablation_weights.png")


if __name__ == "__main__":
    main()
