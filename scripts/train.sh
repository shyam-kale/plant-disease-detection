#!/usr/bin/env bash
# train.sh — Fit sklearn models on a labelled dataset and save to models/sklearn/
# Usage: bash scripts/train.sh /path/to/dataset
#
# NOTE: MobileNetV2 uses pre-trained PlantVillage weights (no fine-tuning).
#       This script trains only the classical sklearn classifiers.
set -euo pipefail
DATASET="${1:?Usage: bash scripts/train.sh /path/to/dataset}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python - <<PYEOF
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path('.')))
from evaluate import load_dataset
from ml_models import SklearnRegistry

print(f"Loading dataset: $DATASET")
X, y, paths = load_dataset('$DATASET')
print(f"Loaded {len(X)} samples.")

registry = SklearnRegistry()
registry.pipelines = {}
registry.training_stats = {}
registry.cv_results = {}
registry.active = 'random_forest'
import threading; registry._lock = threading.Lock()
registry._definitions = registry._build_definitions()

print("Training with 5-fold CV evaluation...")
stats = registry.fit(X, y, cv_folds=5)
print("Training complete. Models saved to models/sklearn/")
for name, s in stats.items():
    agg = s.get('cv_accuracy', {})
    print(f"  {name}: CV acc={agg.get('mean',0):.4f}±{agg.get('std',0):.4f}")
PYEOF
