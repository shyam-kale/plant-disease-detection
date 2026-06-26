#!/usr/bin/env bash
# evaluate.sh — Run evaluation pipeline on a labelled dataset
# Usage: bash scripts/evaluate.sh /path/to/dataset [n_folds]
set -euo pipefail
DATASET="${1:?Usage: bash scripts/evaluate.sh /path/to/dataset [n_folds]}"
FOLDS="${2:-5}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python evaluate.py --data "$DATASET" --folds "$FOLDS"
echo "Evaluation complete. Results in: results/"
