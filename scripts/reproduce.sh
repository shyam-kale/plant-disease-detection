#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# reproduce.sh — One-command full reproducibility script
#
# Usage:
#   bash scripts/reproduce.sh /path/to/dataset
#
# What it does:
#   1. Verifies Python version (requires 3.11)
#   2. Installs all pinned dependencies
#   3. Runs the full test suite (must pass)
#   4. Runs 5-fold stratified CV evaluation on the provided dataset
#   5. Runs ensemble weight ablation study
#   6. Runs statistical significance tests
#   7. Prints final summary
#
# All results are written to: ./results/
# All figures  are written to: ./results/figures/
# LaTeX table  is written to : ./results/summary_table.tex
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DATASET="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ── Colour output ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo "========================================================================"
echo "  Spinach Disease Detection — Full Reproducibility Run"
echo "  $(date)"
echo "========================================================================"

# ── Step 0: Dataset check ────────────────────────────────────────────────────
if [[ -z "$DATASET" ]]; then
    error "Usage: bash scripts/reproduce.sh /path/to/dataset"
fi
if [[ ! -d "$DATASET" ]]; then
    error "Dataset directory not found: $DATASET"
fi
info "Dataset: $DATASET"

# ── Step 1: Python version ────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
info "Python version: $PY_VER"
if [[ "$PY_VER" != 3.11* ]]; then
    warn "Python 3.11.x recommended (found $PY_VER). Continuing anyway."
fi

# ── Step 2: Install dependencies ─────────────────────────────────────────────
info "[1/5] Installing pinned dependencies from requirements.txt…"
$PYTHON -m pip install --upgrade pip --quiet
$PYTHON -m pip install -r requirements.txt --quiet
info "Dependencies installed."

# ── Step 3: Run tests ────────────────────────────────────────────────────────
info "[2/5] Running test suite…"
$PYTHON -m pytest tests/ \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=html:results/coverage_html \
    -v --tb=short 2>&1 | tee results/test_output.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    error "Tests failed. Fix tests before publishing results."
fi
info "All tests passed."

# ── Step 4: Evaluation ───────────────────────────────────────────────────────
info "[3/5] Running 5-fold stratified CV evaluation…"
$PYTHON evaluate.py \
    --data "$DATASET" \
    --folds 5 \
    --mlflow-uri ./mlruns 2>&1 | tee results/evaluate_output.log
info "Evaluation complete."

# ── Step 5: Ablation ─────────────────────────────────────────────────────────
info "[4/5] Running ensemble weight ablation study…"
$PYTHON ablation.py \
    --data "$DATASET" \
    --steps 11 2>&1 | tee results/ablation_output.log
info "Ablation complete."

# ── Step 6: Statistical tests ─────────────────────────────────────────────────
info "[5/5] Running statistical significance tests…"
$PYTHON statistical_tests.py \
    --results results/evaluation_results.json 2>&1 | tee results/stats_output.log
info "Statistical tests complete."

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  REPRODUCIBILITY RUN COMPLETE"
echo "========================================================================"
echo "  Results JSON : $(realpath results/evaluation_results.json)"
echo "  Summary JSON : $(realpath results/summary_table.json)"
echo "  LaTeX table  : $(realpath results/summary_table.tex)"
echo "  Figures      : $(realpath results/figures/)"
echo "  Test coverage: $(realpath results/coverage_html/index.html)"
echo "  Statistical  : $(realpath results/statistical_tests.json)"
echo "  Ablation     : $(realpath results/ablation_weights.json)"
echo "========================================================================"
echo "  Git commit   : $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================================"
