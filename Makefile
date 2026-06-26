# ── Spinach Disease Detection — Makefile ─────────────────────────────────────
# Usage:
#   make install         Install all dependencies
#   make test            Run unit + integration tests with coverage
#   make evaluate DATA=<path>   Run full evaluation pipeline
#   make reproduce DATA=<path>  Full end-to-end reproducibility run
#   make docker-build    Build Docker image
#   make docker-up       Start services with Docker Compose
#   make clean           Remove generated artefacts

PYTHON     := python
PIP        := pip
PYTEST     := pytest
DATA       ?= ./dataset
FOLDS      ?= 5
MLFLOW_URI ?= ./mlruns
RESULTS    := results

.PHONY: all install test evaluate ablation stats reproduce \
        docker-build docker-up docker-down clean lint

# ── Default target ───────────────────────────────────────────────────────────

all: install test

# ── Environment ──────────────────────────────────────────────────────────────

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

# ── Tests ────────────────────────────────────────────────────────────────────

test:
	$(PYTEST) tests/ \
		--cov=. \
		--cov-report=term-missing \
		--cov-report=html:results/coverage_html \
		--cov-report=xml:results/coverage.xml \
		-v \
		--tb=short
	@echo "Test report: results/coverage_html/index.html"

test-unit:
	$(PYTEST) tests/test_ml_models.py tests/test_metrics.py tests/test_statistical_tests.py \
		-v --tb=short

test-api:
	$(PYTEST) tests/test_api.py -v --tb=short

# ── Evaluation ───────────────────────────────────────────────────────────────

evaluate:
	@echo "Running evaluation pipeline on dataset: $(DATA)"
	@echo "CV folds: $(FOLDS)"
	$(PYTHON) evaluate.py \
		--data $(DATA) \
		--folds $(FOLDS) \
		--mlflow-uri $(MLFLOW_URI)
	@echo "Results saved to: $(RESULTS)/"

ablation:
	@echo "Running ensemble weight ablation study…"
	$(PYTHON) ablation.py --data $(DATA) --steps 11
	@echo "Ablation results: $(RESULTS)/ablation_weights.json"

stats:
	@echo "Running statistical significance tests…"
	$(PYTHON) statistical_tests.py --results $(RESULTS)/evaluation_results.json
	@echo "Statistical test results: $(RESULTS)/statistical_tests.json"

# ── Reproducibility ───────────────────────────────────────────────────────────

reproduce:
	@echo "========================================================"
	@echo " FULL REPRODUCIBILITY RUN"
	@echo " Dataset: $(DATA)"
	@echo "========================================================"
	@echo "[1/4] Installing dependencies…"
	$(MAKE) install
	@echo "[2/4] Running tests…"
	$(MAKE) test
	@echo "[3/4] Running evaluation…"
	$(MAKE) evaluate DATA=$(DATA)
	@echo "[4/4] Running ablation + statistical tests…"
	$(MAKE) ablation DATA=$(DATA)
	$(MAKE) stats
	@echo "========================================================"
	@echo " DONE — All results in: $(RESULTS)/"
	@echo " Figures : $(RESULTS)/figures/"
	@echo " LaTeX   : $(RESULTS)/summary_table.tex"
	@echo "========================================================"

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker build -t spinach-disease-api:1.0.0 .

docker-up:
	docker compose up --build -d
	@echo "API: http://localhost:5000"
	@echo "MLflow: http://localhost:5001"

docker-down:
	docker compose down

# ── Maintenance ───────────────────────────────────────────────────────────────

lint:
	$(PYTHON) -m py_compile app.py ml_models.py evaluate.py \
		statistical_tests.py ablation.py
	@echo "Syntax check passed."

clean:
	@echo "Removing generated artefacts…"
	-rm -rf results/figures/*.png
	-rm -f  results/evaluation_results.json
	-rm -f  results/summary_table.json
	-rm -f  results/summary_table.tex
	-rm -f  results/statistical_tests.json
	-rm -f  results/ablation_weights.json
	-rm -rf results/coverage_html
	-rm -f  results/coverage.xml
	-rm -rf __pycache__ tests/__pycache__ .pytest_cache
	@echo "Clean complete."
