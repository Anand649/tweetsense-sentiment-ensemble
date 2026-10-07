PYTHON ?= python
CONFIG ?= configs/benchmark.yaml

.PHONY: install install-cpu lint format test smoke data check benchmark report serve mlflow docker clean

install:            ## editable install with dev tools
	$(PYTHON) -m pip install -e ".[dev]"

install-cpu:        ## same, with CPU-only PyTorch (Linux CI / servers)
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check src app tests && ruff format --check src app tests

format:
	ruff check --fix src app tests && ruff format src app tests

test:
	pytest -q

smoke:              ## one-minute grid on the committed samples
	sentiment-benchmark benchmark --smoke

data:               ## download what can be downloaded, then write the manifest
	sentiment-benchmark data download
	sentiment-benchmark data manifest

check:
	sentiment-benchmark data check

benchmark:          ## full grid -> reports/, models/, MLflow runs in reports/mlflow.db
	sentiment-benchmark benchmark --config $(CONFIG)

report:             ## regenerate figures, results.md and the README tables from saved probabilities
	sentiment-benchmark report

serve:              ## Flask dashboard + API on http://127.0.0.1:5000
	sentiment-benchmark serve

mlflow:             ## MLflow UI over the local run store
	mlflow ui --backend-store-uri sqlite:///$(CURDIR)/reports/mlflow.db --port 5001

docker:             ## API + PostgreSQL + MLflow server (reads .env)
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info reports/smoke models/smoke
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
