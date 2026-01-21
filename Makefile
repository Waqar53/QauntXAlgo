.PHONY: install dev test lint format clean run docker-up docker-down

# =============================================================================
# INSTALLATION
# =============================================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev,research]"

# =============================================================================
# DEVELOPMENT
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=quantxalgo --cov-report=html --cov-report=term

lint:
	ruff check quantxalgo/ tests/
	mypy quantxalgo/

format:
	black quantxalgo/ tests/
	ruff check --fix quantxalgo/ tests/

# =============================================================================
# RUNNING
# =============================================================================

run:
	python -m quantxalgo.api.main

run-dev:
	uvicorn quantxalgo.api.main:app --reload --host 0.0.0.0 --port 8000

backtest:
	python scripts/run_backtest.py

# =============================================================================
# DOCKER
# =============================================================================

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-build:
	docker build -f docker/Dockerfile -t quantxalgo:latest .

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

# =============================================================================
# DATA
# =============================================================================

download-data:
	python scripts/download_data.py

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ htmlcov/ .coverage

# =============================================================================
# DOCS
# =============================================================================

docs:
	mkdocs serve

docs-build:
	mkdocs build
