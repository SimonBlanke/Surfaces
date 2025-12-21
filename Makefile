# =============================================================================
# Surfaces Makefile
# =============================================================================

.PHONY: build install uninstall test lint format check clean help

# =============================================================================
# Installation
# =============================================================================

install-editable:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-test:
	pip install -e ".[test]"

install-test-minimal:
	pip install -e ".[test-minimal]"

install-build:
	pip install build

reinstall-editable:
	pip uninstall -y surfaces || true
	pip install -e .

# =============================================================================
# Building
# =============================================================================

build:
	python -m build

install: build
	pip install dist/*.whl

uninstall:
	pip uninstall -y surfaces || true
	rm -fr build dist *.egg-info

reinstall: uninstall install

# =============================================================================
# Testing
# =============================================================================

py-test:
	python -m pytest -x -p no:warnings tests/

test-examples:
	cd tests && python _test_examples.py

test: py-test test-examples

# Test with minimal dependencies (no sklearn, no viz, no GFO)
test-minimal:
	python -m pytest -x -p no:warnings \
		tests/test_1d_functions.py \
		tests/test_2d_functions.py \
		tests/test_nd_functions.py \
		tests/test_all_test_functions.py \
		tests/test_api/test_input_type.py \
		tests/test_api/test_metric.py \
		tests/test_api/test_sleep.py

# Integration tests with optimization libraries (requires GFO, optuna, scipy)
test-integrations:
	python -m pytest -x -p no:warnings \
		tests/test_optimization.py \
		tests/test_api/test_search_space.py

# Test with coverage
test-cov:
	python -m pytest --cov=surfaces --cov-report=term-missing --cov-report=xml -p no:warnings tests/

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check .

lint-fix:
	ruff check --fix .

format:
	ruff format .

format-check:
	ruff format --check .

check: lint format-check

fix: lint-fix format

# =============================================================================
# Pre-commit
# =============================================================================

pre-commit-install:
	pre-commit install

pre-commit-all:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# =============================================================================
# Data Collection
# =============================================================================

database:
	python -m collect_search_data.py

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -fr build dist *.egg-info
	rm -fr .pytest_cache .ruff_cache .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Surfaces Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Installation:"
	@echo "  install-editable      Install package in editable mode"
	@echo "  install-dev           Install with dev dependencies"
	@echo "  install-test          Install with test dependencies (full)"
	@echo "  install-test-minimal  Install with minimal test dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test                  Run all tests"
	@echo "  py-test               Run pytest only"
	@echo "  test-minimal          Run core tests (no optional deps)"
	@echo "  test-integrations     Run integration tests (GFO, optuna, scipy)"
	@echo "  test-cov              Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint                  Check code with ruff"
	@echo "  lint-fix              Fix linting issues"
	@echo "  format                Format code with ruff"
	@echo "  format-check          Check code formatting"
	@echo "  check                 Run all checks (lint + format)"
	@echo "  fix                   Fix all issues (lint + format)"
	@echo ""
	@echo "Pre-commit:"
	@echo "  pre-commit-install    Install pre-commit hooks"
	@echo "  pre-commit-all        Run pre-commit on all files"
	@echo "  pre-commit-update     Update pre-commit hooks"
	@echo ""
	@echo "Other:"
	@echo "  build                 Build package"
	@echo "  clean                 Remove build artifacts"
