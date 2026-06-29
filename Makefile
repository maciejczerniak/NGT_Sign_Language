PYTORCH_GPU = https://download.pytorch.org/whl/cu124

.PHONY: install install-gpu format lint typecheck test check run docs docs-open help

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install:  ## Install all dependencies with CPU torch (default, Docker, CI)
	poetry install --with dev --no-interaction

install-gpu: install  ## Overwrite torch with the CUDA 12.4 build for local GPU dev
	poetry run pip install torch torchvision \
		--index-url $(PYTORCH_GPU) \
		--upgrade \
		--quiet
	@echo "GPU torch installed. Verify: poetry run python -c \"import torch; print(torch.cuda.is_available())\""

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

format:  ## Run Black formatter
	poetry run black src/ tests/

lint:  ## Run Flake8 linter
	poetry run flake8 src/

typecheck:  ## Run MyPy type checker
	poetry run mypy src/

test:  ## Run tests with coverage
	poetry run pytest --cov=src --cov-report=term-missing --cov-fail-under=90

check: format lint typecheck test  ## Run all quality checks

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

run:  ## Start the API server
	poetry run sign-language serve --host 0.0.0.0 --port 8080

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

docs:  ## Build Sphinx documentation
	poetry run sphinx-apidoc -f -o docs/reference src/sign_language
	poetry run sphinx-build -b html docs/ docs/_build/html

docs-open: docs  ## Build and open documentation in browser
	open docs/_build/html/index.html

# ---------------------------------------------------------------------------

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
