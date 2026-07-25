VENV := .venv
PYTHON := python3.10

.PHONY: venv install init clean format format-check lint test tests check build upload version-patch version-minor version-major

venv: 
	@echo "Using Python version: ${PYTHON_VERSION}"
	uv venv -p "${PYTHON_VERSION}"

install:
	uv sync --all-extras
	uv lock

clean:
	rm -rf $(VENV)

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

test:
	uv run python -m pytest


check:
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test

build:
	uv build

version-patch:
	uv version --bump patch

version-minor:
	uv version --bump minor

version-major:
	uv version --bump major

upload:
	twine upload dist/*

