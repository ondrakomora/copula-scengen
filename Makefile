VENV := .venv
PYTHON := python3.10

.PHONY: venv install init clean format lint test tests build upload

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

lint:
	uv run ruff check .

test:
	uv run python -m pytest


check:
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

build:
	uv build

upload:
	twine upload dist/*

