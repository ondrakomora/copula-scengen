VENV := .venv
PYTHON := python3.10

.PHONY: venv install init clean

venv: 
	@echo "Using Python version: ${PYTHON_VERSION}"
	uv venv -p "${PYTHON_VERSION}"

install:
	uv sync --all-extras
	uv lock

clean:
	rm -rf $(VENV)

tests:
	uv run python -m pytest

build:
	python -m build

upload:
	twine upload dist/*

