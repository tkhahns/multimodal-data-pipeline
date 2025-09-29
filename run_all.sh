#!/bin/bash
# Simple helper to install dependencies with Poetry and run the full pipeline.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v poetry >/dev/null 2>&1; then
    echo "[ERROR] Poetry is required but not installed. See https://python-poetry.org/docs/#installation" >&2
    exit 1
fi

echo "[INFO] Installing project dependencies via Poetry..."
poetry install

echo "[INFO] Running multimodal data pipeline..."
poetry run python run_pipeline.py "$@"

echo "[INFO] Pipeline run completed."
