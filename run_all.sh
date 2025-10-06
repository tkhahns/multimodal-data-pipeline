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

PYFEAT_RUNNER_DIR="$ROOT_DIR/packages/cv_models/external/pyfeat_runner"
if [ -d "$PYFEAT_RUNNER_DIR" ]; then
    if command -v python3.11 >/dev/null 2>&1; then
        echo "[INFO] Installing Py-Feat runner environment (Python 3.11)..."
        (
            cd "$PYFEAT_RUNNER_DIR"
            poetry env use python3.11 >/dev/null 2>&1 || true
            poetry install
        )
    else
        echo "[WARN] python3.11 not found; skipping Py-Feat runner setup.\n       Install Python 3.11 to enable pf_* facial features." >&2
    fi
fi

echo "[INFO] Running multimodal data pipeline..."
poetry run python run_pipeline.py "$@"

echo "[INFO] Pipeline run completed."
