#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-/venv/main/bin/python3}"
if [ ! -f "$PYTHON_BIN" ]; then PYTHON_BIN=python3; fi

echo "Installing dependencies..."
"$PYTHON_BIN" -m pip install -q scikit-learn matplotlib numpy

echo "Running category breakdown..."
"$PYTHON_BIN" category_breakdown/category_breakdown.py "$@"

echo "Done."
