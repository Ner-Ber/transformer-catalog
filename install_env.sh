#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install --no-deps -e ../eq_mag_prediction
pip install -e .
echo "Done. Activate with: source $ROOT/.venv/bin/activate"
