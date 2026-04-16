#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p venvs
python -m venv venvs/catalog_tfm
# shellcheck disable=SC1091
source venvs/catalog_tfm/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install --no-deps -e ../eq_mag_prediction
pip install -e .
echo "Done. Activate with: source $ROOT/venvs/catalog_tfm/bin/activate"
