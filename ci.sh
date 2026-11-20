#!/usr/bin/env bash
# CI local: venv con dependencias fijadas por hash + pytest. No necesita red (salvo para instalar) ni GPU.
# En el clúster: `module load python/3.11` antes de ejecutarlo.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
    "$PY" -m venv .venv
    .venv/bin/python -m pip install -q --upgrade pip
    .venv/bin/python -m pip install -q --require-hashes -r requirements.lock
    .venv/bin/python -m pip install -q --no-deps -e .
fi

.venv/bin/python -m medidor entorno
.venv/bin/python -m pytest -q "$@"
