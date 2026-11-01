#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then echo "Missing .venv. Follow README.md setup first." >&2; exit 1; fi
source .venv/bin/activate
exec streamlit run app.py
