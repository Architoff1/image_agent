#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then echo "Missing .venv. Follow README.md setup first." >&2; exit 1; fi
source .venv/bin/activate
exec python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
