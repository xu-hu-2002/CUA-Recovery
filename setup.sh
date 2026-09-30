#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

bash scripts/setup_third_party.sh "$@"

[[ -f .env ]] || cp .env.example .env
echo "Setup complete. Fill in .env, then: source .venv/bin/activate"
