#!/usr/bin/env bash
# run.sh — Launch Copycat AI Avatar
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    echo "Error: venv not found. Run 'bash setup.sh' first."
    exit 1
fi

source venv/bin/activate
python3 src/main.py
