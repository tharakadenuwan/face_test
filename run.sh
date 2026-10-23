#!/bin/bash
# One-click runner for Raspberry Pi
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists; if not, create and install
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv' on Raspberry Pi..."
    python3 -m venv venv
    echo "Installing required packages from requirements.txt..."
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

# Run predictions
./venv/bin/python predict.py "$@"
