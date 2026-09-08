#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$1" == "--cli" ]; then
    echo "=== Launching Interactive CLI Demo ==="
    python3 demo/cli_demo.py
else
    echo "=== Launching Streamlit Web Demo ==="
    echo "Opening Streamlit UI at http://localhost:8501"
    python3 -m streamlit run demo/app.py --server.port 8501
fi
