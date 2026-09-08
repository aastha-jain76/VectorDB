#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Launching VectorDB REST API Server ==="
echo "Swagger Documentation: http://localhost:8000/docs"
echo "OpenAPI JSON Schema  : http://localhost:8000/openapi.json"
python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000
