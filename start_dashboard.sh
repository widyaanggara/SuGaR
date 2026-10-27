#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8080}"

echo "Dashboard tersedia di http://localhost:${PORT}"
cd "$PROJECT_ROOT"
python "$PROJECT_ROOT/web_dashboard/server.py" "$PORT"
