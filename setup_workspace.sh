#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$PROJECT_ROOT/pipeline_tools/init_dataset_folders.py"
