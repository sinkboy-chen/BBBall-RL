#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env_vars.sh"
source "${WORKSPACE_ROOT}/.venv/bin/activate"

RAY_PORT="${RAY_PORT:-6379}"
ray start --head --num-gpus=1 --port="${RAY_PORT}" --dashboard-host=0.0.0.0
