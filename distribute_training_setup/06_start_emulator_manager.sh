#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env_vars.sh"
source "${WORKSPACE_ROOT}/.venv/bin/activate"

REPO_DIR="${REPO_DIR:-${HOME}/Desktop/BBBall-RL}"
python3 "${REPO_DIR}/env_scripts/emulator_manager.py"
