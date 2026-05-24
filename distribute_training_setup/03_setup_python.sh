#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env_vars.sh"

cd "${WORKSPACE_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first (https://astral.sh/uv)."
  exit 1
fi

uv venv --python 3.12
source "${WORKSPACE_ROOT}/.venv/bin/activate"

REPO_DIR="${REPO_DIR:-${HOME}/Desktop/BBBall-RL}"
uv pip install -r "${REPO_DIR}/requirements-base.txt"

# Torch install strategy:
# - CPU nodes: TORCH_VARIANT=cpu
# - GPU nodes: default (CUDA-enabled wheel) or set TORCH_INDEX_URL explicitly
TORCH_VARIANT="${TORCH_VARIANT:-cpu}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

if [ -n "${TORCH_INDEX_URL}" ]; then
  uv pip install --index-url "${TORCH_INDEX_URL}" torch torchvision
elif [ "${TORCH_VARIANT}" = "cpu" ]; then
  uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
else
  uv pip install torch torchvision
fi
