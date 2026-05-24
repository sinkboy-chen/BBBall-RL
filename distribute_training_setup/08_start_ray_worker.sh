#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env_vars.sh"
source "${WORKSPACE_ROOT}/.venv/bin/activate"

if [ -z "${HEAD_ADDR:-}" ]; then
  echo "HEAD_ADDR is required (e.g. 192.168.1.200:6379)"
  exit 1
fi

NUM_CPUS="${NUM_CPUS:-3}"
ray start --address="${HEAD_ADDR}" --num-cpus="${NUM_CPUS}"
