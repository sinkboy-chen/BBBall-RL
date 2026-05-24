#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env_vars.sh"

LOCK_DIR="${BBBALL_LOCK_DIR:-${WORKSPACE_ROOT}/emu_locks}"
mkdir -p "${LOCK_DIR}"

if [ "${1:-}" = "--clean" ]; then
  rm -f "${LOCK_DIR}/slot_"*.lock
fi

echo "Lock dir ready: ${LOCK_DIR}"
