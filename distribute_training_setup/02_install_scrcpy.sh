#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env_vars.sh"

cd "${WORKSPACE_ROOT}"

SCRCPY_VERSION="v4.0"
ARCHIVE="scrcpy-linux-x86_64-${SCRCPY_VERSION}.tar.gz"
URL="https://github.com/Genymobile/scrcpy/releases/download/${SCRCPY_VERSION}/${ARCHIVE}"

if [ ! -f "${ARCHIVE}" ]; then
  wget "${URL}"
fi

tar -xvf "${ARCHIVE}"
