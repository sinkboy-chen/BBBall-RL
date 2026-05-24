#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/sinkboy-chen/BBBall-RL.git}"
DEST="${DEST:-${HOME}/Desktop/BBBall-RL}"

if [ -d "${DEST}/.git" ]; then
  echo "Repo already exists at ${DEST}. Skipping clone."
  exit 0
fi

mkdir -p "$(dirname "${DEST}")"
git clone "${REPO_URL}" "${DEST}"
