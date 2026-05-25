#!/usr/bin/env bash
set -euo pipefail

# Pull a golden snapshot from another workstation.
# Example:
#   SOURCE_HOST=140.112.30.189 SOURCE_USER=b12902131 SNAP_NAME=game_ready \
#   bash distribute_training_setup/clone_snapshot.sh

SOURCE_HOST="${SOURCE_HOST:-140.112.30.189}"
SOURCE_USER="${SOURCE_USER:-b12902131}"
AVD_NAME="${AVD_NAME:-pixel5_api31}"
SNAP_NAME="${SNAP_NAME:-game_ready}"

SRC="/tmp2/${SOURCE_USER}/DRL_final_workspace/.android/avd/"
DST="/tmp2/${USER}/DRL_final_workspace/.android/avd/"

mkdir -p "${DST}"
rsync -avz --progress "${SOURCE_USER}@${SOURCE_HOST}:${SRC}" "${DST}"

# Patch all path references from source user to local user.
find "${DST}" -type f \( -name "*.ini" -o -name "*.cfg" \) | while read -r f; do
    sed -i "s|/tmp2/${SOURCE_USER}/DRL_final_workspace|/tmp2/${USER}/DRL_final_workspace|g" "$f"
done