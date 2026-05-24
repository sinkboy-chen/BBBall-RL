#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env_vars.sh"

mkdir -p "${ANDROID_USER_HOME}/cache"
mkdir -p "${ANDROID_AVD_HOME}"
mkdir -p "${ANDROID_HOME}/cmdline-tools"
cd "${ANDROID_HOME}/cmdline-tools"

TOOLS_ZIP="commandlinetools-linux-14742923_latest.zip"
if [ ! -d "latest" ]; then
  if [ ! -f "${TOOLS_ZIP}" ]; then
    wget "https://dl.google.com/android/repository/${TOOLS_ZIP}"
  fi
  unzip -q "${TOOLS_ZIP}"
  mv cmdline-tools latest
fi

yes | sdkmanager --licenses

sdkmanager \
  "platform-tools" \
  "platforms;android-31" \
  "system-images;android-31;google_apis;x86_64" \
  "emulator"

AVD_NAME="${AVD_NAME:-pixel5_api31}"
if ! avdmanager list avd | grep -q "Name: ${AVD_NAME}"; then
  echo "no" | avdmanager create avd \
    -n "${AVD_NAME}" \
    -k "system-images;android-31;google_apis;x86_64" \
    -d "pixel_5"
fi
