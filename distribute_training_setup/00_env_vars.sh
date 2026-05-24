#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/tmp2/${USER}/DRL_final_workspace"
export WORKSPACE_ROOT

export ANDROID_HOME="${WORKSPACE_ROOT}/android-sdk"
export ANDROID_SDK_ROOT="${ANDROID_HOME}"
export ANDROID_USER_HOME="${WORKSPACE_ROOT}/.android"
export ANDROID_AVD_HOME="${ANDROID_USER_HOME}/avd"
export ANDROID_EMULATOR_HOME="${ANDROID_USER_HOME}"

export PATH="${PATH}:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${ANDROID_HOME}/emulator"
