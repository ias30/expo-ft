#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Reuse the reference environment if this repository has not been installed yet.
python_bin="${RTC_PYTHON:-/home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/ren9/.cache/huggingface/lerobot}"
exec "$python_bin" train_realman_rtc.py pi05_realman_tape2_rtc \
    --exp-name="${RTC_EXP_NAME:-rtc_native_h64_d5_bs16_acc4}" \
    --checkpoint-base-dir="$PWD/checkpoints/realman_rtc" "$@"
