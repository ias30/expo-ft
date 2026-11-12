#!/usr/bin/env bash
# Lower microbatch activation memory; preserve an effective batch of 64.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Honor either spelling accepted by the installed jaxlib. Do not set both.
if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" && -z "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_CLIENT_MEM_FRACTION=0.90
fi
export RTC_EXP_NAME="${RTC_EXP_NAME:-rtc_native_h64_d5_bs4_acc16}"
exec bash scripts/realman_tape/offline_train.sh \
    --batch-size 4 \
    --gradient-accumulation-steps 16 \
    --gradient-accumulation-offload "$@"
