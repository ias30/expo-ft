# Source and local changes

Source snapshot: `/home/ren9/smw/lcx/openpi/openpi-git`, 2026-11-03.
Reference HEAD: `16498f30c36f862eeae69f93a9ef853d5a688569`.
The snapshot includes the reference working-tree RTC-SFT changes, not only HEAD.
Copied `src`, `scripts`, `packages`, `assets`, package metadata/lockfile and license;
no virtual environment, checkpoints, datasets, Git internals or robot logs.

Already present in the reference snapshot:
- rank-128 language LoRA + full action expert / vision recipe;
- 64-step RealMan LeRobot training data transforms;
- RTC-SFT model config, clean prefix, per-token adaRMS and suffix loss;
- gradient accumulation with per-microbatch RNG, average-before-clipping,
  optional host offload, optimizer-step checkpoint accounting and resume.

Changes on top of that snapshot:
- Pin RTC recipe to microbatch 16, accumulation 4 (base non-RTC recipe unchanged).
- Adapt the reference accumulation regression tests to the RTC recipe instead
  of the stale `pi05_realman_tape` config name; use max prefix 1 for the tiny
  two-action CPU test and control prefix randomness for full-batch equivalence.

This snapshot supports the standalone RealMan RTC-SFT entrypoint. It is NOT the
modified upstream OpenPI fork required by EXPO-FT's original online RL wrappers.
Do not point those wrappers at it as a replacement for their documented fork.

Native objective alignment: RTC loss now delegates to the shared original
EXPO-FT implementation in `expo_ft/rtc_sft.py`. The non-accumulated trainer calls
its native optimizer step; the accumulated trainer divides summed numerator
gradients by summed suffix counts. Original pre-refactor functions are preserved
as independent regression fixtures. The standalone vendored model/trainer now
requires the parent EXPO-FT repository on Python's import path (the launcher
sets this automatically).

The upstream model time-conditioning implementation was inspected at:
https://github.com/pd-perry/openpi/blob/real-time-expo-ft/src/openpi/models/pi0.py
https://github.com/pd-perry/openpi/blob/real-time-expo-ft/src/openpi/models/gemma.py
Its per-token time MLP and adaRMS semantics match our retained model support.
