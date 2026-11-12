# RealMan π0.5 RTC-SFT

This offline entrypoint uses LeRobot directly. It does not use the DROID HDF5
loader, replay buffer, RL learner or robot client.

## Recipe

- π0.5; action horizon 64, padded model action dimension 32 (7 physical dimensions).
- Language Transformer attention and FFN LoRA rank=128, alpha=128; base language weights frozen.
- Action expert, action/time projections and SigLIP vision encoder fully trainable.
- RTC-SFT: independently sample a clean prefix of length 0..5 for each example;
  prefix flow time is zero, suffix receives noise and the flow-matching loss.
- Use EXPO-FT's native objective: sum masked errors / total valid suffix positions.
  Both the DROID wrapper and this entrypoint share `expo_ft/rtc_sft.py`.
  With accumulation, sum error numerators and their gradients across microbatches,
  sum valid counts, then divide once. Samples with longer suffixes correctly
  contribute more positions; there is no per-example or per-microbatch mean.
- Global microbatch 16 × 4 gradient accumulation steps = effective batch 64.
  These are global sizes across GPUs, not per-GPU batch sizes.
- One optimizer step consumes four fresh microbatches at fixed parameters. Each
  has its own random key for augmentation, flow noise/time and prefix sampling.
  Divide accumulated gradients by the total suffix count before clipping/Adam;
  update step, LR schedule and EMA once.
- 10,000 optimizer updates; checkpoints every 3,333 updates and at completion.
- Dataset `realman_tape2` under `HF_LEROBOT_HOME`; episodes 0..79 only. The current
  dataset metadata lists 95 episodes; this recipe deliberately preserves the
  reference's explicit 80-episode training subset and does not automatically
  treat all 95 as training data. No automatic validation loop is added.
- Reuse copied normalization assets from the reference configuration. If the
  dataset or units changed since those stats were computed, recompute them.

## Run

```bash
bash scripts/realman_tape/offline_train.sh
```

The launcher defaults to the existing reference OpenPI Python environment:
`/home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python`. `RTC_PYTHON` can select
another compatible environment. All model/trainer imports come from this repo's
`third_party/openpi_rtc`, not the editable OpenPI installed in that environment.
The launcher sets the checkpoint root to `expo-ft/checkpoints/realman_rtc`.

Start from a compatible ordinary SFT checkpoint instead of pi05_base:

```bash
bash scripts/realman_tape/offline_train.sh \
  --weight-loader.params-path=/absolute/path/to/STEP/params
```

Resume an interrupted RTC run with the same experiment name and configuration:

```bash
bash scripts/realman_tape/offline_train.sh --resume
```

Override experiment name / environment / dataset cache as needed:

```bash
RTC_EXP_NAME=my_rtc_run RTC_PYTHON=/path/to/python \
HF_LEROBOT_HOME=/path/to/lerobot bash scripts/realman_tape/offline_train.sh
```

Optional reference implementation feature: `--gradient-accumulation-offload`
keeps accumulated gradients in host RAM between microbatches. Default is false.

## Implementation map

- `train_realman_rtc.py`: selects the vendored source/client and dispatches to its JAX trainer.
- `third_party/openpi_rtc/src/openpi/training/config.py`: `pi05_realman_tape2_rtc`, batch=16, accumulation=4, model/data/freeze settings.
- `third_party/openpi_rtc/src/openpi/models/pi0_config.py`: RTC switch and max-prefix validation.
- `expo_ft/rtc_sft.py`: native EXPO-FT velocity forward, four-way RNG split, prefix loss terms and single optimizer step.
- `expo_ft/agents/vla/pi05.py`: imports the shared native step; original DROID path retains its interface.
- `third_party/openpi_rtc/src/openpi/models/pi0.py`: RTC loss delegates to the shared objective; retains per-token time embedding support.
- `third_party/openpi_rtc/src/openpi/models/gemma.py`: rank-128 LoRA variant and per-token adaRMS conditioning.
- `third_party/openpi_rtc/scripts/train.py`: separate gradient computation, summation, mean and optimizer update; checkpoint/resume support.
- `third_party/openpi_rtc/src/openpi/training/config.py` / `policies/realman_policy.py`: LeRobot cameras and relative-joint / absolute-gripper transforms.

The source snapshot and its modifications are described in
`third_party/openpi_rtc/LOCAL_CHANGES.md`. Training support does not implement the
prefix-clamped RTC-SFT deployment sampler or the RealMan asynchronous control
loop. The copied guidance-based inference is not a substitute for those.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
PYTHONPATH="$PWD:$PWD/third_party/openpi_rtc/src:$PWD/third_party/openpi_rtc/packages/openpi-client/src" \
/home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python -m pytest -o addopts='' \
  tests/test_realman_rtc_config.py \
  third_party/openpi_rtc/src/openpi/models/rtc_training_test.py \
  third_party/openpi_rtc/scripts/train_accumulation_test.py \
  third_party/openpi_rtc/scripts/native_rtc_test.py -q
```

Tests include small real Pi0.5 forward/backward updates, frozen/trainable groups,
accumulation versus a full batch with controlled randomness, and toy-model
checkpoint/resume. They do not constitute a full-size pretrained GPU run.

Validation result (2026-11-03): 16 model/config/accumulation checks passed;
4 checkpoint/resume checks passed outside the restricted sandbox. Inside the
sandbox, Orbax asynchronous saving stalled; unrestricted rerun completed in
14.67 seconds. At that validation stage, full-size GPU training and robot deployment were not run;
see the subsequent full-model smoke test below.

## If the first backward pass runs out of GPU memory

```bash
bash scripts/realman_tape/offline_train_low_memory.sh --no-wandb-enabled
```

This alternative uses batch 4 × accumulation 16 (effective batch 64), CPU
accumulator offload and a JAX memory fraction of 0.90 unless either supported
memory-fraction variable is already set. It retains the full vision/action
training recipe. A full pretrained GPU smoke test subsequently passed on an RTX 5090 (32 GB)
with this batch/offload configuration and memory fraction 0.95; see below. Offload only helps after the first microbatch; reducing microbatch
size reduces activation memory, not the persistent model/optimizer state.
The higher memory fraction assumes other processes leave sufficient free VRAM.

## Native objective alignment

The top-level DROID driver cannot consume RealMan LeRobot directly. We retain
OpenPI's data loader, freeze filter, optimizer configuration, accumulation and
checkpoint driver. Without accumulation the driver calls the native
`train_step_p1_prefix` directly; with accumulation it differentiates the numerator
of that same shared loss and divides by the aggregate suffix count.

Original objective and update code before refactoring is frozen in
`tests/fixtures/original_rtc_step.py`. Numerical tests compare loss, full gradient
trees, parameters, Adam state, EMA and step. Scalar maximum delays and explicit
per-example delays are covered. Unequal suffix counts are tested with both CPU
offload and device accumulation. Tests use a reduced-size Pi0.5, not pretrained
full-size GPU training.

For accumulation, the micro-step is folded into the RNG to obtain independent
randomness. This preserves the objective/distributions but does not promise the
same random draws or bitwise trajectory as one monolithic batch with the same
seed. Data augmentation upstream of the objective remains specific to each data
pipeline (RealMan/OpenPI versus DROID/replay-buffer augmentation).

Use a new experiment name, e.g. `RTC_EXP_NAME=rtc_native_h64_d5_bs4_acc16`, for
runs intended to use the native objective from their first update. Old checkpoint
weights are loadable, but resuming them cannot undo updates under the old
per-example-normalized objective. Normalization statistics are unaffected.

Native-alignment validation: 23 tests passed (19 model/objective/accumulation
checks plus 4 checkpoint/resume checks). The frozen original function fixture
was AST-compared to repository HEAD and matches exactly. Numerical comparisons
use tolerance for floating-point reduction order; full-size GPU validation is recorded below. Launchers now default to `rtc_native_...` experiment names so the old
objective's runs are not silently reused.

## Full-model GPU smoke test (2026-11-03)

Revalidated all 23 CPU tests. On the local RTX 5090 (32607 MiB), loaded the
real `realman_tape2` dataset (training subset 0..79), existing norm statistics,
and full pi05_base checkpoint, using H=64, prefix 0..5, rank=128 language LoRA,
full action expert and vision updates, microbatch=4, accumulation=16,
CPU gradient offload and `XLA_CLIENT_MEM_FRACTION=0.95`.

A fresh run completed update 1 and saved. A separate process restored that
checkpoint and completed updates 2 and 3. Both exited 0 with no OOM. Logged
losses were 0.0350, 0.0443, 0.0378 and gradient norms 0.8195, 0.8008, 0.8381.
The latest checkpoint's normalization JSON matches the training asset exactly.
The first attempt was accidentally stopped by the user (SIGTERM); the successful
retry and resume are the evidence for the result.

Commands (use a fresh experiment name to reproduce):

```bash
XLA_CLIENT_MEM_FRACTION=0.95 RTC_EXP_NAME=verify_native_full_20261103_retry \
bash scripts/realman_tape/offline_train_low_memory.sh \
  --no-wandb-enabled --num-train-steps 1 --save-interval 1 --log-interval 1

XLA_CLIENT_MEM_FRACTION=0.95 RTC_EXP_NAME=verify_native_full_20261103_retry \
bash scripts/realman_tape/offline_train_low_memory.sh \
  --no-wandb-enabled --num-train-steps 3 --save-interval 2 --log-interval 1 --resume
```

Evidence: `verification/native_rtc/results.json`, `environment.json`,
`source_audit.json`, `cpu_tests.log`, `gpu_step1_retry.log`, `gpu_resume.log`.
This verifies objective parity on reduced test models and feasibility of the
full-model training/checkpoint path. It does not establish long-run convergence,
robot success rates, or that microbatch 16 fits this GPU.
# RTC-SFT 实机推理（10000 checkpoint）

入口是仓库根目录的 `realman_inference_tape_trainingrtc.py`，参考
`openpi-git/examples/realman_tape/realman_inference_tape.py`，使用本仓库的
`pi05_realman_tape2_rtc` 模型和 checkpoint 自带的归一化统计。
默认 checkpoint：
`checkpoints/realman_rtc/pi05_realman_tape2_rtc/rtc_native_h64_d5_bs8_acc8/10000`。

```bash
cd /home/ren9/smw/lcx/RTC/expo-ft

# 离线推理：不连接 ROS、相机或机械臂。
XLA_CLIENT_MEM_FRACTION=0.55 \
  /home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python \
  realman_inference_tape_trainingrtc.py --benchmark-only --benchmark-iterations 30

# 实时相机测速：连接 ROS，相机输入 + 固定初始关节状态，不连接机械臂。
XLA_CLIENT_MEM_FRACTION=0.55 \
  /home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python \
  realman_inference_tape_trainingrtc.py --benchmark-cameras

# 实机：连接右臂、释放夹爪并回到采集初始位置，再开始异步推理执行。
XLA_CLIENT_MEM_FRACTION=0.55 \
  /home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python \
  realman_inference_tape_trainingrtc.py
```

右臂末端默认启用与参考脚本一致的软件 boundary box（基座坐标系，单位 mm）：
`min=(-170, -500, -118)`，`max=(114, -150, 50)`。
初始化连接及回初始位置不检查 workspace；每次推理生成的 RTC/同步关节运动命令前检查实际 TCP、目标 TCP
及每关节最大间隔 0.5° 的插值路径。实际位置越界、目标/采样路径越界、反馈异常、
工作坐标系不是基座零变换，或正运动学与实际 TCP 偏差超过 2mm 时，拒绝运动并停臂退出。
允许从盒外执行初始化回位；初始化结束后的推理动作仍要求实际末端及目标/采样路径在盒内。
运行时保持工作/工具坐标系不变。此检查是软件采样保护，不是控制器连续轨迹硬限位；
跟踪误差、制动距离和采样点间的轨迹不构成硬件级保证。新增的 SDK 查询也会增加控制耗时。

初始右臂关节角保持参考脚本原值（度）：
`[109.89800262451172, 84.50399780273438, 52.55799865722656, -43.3380012512207, 68.61699676513672, -41.744998931884766]`，
回初始位置速度为 `5`，夹爪初始为释放状态。`--no-reset-pose` 可跳过回位。

默认参数与连接：

| 参数 | 默认值 / 含义 |
| --- | --- |
| `--control-hz` | `30`，动作发送频率 |
| `--steps-to-execute` | `8`，两次推理启动之间的最小动作步数 |
| `--rtc-delay-steps` | `5`，固定干净前缀长度，必须在训练范围 `1..5` 内 |
| `--denoising-steps` | `10` |
| `--right-robot-ip` | `169.254.128.19`，端口 `8080` |
| `--head-camera-topic` | `/head_camera/image_raw` → `cam_high` |
| `--wrist-camera-topic` | `/left_hand_camera/image_raw` → `cam_right_wrist` |
| `--ros-python-path` | `/home/ren9/zw/realman_deploy5090/ros_python` |
| `--output` | `realman_tape_trainingrtc.csv`，关节命令以度记录 |

ROS master 和本机 IP 沿用参考脚本：`169.254.128.20:11311`、`169.254.128.101`。
夹爪阈值和 SDK 非阻塞命令沿用参考逻辑。RTC 执行期间闭合、释放都会先停臂，
等待连续两次健康的完成反馈（闭合 mode=2/6，释放 mode=1），再丢弃旧 chunk、
重新观测推理并重置控制时钟，避免 SDK 实际阻塞约 1 秒触发控制超时。
等待超时分别由 `--gripper-close-timeout`、`--gripper-open-timeout` 设置，均默认 10 秒。
启动时首次释放不进入 RTC 暂停/重规划流程。

采样与训练一致：将尚待执行的前 `d` 个绝对动作转换为当前观测下的归一化增量，
每次去噪都固定这段前缀，前缀 token 的时间设为 `0`，只生成剩余动作。
推理期间执行旧 chunk 的这 `d` 步，新 chunk 从索引 `d` 接入。
提前返回的结果等到第 `d` 步再接入；超过这一边界尚未返回则停止并报告超时。
`--no-rtc` 可用于同步执行对照，此时使用零长度前缀。

2026-11-10 本机离线验证：10000 checkpoint，H=64、D=7、10 次去噪，
10 次预热后调用约 `73–75ms`，在 30Hz 下折算为 3 步。
这不包含真实相机、SDK 和线程调度开销；实机时应结合相机测速与运行日志判断。
如推理超过 5 步窗口，应降低 `--control-hz`，不能把前缀长度调到训练范围之外。
测试日志位于 `verification/training_rtc_inference/`。

