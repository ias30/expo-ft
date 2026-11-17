# RealMan native training-time RTC → online EXPO RL

中文操作指南：[真机启动、人工标注与恢复运行](REAL_ROBOT_START_CN.md)。

Entry point: `train_realman_rl.py`. It uses `RealTimeEXPOFTLearner` (the existing critic, edit actor, filter critic and optional prefix BC update), with the same vendored OpenPI implementation as the RTC-SFT checkpoint. The robot environment and learner run on this workstation in one process. DROID's client, SpaceMouse actions, automatic reset/retry and dummy zero actions are not used.

## Environment

Use the Python environment that ran RTC-SFT. The RL distribution additionally needs the repository's declared TensorFlow Probability dependency:

```bash
cd /home/ren9/smw/lcx/RTC/expo-ft
PY=/home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python
uv pip install --python "$PY" 'tensorflow-probability==0.25.0' 'dm-tree==0.1.9'
```

The entry point pins `third_party/openpi_rtc` on its import path. Norm statistics must exist at `<checkpoint>/assets/realman_tape2/norm_stats.json`; missing statistics are an error. Initial SFT weights come from `<checkpoint>/params`. `--resume` restores an **RL** checkpoint in `--output-dir`.

## Verify inference without hardware

```bash
"$PY" train_realman_rl.py --benchmark-only --candidates 2 --edit-candidates 2
```

This loads the full checkpoint, compiles both base-only and edit/Q selection, checks the physical action prefix, and measures the candidate sampling budget. It connects to neither ROS nor the SDK. With delay 5 and 30 Hz, the VLA has 167 ms. If the benchmark fails, lower control Hz or candidate count (at least 2 for this learner's edit/Q path). Measure again after changing model size or candidate count. This is an inference benchmark, not proof that full-size RL gradients fit in GPU memory.

## Collect and label

```bash
"$PY" train_realman_rl.py --mode collect --output-dir logs/tape_rl_collect
```

After model warmup, the terminal prompts before connecting the robot, before homing, and before beginning each episode. Put the tape back in place after homing. During execution:

- `1` + Enter: success, terminal reward 1.
- `2` + Enter: failure, terminal reward 0.
- No input / `3` + Enter: continue.
- Ctrl-C / SDK, camera or actual-pose/work-frame fault: stop; retain an aborted episode for diagnosis, exclude it from replay. Planning/control-dispatch deadline misses also stop and abort the episode, then return to the manual restart prompt. SDK/gripper timeouts remain latched faults and are never automatically retried.

Predicted target/path workspace violations in the RL environment are recoverable only after a successful stop and a fresh check of actual pose/FK/frame. The unsent action is rejected; collection returns a truncated episode with `end_reason=workspace_rejected`, retains earlier transitions and excludes cutoff windows from replay sampling. The normal update/save path runs before the next manual reset prompt. Another policy action is forbidden until reset completes. Actual-pose violations, invalid feedback, frame mismatch, or stop failures remain fatal. The standalone inference script still exits on workspace rejection.

A step limit (`--max-episode-steps`, default 600) ends with `truncated=True`, not a failure. A queued manual success/failure label at the final step takes precedence. A label never sends another action. The label belongs to the last command actually sent. Manual success labels are **not** teleoperation/HIL actions. Initial/reset motion is exempt from the workspace check; policy motion uses base-frame mm bounds min (-170,-500,-118), max (114,-150,50). SDK joints use degrees, model/environment actions use radians.

Gripper transitions stream alongside joint commands: wait only for the setting ACK, with no arm stop, physical completion wait, automatic retry or plan invalidation. The cached gripper value is the acknowledged command target. Workspace and fault stops remain active. SDK errors still abort; physical gripper completion is not confirmed by this mode.

Each episode is saved atomically under `episodes/*.npz`: T+1 observations (two resized RGB cameras, raw state), T actual absolute commands, controller-pause boundaries, and explicit success/aborted/truncated labels. Failed, completed episodes train the critic too; aborted episodes do not. Windows ending at an unknown time-limit cutoff are excluded before sampling, so they affect neither critic, edit, filter nor temperature updates. Earlier windows from truncated episodes remain usable. Older NPZ files lacking `truncated` retain their original terminal semantics; old timeout recordings cannot be distinguished from manual failures automatically.

On ordinary ticks, the next observation is fetched once at the next control tick, completing the previous transition and supplying the next policy input. After a pause it is fetched after stopping and draining the old plan. The final action also gets a next observation before labels or the time limit end the episode.

## Train and run RL

```bash
"$PY" train_realman_rl.py --mode rl --output-dir logs/tape_rl_run \
  --replay-dir logs/tape_rl_collect/episodes \
  --batch-size 2 --utd-ratio 20 --updates-per-episode 10 --filter-candidates 32
```

Updates run only after the robot has stopped at episode end. Defaults freeze the VLA, train critic/edit/filter, and keep collecting with RTC-SFT until both 10 local rollout episodes and 100 critic updates have completed. Then edit/Q selection is enabled. `--warmup-episodes` and `--min-critic-updates` control these gates; they are collection thresholds, not guarantees of critic quality.

`--filter-candidates` (alias `--filter-N`) defaults to **32** and sets the learner's `filter_N`. During each Bellman backup the filter scores this many Gaussian noise seeds per replay sample, keeps one, and runs VLA denoising only for that survivor. Backup selection then compares one base action with one edited action (`filter_n_edit=1`). This is independent of the rollout `--candidates` / `--edit-candidates` and their RTC latency budget. More seeds increase filter scoring work and temporary tensors, not the number of VLA candidates denoised per sample. The inference-only benchmark does not measure this training cost.

Changing filter seed count does not change checkpoint parameter shapes, so it may be changed on resume. Older runs used 2; pass `--filter-candidates 2` to preserve that setting. Startup logs and each training metrics row report the effective count; new runs also save it in `config.json`.

`--utd-ratio` is currently **10** by default; the example above explicitly requests **20**. Each update call performs `utd_ratio` sequential critic/filter optimizer steps on freshly sampled minibatches, then one edit/temperature step using the last minibatch. If `--train-base-actor` is enabled and successful windows exist, it also performs one VLA step from a separate success-only batch. These are separate optimizer steps, not gradient accumulation. `--batch-size` is the size of each minibatch; there is no requirement that it be divisible by UTD. Minibatches are sampled and transferred one at a time instead of putting `batch_size * UTD` camera observations on the GPU together.

With `--updates-per-episode 10 --utd-ratio 20`, each episode with eligible replay data is followed by **200 critic/filter steps and 10 edit/temperature steps**, plus up to 10 optional VLA steps. `--offline-updates` also counts update calls. Use `--utd-ratio 1` for the previous 1:1 schedule. UTD does not affect the inference benchmark, but increases training time between episodes. This matches the paper's critic-to-policy update ratio only: this entry point still uses fixed calls per episode and starts training as soon as eligible replay is available; the 10-episode gate controls edit/Q deployment, not the start of training.

UTD can change on resume without changing checkpoint shapes. Specify it explicitly when restoring an older run to avoid silently adopting the new default. The startup log reports the effective schedule; each metrics row records UTD, minibatch size, cumulative critic/edit/VLA steps, and last-minibatch losses. Nonfinite metrics in any intermediate UTD step abort the run before another rollout.

To initialize the critic from already labelled data without hardware:

```bash
"$PY" train_realman_rl.py --mode offline --output-dir logs/tape_rl_offline \
  --replay-dir logs/tape_rl_collect/episodes --offline-updates 100
```

Continue that run using `--mode rl --resume --output-dir logs/tape_rl_offline`. Resume must use the same model/action/timing settings. Native sampling horizon is 64; RL execution window comes from `configs/task/realman_tape.py` (currently 30), prefix delay 5. This differs from the standalone inference script's configurable execution interval.

Use `--train-base-actor` **from the beginning of a separate run** to also train the VLA. Its actor batch contains successful full 64-step windows only. If no such window exists yet, only critic/edit/filter update. This requires substantially more memory than freezing the base actor. The option cannot be toggled on resume because optimizer state differs.

`--replay-capacity` defaults to 20000 transitions (about 6 GB of image data plus overhead) and evicts whole old episodes. Only one replay is allocated; total training steps do not determine memory capacity. Images are stored as uint8; candidates and action windows are small. Reduce batch size, capacity, update count or model training scope according to measurements. Metrics are written to `metrics.jsonl`. Checkpoints include optimizer states, encoder, target critic, edit/filter/temperature state and RNG, are saved synchronously at episode boundaries, and retain the latest two. With the VLA frozen, its weights and optimizer are omitted and reconstructed from the original SFT checkpoint; path and checkpoint/normalization fingerprint must match. With VLA training enabled, full learner state is saved. Episodes remain separately on disk.

## Import existing LeRobot demonstrations

Provide explicit success labels, e.g. `labels.json` containing `{"0": true, "1": false}`. Unlisted episodes are skipped. Data must already be six absolute joint angles in radians plus absolute gripper (0 closed, 1 open), matching RTC-SFT.

```bash
"$PY" scripts/realman_tape/import_rl_demos.py \
  --repo-id realman_tape2 --root /path/to/realman_tape2 \
  --labels /path/to/labels.json --output data/realman_rl_demos
```

Pass that output with `--replay-dir`. The final dataset row supplies the terminal observation; its action is omitted because no next observation exists. The supplied success label is associated with the preceding action. Do not label an episode successful unless the final retained observation shows the task completed. Demonstrations must match the deployment control cadence; changing sampling frequency is not performed by the importer.

The importer uses the controller's 0.3/0.7 hysteresis to reconstruct binary commands without marking gripper transitions as pauses. Re-import previously converted demos with inferred pause flags into a new directory. Historical online episodes retain their recorded actual pause boundaries.

## Action-coordinate invariant

Replay keeps absolute commands. Each sampled chunk is converted relative to **its own starting state**, then normalized. Delayed VLA candidates are decoded using the delayed state, then rebased against the current state before edit/Q. Training backups perform the identical conversion. Prefixes outside the model remain absolute; they are re-encoded against each new VLA observation. The gripper is never converted to a joint delta.

Critic windows stop at the first controller pause or episode end, or after r actions. Every pre-pause start remains eligible, including one-step grasp transitions. Replay supplies the actual window length k and discount gamma**k; a pause is not terminal. After a pause the bootstrap uses the post-pause observation and delay=0. Normal rows use the configured delay. Mixed-delay batches carry matching image, continuous state, state-containing prompt tokens and masks; the sampler clamps each row separately and extracts its execution window at that row's offset. Actor windows contain H actually executed actions from successful episodes. The critic canonicalizes gripper proposals using the same 0.3/0.7 hysteresis as execution, and retains actions after gripper transitions, masking only beyond the actual window length. Absent tail joints are zero and the tail gripper uses an explicit out-of-range sentinel; this representation is shared by replay loss, online Q selection, backup Q selection and edit training. A straight-through derivative is used for the binary gripper in edit training (a biased surrogate). No reset motions or padded post-boundary actions are treated as executed commands. Critic bootstrap uses the explicit terminal observation and a zero continuation mask on terminal windows.

## Tests

```bash
JAX_PLATFORMS=cpu PYTHONPATH=third_party/openpi_rtc/src:third_party/openpi_rtc/packages/openpi-client/src:. "$PY" -m pytest tests/test_realman_rl.py \
  tests/test_realman_workspace.py tests/test_realman_gripper.py -q
```

The integration test uses reduced-width production Pi0 modules and the actual EXPO critic/edit/filter update and checkpoint restore. It does not connect hardware or validate the full checkpoint's GPU training memory budget.


This revision changes critic action representation and replay/backup semantics (version 2).
Start a new RL output directory; old version-1 RL weights are not resumed silently.
Previously collected NPZ episodes can be imported with `--replay-dir` and are rebuilt
using the corrected pipeline. Keep the original SFT checkpoint for lightweight restore.
The example command uses batch size 2 for initial checks; current CLI defaults are
set in `train_realman_rl.py`. Increase batch size independently of candidate count.
Candidate count directly consumes the
fixed RTC latency budget; a larger N is not a substitute for a larger training batch.

Execution semantics version 3 uses continuous gripper dispatch. Start a new RL output directory from the same RTC-SFT weights; resume from version 2 RL checkpoints is rejected because critic action-window semantics changed. Legacy gripper timeout/retry CLI options are accepted but unused.
