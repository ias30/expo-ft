# RealMan 胶带任务：真机在线 RL 启动指南

入口：`train_realman_rl.py`。从已经训练好的 RTC-SFT `10000` checkpoint 开始，执行真机任务、人工标注结果，并在每集结束后更新 critic/edit/filter。

本文命令显式指定参数，避免修改代码默认值后不清楚实际运行配置。当前代码默认值（2026-11-12）为执行窗口 30、候选 3/3、batch size 64、UTD 10、filter_N 32。下面首次真机命令把 **batch size 显式设为 2**；完整训练更新验证通过后，再按显存情况提高。

## 1. 打开终端，设置路径

后面的命令在同一个交互终端执行。新开终端后需要重新定义这些变量。

```bash
cd /home/ren9/smw/lcx/RTC/expo-ft
PY=/home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python
CKPT=/home/ren9/smw/lcx/RTC/expo-ft/checkpoints/realman_rtc/pi05_realman_tape2_rtc/rtc_native_h64_d5_bs8_acc8/10000
RUN_DIR=logs/tape_rl_realman_r30_n3
```

新实验使用一个新的 `RUN_DIR`。已有实验需要按第 7 节恢复，不要删除目录来绕过检查。

权重从 `$CKPT/params` 加载，归一化统计从 `$CKPT/assets/realman_tape2/norm_stats.json` 加载。RL 入口没有 `--norm-stats` 参数；统计文件缺失时会直接报错。

当前 Python 环境已经安装 RL 所需依赖。如果换环境后提示缺少 `tensorflow_probability` / `tree`，可执行：

```bash
uv pip install --python "$PY" 'tensorflow-probability==0.25.0' 'dm-tree==0.1.9'
```

## 2. 真机连接配置

先用现有部署流程启动 ROS 相机发布节点，并确认机械臂已上电、网络连通。此脚本订阅已有图像，不负责启动相机驱动或机械臂控制器。

| 项目               | 当前设置                                        |
| ------------------ | ----------------------------------------------- |
| 右臂 IP / SDK 端口 | `169.254.128.19:8080`                         |
| ROS master         | `http://169.254.128.20:11311`                 |
| 本机 ROS IP        | `169.254.128.101`                             |
| ROS Python 路径    | `/home/ren9/zw/realman_deploy5090/ros_python` |
| 顶部相机 topic     | `/head_camera/image_raw`                      |
| 腕部相机 topic     | `/left_hand_camera/image_raw`                 |
| 控制频率           | 30 Hz                                           |

腕部 topic 虽然叫 `left_hand_camera`，但映射到模型的 `cam_right_wrist`，与采集训练时的映射一致。

右臂 IP、相机 topic 可通过 `--right-robot-ip`、`--head-camera-topic`、`--wrist-camera-topic` 覆盖。ROS master 和本机 ROS IP 当前在 `realman_inference_tape_trainingrtc.py` 的 `load_ros_client()` 中设置；更换网络时需要核对该函数。

策略运动保留右臂基座坐标系的 boundary box，单位 mm：

```text
min = (-170, -500, -118)
max = ( 114, -150,   50)
```

初始化回位豁免 workspace 检查。正常策略下发会检查当前 TCP 和采样的关节插值路径，要求 work frame 为基座坐标系；它是软件运动检查，不替代机械臂控制器的安全功能。

## 3. 先跑推理 benchmark

```bash
"$PY" train_realman_rl.py \
  --benchmark-only \
  --checkpoint "$CKPT" \
  --replan-steps 30 \
  --rtc-delay-steps 5 \
  --control-hz 30 \
  --candidates 2 \
  --edit-candidates 2 \
  --filter-candidates 32
```

预期最终看到：

```text
Benchmark passed; no hardware connected
```

前几次数秒的耗时包含 JIT 编译。这里的时间预算为 `5 / 30 ≈ 167 ms`，VLA 多候选推理必须在预算内完成。执行窗口 30 并不意味着有 30 步推理时间；队列剩 5 步时才发起下一次推理。

如果出现 `Multi-candidate inference exceeds prefix budget`，先把候选改为 `--candidates 2 --edit-candidates 2` 重测，并在后续真机命令中使用同样设置。你的历史 2 候选测试约 110 ms，8 候选约 350 ms；实际时延以本次测量为准。当前 checkpoint 的训练最大 delay 为 5，不能直接把 delay 加到 10 来绕过检查。

Benchmark 不连接 ROS/机械臂，也不执行训练反向传播或 filter backup，因此不验证真机 SDK 时延、batch 64 的训练显存或 filter_N=32 的训练开销。

## 4. 启动真机在线 RL

此命令使用候选 3/3，要求前面的同配置 benchmark 已通过。

```bash
"$PY" train_realman_rl.py \
  --mode rl \
  --checkpoint "$CKPT" \
  --output-dir "$RUN_DIR" \
  --episodes 20 \
  --max-episode-steps 600 \
  --replan-steps 30 \
  --rtc-delay-steps 5 \
  --control-hz 30 \
  --candidates 2 \
  --edit-candidates 2 \
  --filter-candidates 32 \
  --batch-size 2 \
  --utd-ratio 10 \
  --updates-per-episode 10 \
  --warmup-episodes 10 \
  --min-critic-updates 100
```

启动日志会显示 checkpoint、执行窗口、控制频率、batch、UTD、rollout 候选数和 backup 种子数。先核对这些值，再按终端提示连接设备。

| 参数                                         | 此命令的含义                                                         |
| -------------------------------------------- | -------------------------------------------------------------------- |
| `--mode rl`                                | 采集真机数据，并在每集结束后训练                                     |
| `--episodes 20`                            | 本次进程最多尝试 20 集；推理超时中止也占一次尝试                     |
| `--max-episode-steps 600`                  | 每集最多 600 步动作；30 Hz 下约 20 秒，不因夹爪切换暂停            |
| `--replan-steps 30`                        | 每轮执行窗口 30 步，夹爪切换不截断窗口                         |
| `--rtc-delay-steps 5`                      | 剩余 5 步动作作为前缀，在这段时间内计算下一轮候选                    |
| `--candidates 3` / `--edit-candidates 3` | 生成 3 个原始候选和 3 个修正候选，供 Q 选择                          |
| `--filter-candidates 32`                   | backup 时给 32 个噪声种子打分，仅对选出的 1 个做 VLA 去噪            |
| `--batch-size 2`                           | 每次 critic 更新从 replay 抽取 2 个窗口；没有梯度累积                |
| `--utd-ratio 10`                           | 每个 update call 先更新 critic/filter 10 次，再更新 edit/温度各 1 次 |
| `--updates-per-episode 10`                 | 每集结束进行 10 个 update call                                       |
| `--warmup-episodes 10`                     | 启用 edit/Q 控制前，至少完成 10 集本运行目录中的有效 rollout         |
| `--min-critic-updates 100`                 | 启用 edit/Q 控制前，critic 至少更新 100 次                           |

此命令默认冻结 VLA。每集有可训练数据时，共进行 **100 次 critic/filter 更新、10 次 edit/温度更新**。将 `--utd-ratio 10` 改为 `20`，则是 200 次 critic/filter、10 次 edit/温度。逐批更新，不会一次把 20 个 batch 全部放入 GPU，但训练等待时间会增加。

Batch 可以逐步调到 16、32、64；候选数影响在线推理预算，batch size 主要影响每集结束后的训练。未提供 `--batch-size` 时，当前代码会采用 64。

## 5. 每一集，人和机器怎么操作

1. **加载模型和预热。** 此时尚未连接机械臂，等待编译完成。
2. **确认连接。** 出现“模型已就绪……按 Enter 连接设备”后，按 Enter。程序连接 ROS 相机和右臂，并发送夹爪打开命令。
3. **确认回位。** 出现“按 Enter 回到采集初始姿态”后，按 Enter，机械臂回到原采集初始姿态。
4. **摆好胶带。** 回位结束后摆放胶带；出现“摆好胶带后按 Enter 开始”时，再按 Enter 开始本集。
5. **观察抓取并标注。** 在运行该脚本的终端输入 `1` 或 `2`，再按 Enter。只标注整集结果，不需要每步操作。
6. **等待训练和保存。** 本集结束后机器人停止，程序保存数据、更新网络、保存 checkpoint。等下一次回位提示出现后，再开始下一集。

| 执行过程中的输入         | 含义                                                                    |
| ------------------------ | ----------------------------------------------------------------------- |
| `1` + Enter            | 成功结束，终止奖励 1                                                    |
| `2` + Enter            | 失败结束，终止奖励 0；回位仍需等待下一次确认                            |
| 不输入，或 `3` + Enter | 继续执行                                                                |
| Ctrl-C                   | 请求停止退出；在 rollout 中中断的 episode 标记 aborted，不进训练 replay |

对胶带任务，建议固定成功标准，例如“胶带被稳定抓持且离开转盘”，不要仅凭夹爪关闭就标成功。失败和成功的数据都会训练 critic；VLA 的可选 BC 更新只使用合格的成功数据。

600 步耗尽时标记 `truncated=True`，不当作人工失败。碰到未知截断结尾的窗口不参与训练，之前的有效窗口仍可使用。最后一步若已收到人工成功/失败标签，优先使用该标签。

夹爪切换只等待控制器设置确认，不缓停、不等待物理到位、不重发、不丢弃动作队列；下一帧继续下发关节动作。软件会等回到控制循环的检查点才读取人工标签，数字键不是控制器急停。

## 6. 什么时候开始用 edit/Q 控制

首次运行且每集正常产生数据时，前 10 集日志通常为：

```text
Episode ...: RTC-SFT collection
```

这期间执行 SFT 动作，但每集结束后已经在训练 critic/edit/filter。累计满足“10 集本地有效 rollout + 100 次 critic 更新”后，后续集会显示：

```text
Episode ...: VLA + edit + Q
```

这是当前 RealMan 实现的行为：10 集门槛控制 edit/Q 的上线，不是延迟训练开始。UTD 调整只改变更新比例，并未改成论文按采集步数触发 update call 的调度。

## 7. 停止后恢复同一实验

重新设置第 1 节变量，`RUN_DIR` 指向原目录。下面命令对应第 4 节配置：

```bash
"$PY" train_realman_rl.py \
  --mode rl --resume \
  --checkpoint "$CKPT" \
  --output-dir "$RUN_DIR" \
  --episodes 20 \
  --max-episode-steps 600 \
  --replan-steps 30 \
  --rtc-delay-steps 5 \
  --control-hz 30 \
  --candidates 3 \
  --edit-candidates 3 \
  --filter-candidates 32 \
  --batch-size 2 \
  --utd-ratio 10 \
  --updates-per-episode 10 \
  --warmup-episodes 10 \
  --min-critic-updates 100
```

恢复要求目录中存在完整 RL checkpoint。恢复后 `--episodes 20` 表示本次再尝试最多 20 集，不是把总集数补到 20。

模型路径、执行窗口、delay、频率、rollout 候选数、edit scale、去噪步数、prompt 和是否训练 VLA 等必须与保存配置一致。若首次用了 2/2 候选，恢复也必须用 2/2。Batch、UTD、每集 update call 数和 filter 候选数可以调整；启动日志和训练指标记录本次实际值，原 `config.json` 保留建运行时的配置。

## 8. 数据、指标和 checkpoint 在哪里

```text
logs/tape_rl_realman_r30_n3/
├── config.json       # 新建运行时保存的参数
├── episodes/*.npz    # 每集观测、动作、pause、success/aborted/truncated
├── metrics.jsonl     # 每个 update call 的损失和更新步数
└── checkpoints/      # RL 参数与优化器状态，保留最近两个
```

在另一个终端查看指标：

```bash
tail -n 5 /home/ren9/smw/lcx/RTC/expo-ft/logs/tape_rl_realman_r30_n3/metrics.jsonl
```

`update` 是累计 critic 更新次数，`edit_step` 是 edit 更新次数，`vla_step` 是 VLA 更新次数。默认冻结 VLA 时，`vla_step` 不增加。指标还包含实际 `utd_ratio`、`batch_size` 和 `filter_candidates`。

冻结 VLA 的 RL checkpoint 不重复保存整套 VLA 权重，因此恢复时需要保留原始 SFT checkpoint。Episode 数据与模型 checkpoint 分开保存；中断时可能已有新 episode 文件，但相应训练结果尚未形成完整 checkpoint。

## 9. 只采集，或同时微调 VLA

只想先验证真机、采集人工标注数据：使用第 4 节命令，将 `--mode rl` 改成 `--mode collect`，并换一个新输出目录。此模式执行 SFT 动作、保存 episode，不做训练更新。

之后新建 RL 运行时，可在第 4 节命令增加 `--replay-dir 原采集目录/episodes` 导入数据。外部导入 episode 不计入本运行目录的 10 集上线门槛。新导入 demo 的夹爪切换不再标 pause。以前自动推断 pause 的 demo 可重新导入到新目录；真实旧在线记录中的 pause 保留其实际含义，见 [RL.md](RL.md)。

如果要同时微调 VLA：新建独立实验，在第 4 节命令增加 `--train-base-actor`。已有冻结 VLA 的运行不能直接切换此开关恢复。VLA 只使用成功 episode 中完整的 64 步窗口，窗口内部不能跨 pause；没有合格窗口时跳过 VLA 更新。该模式训练显存更高，需要单独验证，不能用推理 benchmark 结果代替。

## 10. 常见现象与处理

| 现象                                                | 处理                                                                               |
| --------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `Unable to initialize backend rocm/tpu`           | NVIDIA 环境中可出现；结合后续 CUDA 运行和最终结果判断，不单凭这两条判失败          |
| 最开始 warmup 很慢                                  | 首次编译；观察后面预热后的耗时和最终结果                                           |
| `Multi-candidate inference exceeds prefix budget` | 减少 rollout 候选，重测；正式运行使用同一组参数                                    |
| 训练 OOM / resource exhausted                       | 降低 `--batch-size`；UTD 是逐批执行，降低 UTD 主要减少总计算量，不替代降低 batch |
| 一集结束后长时间不动                                | 正在训练或保存；日志会输出 `Update call complete`。UTD 越大等待通常越长          |
| `already contains a run`                          | 继续原实验加 `--resume`；新实验换目录                                            |
| `Resume config mismatch`                          | 对照原目录 `config.json`，恢复原模型/动作/时序参数                               |
| 相机超时或图像过期                                  | 检查 ROS master、本机 IP、topic 发布和网络；旧图像不会被继续使用                   |
| Workspace 越界：`predicted_target` / `predicted_path` | 拒绝未下发动作，停止并复查当前位置；符合恢复条件时截断本集、训练保存，再等待确认回位 |
| Workspace 越界：`measured_tcp` / `current_fk`，或坐标系/停机异常 | 锁定故障并退出，检查实际位置与 work/tool frame；不会自动清除故障 |
| 夹爪命令确认失败                               | 本集停止退出，检查 SDK 与实际夹爪状态后再重新运行                                  |
| RTC 候选或控制下发超时                              | 本集 aborted，停止后回到手动重启提示；错误原因及控制耗时需要先处理                 |

预测越界恢复只用于 RL 入口。符合恢复条件时，episode 保存为 `truncated=True`、`end_reason=workspace_rejected`；被拒绝的命令未执行，因此不追加到 actions。之前的完整 transitions 保留，触及截断结尾的训练窗口被 replay 排除，不伪造失败奖励。即使第一步就被拒绝，也会保存一个零动作 episode，但它不进入 replay。

程序保留模型和设备连接，在训练、保存后再次显示“按 Enter 回到采集初始姿态”。你确认回位、摆好胶带，再开始下一集；在完成 reset 前，环境会拒绝任何新策略动作。初始化回位仍按既有规则豁免 workspace 检查。停止失败、停后实际 TCP 越界、无效反馈或坐标系问题均不能进入此恢复流程。边界没有扩宽，也不会自动容忍小幅越界。

## 11. 训练 OOM 后的已验证恢复配置

2026-11-12 实测：原 batch size=2 的运行在 critic 更新阶段 OOM，477 步成功 episode 已保存，但原目录没有完整 RL checkpoint。进一步检查发现冻结 VLA 参数被作为整个 JIT 更新的输出返回，可能产生不必要的大块输出缓冲区。代码已改为在 JIT 外复用冻结参数，同时修复 NPZ 逐帧重复解压整段图像的问题。

修改后，RTX 5090 单卡、完整 `10000` 模型、batch size=1、UTD=10、filter_N=32、候选 2/2，已完成一次包含 10 次 critic/filter 和 1 次 edit/温度更新的离线调用，成功保存 checkpoint。64 项回归测试通过。此验证没有连接机械臂，也不代表 batch 2 或 64 已验证可用；首次训练建议使用这组 batch 1 配置。

以下命令恢复这次验证产生的 checkpoint，并导入原来保存的成功 episode：

```bash
cd /home/ren9/smw/lcx/RTC/expo-ft
PY=/home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 "$PY" train_realman_rl.py \
  --mode rl --resume \
  --output-dir logs/tape_rl_recovery_b1_shared \
  --replay-dir logs/tape_rl_realman_r30_n3/episodes \
  --batch-size 1 --utd-ratio 10 --updates-per-episode 10 \
  --replan-steps 30 --rtc-delay-steps 5 --control-hz 30 \
  --candidates 2 --edit-candidates 2 --filter-candidates 32
```

JAX 显存比例变量名是 `XLA_PYTHON_CLIENT_MEM_FRACTION`，不是 `XLA_CLIENT_MEM_FRACTION`。0.90 是本次空闲单卡验证使用的设置，不会为其他 GPU 进程预留足够显存的保证。原目录中的数据未修改，新增 checkpoint 位于 `logs/tape_rl_recovery_b1_shared/checkpoints/1`。其余 ROS/SDK 的真机控制时延仍以实际运行测量为准。

## 12. 连续动作执行与诊断

夹爪切换按模型输出的开合意图执行（沿用 0.3/0.7 滞回映射）：关节动作正常下发，夹爪切换时发送 `pick/release(speed, ..., False, 1)`。只等待控制器设置确认，不发送缓停、不等待物理到位、不自动重发，也不清空 RTC 队列。夹爪状态缓存表示已确认的命令目标，不是实测开口位置。同目标不重复发送；模型再次切换时立即发送新目标。

模型关节目标继续接受原有 workspace 检查。episode 结束、回位、退出和故障处理中的停止仍保留。SDK 拒绝/通信失败仍报错；没有物理完成轮询，因此不会再因为等待夹爪开合 10 秒而超时，但这也不代表能确认夹爪实际到位。

`--gripper-retry-after`、`--gripper-open-timeout`、`--gripper-close-timeout` 为旧命令行兼容保留，在连续执行模式下不起作用。

诊断参数仍可加在原命令末尾：

```bash
--sdk-debug-dir logs/gripper_diagnostic
```

`*.events.jsonl` 记录 episode/step、最近关节下发、夹爪请求/发送/确认，以及 `gripper_command_accepted`，不将设置确认写成物理完成。原始 SDK DEBUG 输出在终端；该 SDK 的 `*.raw.log` 实测可能只有启动头，建议另外保存完整终端输出。诊断日志会增加 I/O 开销；没有额外的夹爪或停稳反馈查询。

## 13. RL 执行语义变更

连续执行下，critic 保留夹爪切换后的动作，仅按真实窗口长度截断；新 demo 不因切换产生 pause。旧在线 episode 已记录的真实 pause 仍可由 replay 按原边界处理。

执行语义版本已从 2 升至 3。旧 RL checkpoint 的 critic 是按切换截断训练的，不能用 `--resume` 直接接续；请使用新的 `--output-dir`，从原 RTC-SFT `10000` 权重建立新 RL 实验。原权重、旧 checkpoint 和数据文件没有修改。此变更经过无硬件回归测试，完整真机连续执行仍需实测。

## 14. 用指定 RL checkpoint 评估一集

```bash
cd /home/ren9/smw/lcx/RTC/expo-ft
PY=/home/ren9/smw/lcx/openpi/openpi-git/.venv/bin/python
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 "$PY" eval_realman_rl.py \
  --rl-checkpoint /home/ren9/smw/lcx/RTC/expo-ft/logs/tape_rl_realman_r30_n3/checkpoints/20 \
  --episodes 1
```

`eval_realman_rl.py` 从所属运行的 `config.json` 读取原 SFT 路径、归一化来源、候选数、执行窗口、delay 等设置，按指定 step 以只读方式恢复 RL 权重。始终启用 VLA + edit + Q，不受采集热身集数门槛影响，不进行梯度更新或保存新权重，也不加载训练 replay。

完成模型恢复/JIT 后，按提示确认连接、回位、摆放胶带和开始。`1+Enter` 标成功，`2+Enter` 标失败，最多执行原配置的 600 步；一集结束保存评估 episode 后退出。结果默认存到原运行下独立的 `evaluations/checkpoint-20-时间戳/`，不会混进训练的 `episodes/`。

可加 `--benchmark-only` 仅恢复/JIT，完全不连接 ROS/机械臂。也可用 `--max-episode-steps` 或新的 `--output-dir` 指定评估步数上限与结果目录。入口已完成无硬件测试；启动真机仍按交互提示由操作者确认。
