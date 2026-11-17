#!/usr/bin/env python3
"""RealMan training-time RTC online RL. See scripts/realman_tape/RL.md."""
import argparse
import json
import logging
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/openpi_rtc/src'),
               str(ROOT / 'third_party/openpi_rtc/packages/openpi-client/src')]


def parse_args(argv=None):
    # Share the proven ROS/SDK options with SFT; RL options own the policy schedule.
    from realman_inference_tape_trainingrtc import parse_args as hardware_defaults
    from configs.task.realman_tape import get_config
    defaults = hardware_defaults([])
    task = get_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=defaults.checkpoint)
    p.add_argument('--output-dir', default='logs/realman_trainingrtc_rl')
    p.add_argument('--mode', choices=('collect', 'rl', 'offline'), default='collect')
    p.add_argument('--benchmark-only', action='store_true', help='Load/JIT/profile without ROS or SDK')
    p.add_argument('--resume', action='store_true', help='Restore an RL checkpoint, not the SFT checkpoint')
    p.add_argument('--replay-dir', action='append', default=[], help='Additional labelled episode NPZ directories')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--max-episode-steps', type=int, default=600)
    p.add_argument('--replay-capacity', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--utd-ratio', type=int, default=10,
                   help='Critic/filter minibatch updates per one edit/temperature (and optional VLA) update')
    p.add_argument('--updates-per-episode', type=int, default=10, help='Update calls per episode; each has utd-ratio critic steps')
    p.add_argument('--offline-updates', type=int, default=100, help='Number of offline update calls')
    p.add_argument('--warmup-episodes', type=int, default=10)
    p.add_argument('--min-critic-updates', type=int, default=100)
    p.add_argument('--candidates', type=int, default=3)
    p.add_argument('--edit-candidates', type=int, default=3)
    p.add_argument('--filter-candidates', '--filter-N', dest='filter_candidates', type=int, default=32,
                   help='Backup noise seeds scored by the filter critic; only one seed is denoised by VLA')
    p.add_argument('--edit-scale', type=float, default=.05)
    p.add_argument('--train-base-actor', action='store_true')
    p.add_argument('--replan-steps', type=int, default=task.replan_steps)
    p.add_argument('--rtc-delay-steps', type=int, default=task.delay)
    p.add_argument('--control-hz', type=float, default=task.control_hz)
    p.add_argument('--control-timeout-ms', type=float, default=100)
    p.add_argument('--denoising-steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default=task.language_instruction)
    for name in ('ros_python_path', 'right_robot_ip', 'head_camera_topic', 'wrist_camera_topic',
                 'camera_max_age', 'camera_connect_timeout', 'gripper_speed', 'gripper_force',
                 'gripper_close_timeout', 'gripper_open_timeout', 'gripper_retry_after', 'sdk_debug_dir'):
        value = getattr(defaults, name)
        p.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)
    args = p.parse_args(argv)
    if not 0 <= args.rtc_delay_steps <= min(5, args.replan_steps) or not 1 <= args.replan_steps <= 32:
        p.error('Require 0 <= delay <= min(5, replan_steps), 1 <= replan_steps <= 32')
    if args.candidates < 2 or not 0 < args.edit_candidates <= args.candidates:
        p.error('Require candidates >= 2 and 0 < edit-candidates <= candidates')
    if min(args.episodes, args.batch_size, args.utd_ratio, args.updates_per_episode, args.offline_updates,
           args.max_episode_steps, args.denoising_steps, args.filter_candidates) < 1:
        p.error('Counts must be positive')
    if args.replay_capacity < max(64, args.max_episode_steps):
        p.error('Replay capacity must hold at least one complete episode and horizon 64')
    if any(not math.isfinite(v) or v <= 0 for v in (
        args.control_hz, args.edit_scale, args.control_timeout_ms, args.camera_max_age,
        args.camera_connect_timeout, args.gripper_close_timeout, args.gripper_open_timeout)):
        p.error('Control Hz, edit scale and timeouts must be finite and positive')
    if not 1 <= args.gripper_speed <= 1000 or not 50 <= args.gripper_force <= 1000:
        p.error('Invalid gripper speed/force')
    if not math.isfinite(args.gripper_retry_after) or args.gripper_retry_after < 0:
        p.error('gripper-retry-after must be finite and nonnegative')
    if min(args.warmup_episodes, args.min_critic_updates) < 0:
        p.error('Warmup thresholds must be nonnegative')
    args.rtc, args.reset_pose, args.enable_visualization, args.speed = True, False, False, 30
    args.workspace_min, args.workspace_max = tuple(task.workspace_min), tuple(task.workspace_max)
    return args


def warmup(agent, args):
    import jax
    import numpy as np
    from realman_inference_tape_trainingrtc import INITIAL_RIGHT_JOINTS_DEG
    observation = dict(state=np.array([*np.deg2rad(INITIAL_RIGHT_JOINTS_DEG), 1], np.float32),
                       images={k: np.zeros((3, 224, 224), np.uint8) for k in ('cam_high', 'cam_right_wrist')},
                       prompt=args.prompt)
    prefix = None
    for i in range(6):
        start = time.monotonic()
        chunks, agent = agent.sample_pre_cache(observation, prefix)
        jax.block_until_ready(chunks)
        sampled = time.monotonic()
        delay = 0 if prefix is None else args.rtc_delay_steps
        if prefix is not None and delay:
            np.testing.assert_allclose(np.asarray(chunks)[:, :delay],
                                       np.broadcast_to(prefix, (args.candidates, delay, 7)), atol=2e-5)
        actions, agent, _ = agent.sample_actions(observation, chunks, delay=delay, only_base_actions=(i % 2 == 0))
        jax.block_until_ready(actions)
        prefix = np.asarray(actions)[-args.rtc_delay_steps:] if args.rtc_delay_steps and i > 0 else None
        logging.info('Warmup %d: VLA %.1f ms; edit/select %.1f ms', i, (sampled-start)*1000,
                     (time.monotonic()-sampled)*1000)
        if i >= 4 and args.rtc_delay_steps and sampled-start > args.rtc_delay_steps/args.control_hz:
            raise RuntimeError('Multi-candidate inference exceeds prefix budget. Lower --control-hz or --candidates')
    return agent


def main(argv=None):
    args = parse_args(argv)
    # parse_args imports OpenPI modules which may install a WARNING-level handler.
    # Replace it so warmup timings and benchmark completion remain visible.
    logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s %(levelname)s %(message)s')
    # ROS imports must precede JAX compilation; no node registration or SDK here.
    if not args.benchmark_only and args.mode != 'offline':
        from realman_inference_tape_trainingrtc import load_ros_client
        load_ros_client(args.ros_python_path)
    import jax
    from expo_ft.realman.build import build_agent
    from expo_ft.realman.replay import Episode, RealmanReplay
    from expo_ft.realman.learning import update_from_replay
    from expo_ft.realman.checkpoints import create_manager, save as save_checkpoint, restore as restore_checkpoint, fingerprint

    output = pathlib.Path(args.output_dir).resolve()
    if not args.benchmark_only and output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f'{output} already contains a run; use a new output directory or --resume')
    args.replay_semantics_version = 3  # Continuous execution across gripper transitions.
    args.sft_fingerprint = fingerprint(args.checkpoint)
    logging.info('Building RL learner: checkpoint=%s, replan_steps=%d, delay=%d, control_hz=%.1f',
                 args.checkpoint, args.replan_steps, args.rtc_delay_steps, args.control_hz)
    logging.info('Training: batch_size=%d, UTD=%d, calls_per_episode=%d (%d critic steps), train_base_actor=%s',
                 args.batch_size, args.utd_ratio, args.updates_per_episode,
                 args.utd_ratio * args.updates_per_episode, args.train_base_actor)
    logging.info('Candidates: rollout base=%d, edit=%d; backup filter=%d seeds, 1 survivor, 1 edit',
                 args.candidates, args.edit_candidates, args.filter_candidates)
    agent = build_agent(args)
    manager = None
    if not args.benchmark_only:
        output.mkdir(parents=True, exist_ok=True)
        saved_args = output / 'config.json'
        if args.resume and saved_args.exists():
            old = json.loads(saved_args.read_text())
            for key in ('checkpoint', 'train_base_actor', 'replan_steps', 'rtc_delay_steps', 'candidates', 'edit_candidates', 'edit_scale', 'denoising_steps', 'control_hz', 'prompt', 'replay_semantics_version', 'sft_fingerprint'):
                if old.get(key) != vars(args)[key]:
                    raise ValueError(f'Resume config mismatch: {key}')
        manager = create_manager(output / 'checkpoints', args.train_base_actor)
        if args.resume:
            if manager.latest_step() is None:
                raise ValueError('No complete RL checkpoint to resume')
            agent = restore_checkpoint(manager, agent.replace(_infer_cache=None)).cache_infer_params()
        else:
            saved_args.write_text(json.dumps(vars(args), indent=2))
    try:
        logging.info('Starting inference warmup; first calls include JIT compilation (no hardware connected)')
        agent = warmup(agent, args)
        if args.benchmark_only:
            logging.info('Benchmark passed; no hardware connected')
            return
        replay = RealmanReplay(agent.actor, args.replay_capacity, args.replan_steps,
                               args.rtc_delay_steps, seed=args.seed)
        episode_dir = output / 'episodes'
        episode_dir.mkdir(exist_ok=True)
        completed_local = 0
        for directory in [*args.replay_dir, str(episode_dir)]:
            for path in sorted(pathlib.Path(directory).glob('*.npz')):
                episode = Episode.load(path)
                replay.add(episode)
                if path.parent == episode_dir and not episode.aborted and episode.actions:
                    completed_local += 1
        logging.info('Replay: %d transitions, %d valid critic windows', replay.size, len(replay.eligible()))

        def update(count):
            nonlocal agent
            for _ in range(count):
                with jax.sharding.Mesh(agent.actor.mesh.devices, agent.actor.mesh.axis_names):
                    agent, metrics = update_from_replay(agent, replay, batch_size=args.batch_size,
                                                        utd_ratio=args.utd_ratio)
                if metrics is None:
                    logging.info('No valid execution windows yet; skipping updates')
                    break
                logging.info('Update call complete: critic_step=%d, edit_step=%d, UTD=%d',
                             int(agent.critic.step), int(agent.edit_actor.step), args.utd_ratio)
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps({'update': int(agent.critic.step),
                                             'edit_step': int(agent.edit_actor.step),
                                             'vla_step': int(agent.actor_train_state.step),
                                             'batch_size': args.batch_size,
                                             'filter_candidates': args.filter_candidates,
                                             **{k: float(v) for k, v in metrics.items()}}) + '\n')

        checkpoint_step = int(manager.latest_step() or 0)
        if args.mode == 'offline':
            if not replay.eligible():
                raise ValueError('Offline training requires labelled episodes with complete execution windows')
            update(args.offline_updates)
            save_checkpoint(manager, agent.replace(_infer_cache=None), checkpoint_step+1)
            return
        from expo_ft.realman.environment import RealmanEnv
        from expo_ft.realman.rollout import RTCPlanner, collect_episode, PlanningTimeout
        env = RealmanEnv(args)
        try:
            input('模型已就绪。确认机械臂周围可运动，按 Enter 连接设备；Ctrl-C 退出。')
            env.connect()
            for number in range(args.episodes):
                env.diagnostic_episode = number + 1
                env.diagnostic_step = None
                input('按 Enter 回到采集初始姿态；Ctrl-C 退出。')
                env.reset()
                input('摆好胶带后按 Enter 开始。本轮：1+Enter 成功，2+Enter 失败结束。')
                use_rl = args.mode == 'rl' and completed_local >= args.warmup_episodes and int(agent.critic.step) >= args.min_critic_updates
                logging.info('Episode %d: %s', number+1, 'VLA + edit + Q' if use_rl else 'RTC-SFT collection')
                planner = RTCPlanner(agent, args.replan_steps, args.rtc_delay_steps, only_base_actions=not use_rl)
                path = episode_dir / f'{time.time_ns()}.npz'
                try:
                    ep = collect_episode(env, planner, control_hz=args.control_hz,
                                         max_steps=args.max_episode_steps, timeout_ms=args.control_timeout_ms)
                except PlanningTimeout as exc:
                    # collect_episode already stopped the arm and drained the worker.
                    env.last_episode.save(path)
                    agent = planner.agent
                    logging.warning('Episode aborted: %s. Press Enter at the next prompt to restart.', exc)
                    continue
                except BaseException:
                    if getattr(env, 'last_episode', None) is not None:
                        env.last_episode.save(path)
                    raise
                ep.save(path)
                if ep.end_reason == 'workspace_rejected':
                    logging.warning('Workspace rejected an unexecuted action; episode truncated. '
                                    'Training will continue, then wait for your confirmation to reset the arm.')
                agent = planner.agent
                replay.add(ep)
                if ep.actions:
                    completed_local += 1
                logging.info('Saved %s: %d actions, success=%s, truncated=%s, reason=%s',
                             path, len(ep.actions), ep.success, ep.truncated, ep.end_reason)
                if args.mode == 'rl':
                    update(args.updates_per_episode)
                checkpoint_step += 1
                save_checkpoint(manager, agent.replace(_infer_cache=None), checkpoint_step)
                manager.wait_until_finished()
        finally:
            env.close()
    finally:
        if manager is not None:
            manager.close()


if __name__ == '__main__':
    main()
