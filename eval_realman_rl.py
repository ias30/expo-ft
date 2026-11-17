#!/usr/bin/env python3
"""Evaluate a specific RealMan RL checkpoint without updates or checkpoint writes."""
import argparse
import json
import logging
import pathlib
import time

from train_realman_rl import parse_args as training_args, warmup


def evaluation_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rl-checkpoint', required=True, help='Completed RL step directory, e.g. logs/run/checkpoints/20')
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-episode-steps', type=int)
    p.add_argument('--output-dir', help='New directory for evaluation episodes; defaults to a separate timestamped directory')
    p.add_argument('--benchmark-only', action='store_true', help='Restore/JIT the RL policy without connecting hardware')
    options = p.parse_args(argv)
    checkpoint = pathlib.Path(options.rl_checkpoint).expanduser().resolve()
    if not checkpoint.name.isdecimal() or checkpoint.parent.name != 'checkpoints':
        p.error('--rl-checkpoint must be a numbered directory under checkpoints/')
    metadata = checkpoint / '_CHECKPOINT_METADATA'
    if not metadata.is_file() or not json.loads(metadata.read_text()).get('commit_timestamp_nsecs'):
        p.error('RL checkpoint is missing or incomplete')
    config = json.loads((checkpoint.parent.parent / 'config.json').read_text())
    if config.get('replay_semantics_version') != 3:
        p.error('Evaluation requires continuous-execution semantics version 3')
    args = training_args([])
    vars(args).update(config)
    args.mode, args.resume = 'eval', False
    args.episodes = options.episodes
    args.benchmark_only = options.benchmark_only
    if options.max_episode_steps is not None:
        args.max_episode_steps = options.max_episode_steps
    if min(args.episodes, args.max_episode_steps) < 1:
        p.error('Episode and step counts must be positive')
    args.rl_checkpoint = str(checkpoint)
    args.output_dir = str(pathlib.Path(options.output_dir).expanduser().resolve() if options.output_dir else
                          checkpoint.parent.parent / 'evaluations' / f'checkpoint-{checkpoint.name}-{time.time_ns()}')
    # Diagnostic files belong to this evaluation, not the original training run.
    if args.sdk_debug_dir:
        args.sdk_debug_dir = str(pathlib.Path(args.output_dir) / 'sdk')
    return args


def main(argv=None):
    args = evaluation_args(argv)
    logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s %(levelname)s %(message)s')
    if not args.benchmark_only:
        from realman_inference_tape_trainingrtc import load_ros_client
        load_ros_client(args.ros_python_path)
    from expo_ft.realman.build import build_agent
    from expo_ft.realman.checkpoints import create_manager, restore, fingerprint
    if fingerprint(args.checkpoint) != args.sft_fingerprint:
        raise ValueError('Base SFT checkpoint fingerprint differs from the RL training configuration')
    output = pathlib.Path(args.output_dir)
    if not args.benchmark_only and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Evaluation output must be empty: {output}')
    checkpoint = pathlib.Path(args.rl_checkpoint)
    agent = build_agent(args)
    manager = create_manager(checkpoint.parent, args.train_base_actor, read_only=True)
    try:
        agent = restore(manager, agent, step=int(checkpoint.name))
    finally:
        manager.close()
    logging.info('Evaluating %s: critic_step=%d edit_step=%d; no training updates',
                 checkpoint, int(agent.critic.step), int(agent.edit_actor.step))
    agent = warmup(agent, args)
    if args.benchmark_only:
        logging.info('RL evaluation benchmark passed; no hardware connected')
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    from expo_ft.realman.environment import RealmanEnv
    from expo_ft.realman.rollout import RTCPlanner, collect_episode
    env = RealmanEnv(args)
    try:
        input('评估模型已就绪。确认机械臂周围可运动，按 Enter 连接设备；Ctrl-C 退出。')
        env.connect()
        for number in range(args.episodes):
            env.diagnostic_episode, env.diagnostic_step = number + 1, None
            input('按 Enter 回到采集初始姿态；Ctrl-C 退出。')
            env.reset()
            input('摆好胶带后按 Enter 开始评估。本轮：1+Enter 成功，2+Enter 失败结束。')
            logging.info('Evaluation episode %d: VLA + edit + Q (fixed checkpoint %s)', number + 1, checkpoint.name)
            planner = RTCPlanner(agent, args.replan_steps, args.rtc_delay_steps, only_base_actions=False)
            path = output / f'episode-{number + 1}.npz'
            try:
                episode = collect_episode(env, planner, control_hz=args.control_hz,
                    max_steps=args.max_episode_steps, timeout_ms=args.control_timeout_ms)
            except BaseException:
                if getattr(env, 'last_episode', None) is not None:
                    env.last_episode.save(path)
                raise
            agent = planner.agent  # Advance sampling RNG; weights remain fixed.
            episode.save(path)
            logging.info('Evaluation saved: %s; success=%s actions=%d truncated=%s',
                         path, episode.success, len(episode.actions), episode.truncated)
    finally:
        env.close()


if __name__ == '__main__':
    main()
