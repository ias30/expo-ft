"""Build the existing EXPO learner around the native RealMan actor."""
import dataclasses
import pathlib
import json
import flax.nnx as nnx
import jax
import numpy as np
from openpi.training import config as configs, sharding, weight_loaders
from configs.model.realman_trainingrtc_rl import get_config
from expo_ft.realman.native_actor import NativeRealmanActor, realman_augmentation
from expo_ft.agents.alg.realtime_expo_ft import RealTimeEXPOFTLearner


def build_agent(args):
    checkpoint = pathlib.Path(args.checkpoint).resolve()
    metadata = checkpoint / '_CHECKPOINT_METADATA'
    if not metadata.is_file() or not json.loads(metadata.read_text()).get('commit_timestamp_nsecs'):
        raise ValueError('SFT checkpoint must be a completed checkpoint root (not /params)')
    config = configs.get_config('pi05_realman_tape2_rtc')
    config = dataclasses.replace(
        config, weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint / 'params')),
        data=dataclasses.replace(config.data, assets=configs.AssetsConfig(
            assets_dir=str(checkpoint / 'assets'), asset_id='realman_tape2')),
        ema_decay=None,
    )
    if not (checkpoint / 'assets/realman_tape2/norm_stats.json').is_file():
        raise FileNotFoundError('Checkpoint normalization assets missing; refuse unnormalized RL')
    if not args.train_base_actor:
        # Empty optimizer state for the frozen VLA; critic/edit still use Adam.
        config = dataclasses.replace(config, freeze_filter=nnx.All(nnx.Param))
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    actor, state, _ = NativeRealmanActor.initialize(
        config, mesh, jax.random.key(args.seed), resume=False, default_prompt=args.prompt,
        data_sharding=data_sharding, replicated_sharding=replicated, freeze_pi05_encoder=True,
    )
    actor.num_steps = args.denoising_steps
    settings = dict(get_config())
    for name in list(settings):
        if name.startswith('pi05_') or name in ('model_cls', 'freeze_pi05_encoder', 'valids_keep_terminal_windows'):
            settings.pop(name)
    settings.update(N=args.candidates, n_edit_samples=args.edit_candidates,
                    filter_N=args.filter_candidates,
                    train_base_actor=args.train_base_actor, edit_scale=args.edit_scale)
    agent = RealTimeEXPOFTLearner.create(
        args.seed, np.zeros((1, 224, 224, 6), np.float32), np.zeros(7, np.float32), np.zeros(7, np.float32),
        actor=actor, actor_train_state=state, target_actor_params=actor.get_params(state),
        action_horizon=config.model.action_horizon, mesh=mesh, freeze_encoder=True,
        data_sharding=data_sharding, replicated_sharding=replicated,
        default_prompt=args.prompt, replan_steps=args.replan_steps, delay=args.rtc_delay_steps,
        resize_size=224, critic_camera_keys=('base_0_rgb', 'right_wrist_0_rgb'),
        edit_action_xyzg=False, **settings,
    )
    return agent.replace(data_augmentation_fn=realman_augmentation).cache_infer_params()
