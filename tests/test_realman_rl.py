"""No SDK connections: physical action invariants, episode labels, and RL integration."""
import dataclasses
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/openpi_rtc/src'),
               str(ROOT / 'third_party/openpi_rtc/packages/openpi-client/src')]
import jax
import numpy as np
import pytest
from openpi.shared.normalize import NormStats
from openpi import transforms
from expo_ft.realman.actions import ActionCodec
from expo_ft.realman.replay import Episode, RealmanReplay
from expo_ft.realman.rollout import collect_episode


def codec():
    stats = {k: NormStats(mean=np.arange(7)*.01, std=np.ones(7)*.4,
                         q01=-np.ones(7)*.5, q99=np.ones(7)*1.5) for k in ('state', 'actions')}
    return ActionCodec(stats)


def obs(value=0.):
    return dict(state=np.array([value]*6+[1.], np.float32), prompt='pick up the tape',
                images={k: np.full((3, 224, 224), 10, np.uint8) for k in ('cam_high', 'cam_right_wrist')})


def test_codec_matches_training_transforms_and_rebases():
    c = codec()
    state = obs(.7)['state']
    actions = np.array([[1.2]*6+[0.], [1.3]*6+[1.]], np.float32)
    data = transforms.DeltaActions(np.array([True]*6+[False]))(dict(state=state.copy(), actions=actions.copy()))
    expected = transforms.Normalize(c.stats, use_quantiles=True)(data)['actions']
    encoded = c.encode(actions, state)
    np.testing.assert_allclose(encoded, expected, atol=1e-6)
    new_state = obs(1.1)['state']
    rebased = jax.jit(c.rebase)(encoded, c.normalize(state, 'state'), c.normalize(new_state, 'state'))
    np.testing.assert_allclose(c.decode(rebased, new_state), actions, atol=1e-6)
    np.testing.assert_allclose(c.decode(encoded, state), actions, atol=1e-6)


class FakeActor:
    model_config = SimpleNamespace(action_horizon=4, action_dim=32)
    action_codec = codec()
    def process_raw_inputs(self, o):
        return dict(state=np.pad(np.asarray(self.action_codec.normalize(o['state'], 'state')), (0, 25))[None],
                    image={k: np.zeros((1, 8, 8, 3), np.float32) for k in ('base_0_rgb', 'right_wrist_0_rgb', 'left_wrist_0_rgb')},
                    image_mask={k: np.ones(1, bool) for k in ('base_0_rgb', 'right_wrist_0_rgb', 'left_wrist_0_rgb')},
                    tokenized_prompt=np.full((1, 8), round(float(o['state'][0])*100), int), tokenized_prompt_mask=np.ones((1, 8), bool))


def make_episode(n=6, success=True):
    ep = Episode(obs(0))
    for t in range(n):
        ep.append(np.array([t+.5]*6+[1.], np.float32), obs(t+1))
    ep.finish(success)
    return ep


def test_replay_actions_use_one_anchor_and_terminal_reward(tmp_path):
    ep = make_episode(6)
    ep.save(tmp_path/'episode.npz')
    ep = Episode.load(tmp_path/'episode.npz')
    rb = RealmanReplay(FakeActor(), capacity=20, replan_steps=2, delay=1, seed=9)
    rb.add(ep)
    batch = rb.sample(80)
    t = np.asarray(rb.actor.action_codec.unnormalize(batch['state'], 'state'))[:, 0]
    absolute = rb.actor.action_codec.decode(batch['actions'], rb.actor.action_codec.unnormalize(batch['state'], 'state'))
    np.testing.assert_allclose(np.asarray(absolute)[:, 0, 0], t+.5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(absolute)[:, 1, 0], np.minimum(t+1.5, 5.5), atol=1e-5)
    terminal = t > 3.5
    assert terminal.any()
    np.testing.assert_allclose(batch['rewards'], np.where(t > 4.5, 1., np.where(terminal, .99, 0)), atol=1e-6)
    np.testing.assert_array_equal(batch['masks'], ~terminal)
    assert all(t <= 2 for _, t in rb.eligible(success_only=True))


def test_replay_pause_and_aborted_data_excluded():
    ep = make_episode()
    ep.pauses[2] = True
    rb = RealmanReplay(FakeActor(), capacity=12, replan_steps=2, delay=1)
    rb.add(ep)
    assert [t for _, t in rb.eligible()] == [0, 1, 2, 3, 4, 5]
    invalid = make_episode()
    invalid.aborted = True
    rb.add(invalid)
    assert rb.size == 6
    rb.add(make_episode(8))
    assert rb.size == 8


def fake_env(labels, paused=False):
    env = SimpleNamespace(get_observation=Mock(return_value=obs()), stop=Mock(),
                          get_info_for_step=Mock(side_effect=labels),
                          step=Mock(return_value=(obs()['state'], paused)))
    planner = SimpleNamespace(queue=[], pending=None, action=Mock(return_value=obs()['state']),
                              invalidate=Mock(), close=Mock())
    return env, planner


CONTINUE = (False, False, 0., 1.)
SUCCESS = (True, True, 1., 0.)


def test_done_never_dispatches_even_after_inference():
    for labels in ([SUCCESS], [CONTINUE, SUCCESS]):
        env, planner = fake_env(labels)
        ep = collect_episode(env, planner)
        assert ep.success and not ep.actions
        env.step.assert_not_called()
        env.stop.assert_called_once()


def test_manual_reward_labels_last_actual_command_and_pause_discards_plan():
    env, planner = fake_env([CONTINUE, CONTINUE, SUCCESS], paused=True)
    ep = collect_episode(env, planner)
    assert ep.success and len(ep.actions) == 1 and ep.pauses == [True]
    env.step.assert_called_once()
    planner.invalidate.assert_called_once()


def test_fault_stops_and_marks_episode_aborted():
    env, planner = fake_env([CONTINUE, CONTINUE])
    env.step.side_effect = RuntimeError('workspace rejected')
    with pytest.raises(RuntimeError, match='workspace'):
        collect_episode(env, planner)
    assert env.last_episode.aborted
    env.stop.assert_called_once()


@pytest.mark.parametrize('executed', [0, 4])
def test_workspace_rejection_truncates_without_recording_command(tmp_path, executed):
    from expo_ft.realman.workspace import WorkspaceMotionRejected
    env, planner = fake_env([])
    env.get_info_for_step = Mock(return_value=CONTINUE)
    env.get_observation.side_effect = [obs(i) for i in range(executed+1)]
    env.step.side_effect = [(obs()['state'], False)] * executed + [WorkspaceMotionRejected('predicted_target')]
    ep = collect_episode(env, planner, sleep=lambda _: None)
    assert ep.finished and ep.truncated and not ep.success and not ep.aborted
    assert ep.end_reason == 'workspace_rejected'
    assert len(ep.actions) == executed and len(ep.observations) == executed + 1
    assert env.step.call_count == executed + 1
    assert ep.observations[-1]['state'][0] == executed
    env.stop.assert_called_once()
    planner.close.assert_called_once()
    ep.save(tmp_path/'workspace.npz')
    loaded = Episode.load(tmp_path/'workspace.npz')
    assert loaded.truncated and loaded.end_reason == 'workspace_rejected'
    replay = RealmanReplay(FakeActor(), capacity=20, replan_steps=2, delay=1)
    replay.add(loaded)
    assert [t for _, t in replay.eligible()] == list(range(max(0, executed-2)))


def test_workspace_cleanup_failure_is_aborted():
    from expo_ft.realman.workspace import WorkspaceMotionRejected
    env, planner = fake_env([CONTINUE, CONTINUE])
    env.step.side_effect = WorkspaceMotionRejected('predicted_target')
    env.stop.side_effect = RuntimeError('stop failed')
    with pytest.raises(RuntimeError, match='stop failed'):
        collect_episode(env, planner)
    assert env.last_episode.aborted and not env.last_episode.truncated
    planner.close.assert_called_once()


def test_native_rtc_rl_update_and_restore(monkeypatch, tmp_path):
    """Real Pi0 modules at reduced width + real critic/edit/filter optimizers and Orbax."""
    sys.path.insert(0, str(ROOT/'third_party/openpi_rtc'))
    from scripts.train_accumulation_test import make_small_pi05_state
    from openpi.training import sharding
    from expo_ft.realman.native_actor import NativeRealmanActor
    from expo_ft.agents.alg.realtime_expo_ft import RealTimeEXPOFTLearner, save_checkpoint, restore_checkpoint
    import orbax.checkpoint as ocp
    config, state = make_small_pi05_state(monkeypatch, dtype='float32')
    from openpi.training.config import AssetsConfig
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, max_token_len=128), data=dataclasses.replace(config.data, assets=AssetsConfig(
        assets_dir=str(ROOT/'third_party/openpi_rtc/assets/pi05_realman_tape2'), asset_id='realman_tape2')))
    mesh = sharding.make_mesh(1)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    actor = NativeRealmanActor(train_config=config, mesh=mesh,
        train_state_sharding=sharding.fsdp_sharding(state, mesh), data_sharding=data,
        replicated_sharding=replicated, default_prompt='pick up the tape', freeze_pi05_encoder=True)
    actor.num_steps = 2
    # Compare mixed-delay sampling to a pure boot with identical noise.
    from expo_ft.realman.native_actor import native_sample
    import jax.numpy as jnp
    packed = actor.process_raw_inputs(obs(.7))
    packed = jax.tree.map(lambda x: jnp.repeat(x, 2, axis=0), packed)
    fixed_noise = jax.random.normal(jax.random.key(999), (2, 1, 2, 32))
    prefix = jnp.full((2, 1, 32), .25)
    mixed = native_sample(state, packed, jax.random.key(4), prefix, num_samples=1,
                          num_steps=2, noise=fixed_noise, prefix_lengths=jnp.array([0, 1]))
    boot = native_sample(state, packed, jax.random.key(4), prefix[:, :0], num_samples=1,
                         num_steps=2, noise=fixed_noise)
    np.testing.assert_allclose(mixed[0], boot[0], atol=1e-6)
    np.testing.assert_array_equal(mixed[1, :1], prefix[1])
    agent = RealTimeEXPOFTLearner.create(42, np.zeros((1, 224, 224, 6), np.float32), np.zeros(7), np.zeros(7),
        actor=actor, actor_train_state=state, target_actor_params=state.params,
        action_horizon=2, mesh=mesh, freeze_encoder=True, replan_steps=1, delay=1,
        data_sharding=data, replicated_sharding=replicated, resize_size=224,
        N=2, n_edit_samples=2, filter_N=32, filter_n_edit=1, num_qs=2, num_min_qs=2,
        encoder_stage_sizes=(1,), encoder_num_filters=32, hidden_dims=(16,), latent_dim_image=16,
        latent_dim_state=8, actor_drop=0., train_base_actor=False, use_full_augmentation=False,
        critic_camera_keys=('base_0_rgb','right_wrist_0_rgb'), filter_add_delayed_obs=True)
    initial = obs(.7)
    prefix = np.array([[.9]*6+[0.]], np.float32)
    chunks, agent = agent.sample_pre_cache(initial, prefix)
    np.testing.assert_allclose(chunks[:, :1], np.broadcast_to(prefix, (2, 1, 7)), atol=2e-5)
    base, agent, _ = agent.sample_actions(obs(.8), chunks, delay=1, only_base_actions=True)
    np.testing.assert_allclose(base, chunks[0, 1:], atol=2e-5)
    actions, agent, _ = agent.sample_actions(obs(.8), chunks, delay=1)
    assert actions.shape == (1, 7) and np.isfinite(actions).all()
    # Trace the actual backup sampler: 32 scored seeds must still produce only
    # one VLA denoising candidate per observation, independently of rollout N=2.
    denoise_shapes = []
    original_sampler = actor.sample_training_actions_with_prefix
    def sample_survivor(*args, **kwargs):
        assert kwargs['num_samples'] == 1
        denoise_shapes.append(kwargs['noise'].shape)
        assert kwargs['noise'].shape[1:] == (1, config.model.action_horizon, config.model.action_dim)
        return original_sampler(*args, **kwargs)
    monkeypatch.setattr(actor, 'sample_training_actions_with_prefix', sample_survivor)
    rb = RealmanReplay(actor, capacity=8, replan_steps=1, delay=1)
    rb.add(make_episode(4))
    # Include a gripper pause: a single training batch must support delay 0 and 1.
    ep = rb.episodes[0]
    ep.pauses[1] = True
    ep.actions[1][6] = 0
    for i in range(2, len(ep.observations)):
        ep.observations[i]['state'][6] = 0
    for i in range(2, len(ep.actions)):
        ep.actions[i][6] = 0
    batch = rb.sample(8)
    assert set(batch['bootstrap_delay']) == {0, 1}
    assert not np.array_equal(batch['tokenized_prompt'], batch['next_delayed_tokenized_prompt'])
    before = jax.tree.map(lambda x: np.asarray(x).copy(), agent.critic.params)
    with mesh:
        updated, metrics = agent.update(agent, jax.device_put(batch, data), 1)
    jax.block_until_ready(metrics)
    assert int(updated.critic.step) == 1 and int(updated.edit_actor.step) == 1
    assert updated.actor_train_state is agent.actor_train_state
    assert updated.target_actor_params is agent.target_actor_params
    assert all(np.isfinite(np.asarray(x)).all() for x in metrics.values())
    assert any(not np.array_equal(x, y) for x,y in zip(jax.tree.leaves(before),jax.tree.leaves(updated.critic.params)))
    # Success-only VLA update also exercises native prefix BC compatibility.
    from expo_ft.realman.learning import update_from_replay
    trainable = updated.replace(train_base_actor=True, actor_success_only=True)
    sample_calls = []
    def sample(size, success_only=False):
        sample_calls.append((size, success_only))
        return rb.sample(size, success_only=success_only)
    replay = SimpleNamespace(sample=sample)
    with mesh:
        updated, metrics = update_from_replay(trainable, replay, batch_size=2, utd_ratio=20)
    jax.block_until_ready(metrics)
    assert int(updated.actor_train_state.step) == 1
    assert int(updated.critic.step) == int(updated.filter_critic.step) == 21
    assert int(updated.edit_actor.step) == int(updated.temp.step) == 2
    assert sample_calls == [(2, False)] * 20 + [(2, True)]
    assert metrics['critic_steps_this_call'] == 20
    assert all(np.isfinite(np.asarray(x)).all() for x in metrics.values())
    # Memory-constrained runs use batch=1; never squeeze away the batch axis.
    single = updated.replace(train_base_actor=False)
    with mesh:
        updated, metrics = update_from_replay(single, rb, batch_size=1, utd_ratio=1)
    jax.block_until_ready(metrics)
    assert all(np.isfinite(np.asarray(x)).all() for x in metrics.values())
    assert int(updated.critic.step) == 22 and int(updated.edit_actor.step) == 3
    assert updated.actor_train_state is single.actor_train_state
    assert updated.target_actor_params is single.target_actor_params
    # No successful windows: honor the VLA training preference but skip its BC step.
    pending_bc = updated.replace(train_base_actor=True, actor_success_only=True)
    no_success = SimpleNamespace(sample=lambda size, success_only=False: None if success_only else rb.sample(size))
    with mesh:
        updated, metrics = update_from_replay(pending_bc, no_success, batch_size=1, utd_ratio=2)
    assert updated.train_base_actor and int(updated.actor_train_state.step) == 1
    assert int(updated.critic.step) == int(updated.filter_critic.step) == 24
    assert int(updated.edit_actor.step) == int(updated.temp.step) == 4
    updated = updated.replace(train_base_actor=False)
    assert denoise_shapes and agent.filter_N == 32 and agent.N == 2
    updated = updated.replace(_infer_cache=None)
    from expo_ft.realman.checkpoints import create_manager, save, restore, RL_FIELDS
    assert 'actor_train_state' not in RL_FIELDS and 'target_actor_params' not in RL_FIELDS
    with create_manager(tmp_path/'light', False) as light:
        save(light, updated, 1)
        reloaded = restore(light, updated)
        assert int(reloaded.critic.step) == 24
        assert reloaded.actor_train_state is updated.actor_train_state
        for name in RL_FIELDS:
            for x, y in zip(jax.tree.leaves(getattr(reloaded, name)), jax.tree.leaves(getattr(updated, name))):
                np.testing.assert_array_equal(jax.random.key_data(x) if hasattr(x, 'dtype') and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) else x,
                                              jax.random.key_data(y) if hasattr(y, 'dtype') and jax.dtypes.issubdtype(y.dtype, jax.dtypes.prng_key) else y)
    with ocp.CheckpointManager(tmp_path/'rl', item_handlers={k: ocp.PyTreeCheckpointHandler() for k in ('agent','params')},
                              options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False)) as manager:
        save_checkpoint(manager, updated, 1)
        restored = restore_checkpoint(manager, updated)
        assert int(restored.critic.step) == 24
        assert int(restored.edit_actor.step) == 4
        assert int(restored.filter_critic.step) == 24 and int(restored.temp.step) == 4
        for x, y in zip(jax.tree.leaves(restored.target_critic.params), jax.tree.leaves(updated.target_critic.params)):
            np.testing.assert_array_equal(x, y)


def test_pause_short_windows_tokens_and_discounts():
    from expo_ft.agents.alg.batch_utils import prepare_actor_sampling_batch, prepare_actor_sampling_batch_delayed
    ep = make_episode(6)
    ep.pauses[2] = True
    rb = RealmanReplay(FakeActor(), capacity=12, replan_steps=2, delay=1)
    rb.add(ep)
    batch = rb.sample(100)
    t = np.rint(np.asarray(rb.actor.action_codec.unnormalize(batch['state'], 'state'))[:, 0]).astype(int)
    # Every start through the pause is trainable, including the one-step grasp.
    assert set(t) == {0, 1, 2, 3, 4, 5}
    grasp = t == 2
    assert np.all(batch['window_length'][grasp] == 1)
    np.testing.assert_allclose(batch['bootstrap_discount'][grasp], .99)
    ending_at_pause = (t == 1) | grasp
    assert np.all(batch['bootstrap_delay'][ending_at_pause] == 0)
    assert np.all(batch['masks'][ending_at_pause] == 1)  # pause is not terminal
    batch['next_states'] = batch['next_state']
    batch['next_delayed_states'] = batch['next_delayed_state']
    for helper, prefix in ((prepare_actor_sampling_batch, 'next_'),
                           (prepare_actor_sampling_batch_delayed, 'next_delayed_')):
        inputs = helper(batch)
        expected_state = rb.actor.action_codec.unnormalize(batch[prefix+'state'], 'state')[:, 0]
        np.testing.assert_allclose(inputs['tokenized_prompt'][:, 0], np.rint(expected_state*100), atol=1e-3)
        np.testing.assert_array_equal(inputs['tokenized_prompt_mask'], batch[prefix+'tokenized_prompt_mask'])
    # Value can propagate backwards through the grasp: s2 -> s3 -> s5 (terminal).
    assert any(ep2 is ep and start == 2 for ep2, start in rb.eligible())
    # Solve the sampled deterministic Bellman chain backwards. Grasp-adjacent
    # states must receive the terminal success value, not bootstrap from a hole.
    values = {len(ep.actions): 0.0}
    for start in range(len(ep.actions)-1, -1, -1):
        i = np.flatnonzero(t == start)[0]
        end = start + int(batch['window_length'][i])
        values[start] = float(batch['rewards'][i] + batch['bootstrap_discount'][i] * batch['masks'][i] * values[end])
    assert all(values[start] > 0 for start in range(len(ep.actions)))
    np.testing.assert_allclose(values[0], .99**5, atol=1e-6)


def test_critic_gripper_transition_preserves_following_actions():
    from expo_ft.realman.actions import critic_window
    c = codec()
    state = obs(0)['state']
    raw = np.array([[.1]*6+[.4], [.2]*6+[.2], [.3]*6+[.9]], np.float32)
    encoded = c.encode(raw, state)[None]
    result = critic_window(c, encoded.reshape(1, -1), c.normalize(state, 'state')[None]).reshape(1, 3, 7)
    physical = c.decode(result, state)
    np.testing.assert_allclose(physical[0, :, 6], [1, 0, 1], atol=1e-6)
    np.testing.assert_allclose(physical[0, :, :6], raw[:, :6], atol=1e-6)
    changed = encoded.at[:, 2, :].set(1000)
    assert not np.array_equal(critic_window(c, changed.reshape(1, -1), c.normalize(state, 'state')[None]), result.reshape(1, -1))
    masked = critic_window(c, encoded.reshape(1, -1), c.normalize(state, 'state')[None], np.array([2]))
    np.testing.assert_array_equal(critic_window(c, changed.reshape(1, -1), c.normalize(state, 'state')[None], np.array([2])), masked)
    assert masked.reshape(1, 3, 7)[0, 2, 6] < -5
    grad = jax.grad(lambda x: critic_window(c, x, c.normalize(state, 'state')[None]).sum())(encoded.reshape(1, -1))
    assert np.isfinite(grad).all() and float(grad[0, 6]) != 0


def test_planning_timeout_is_recoverable_but_sdk_timeout_is_not():
    from expo_ft.realman.rollout import PlanningTimeout
    env, planner = fake_env([CONTINUE])
    planner.action.side_effect = PlanningTimeout('inference late')
    with pytest.raises(PlanningTimeout):
        collect_episode(env, planner)
    assert env.last_episode.aborted
    env.stop.assert_called_once()
    env, planner = fake_env([CONTINUE, CONTINUE])
    env.step.side_effect = TimeoutError('gripper timeout')
    with pytest.raises(TimeoutError) as error:
        collect_episode(env, planner)
    assert not isinstance(error.value, PlanningTimeout)


@pytest.mark.parametrize('initial', [0, 1])
def test_imported_demo_matches_continuous_online_gripper(initial):
    from expo_ft.realman.demos import labelled_episodes
    from realman_inference_tape_trainingrtc import RealmanInference
    commands = [.5, .2, .4, .8, .6, .2, .8]
    rows = []
    target = initial
    first = obs(0)
    first['state'][6] = initial
    online = Episode(first)
    for t, command in enumerate(commands + [.5]):
        o = obs(t)
        o['state'][6] = target
        action = np.array([t + .5]*6 + [command], np.float32)
        rows.append(dict(episode_index=4, **{'observation.state': o['state'], 'action': action,
            **{f'observation.images.{k}': v for k, v in o['images'].items()}}))
        if t < len(commands):
            next_target = RealmanInference._map_gripper_value_to_state(float(action[6]), target)
            following = obs(t + 1)
            following['state'][6] = next_target
            executed = action.copy()
            executed[6] = next_target
            online.append(executed, following, reset_plan=False)
            target = next_target
    [(index, imported)] = list(labelled_episodes(rows, {'4': True}, first['prompt']))
    assert index == 4 and imported.success
    assert imported.pauses == online.pauses and not any(imported.pauses)
    np.testing.assert_array_equal(imported.actions, online.actions)
    np.testing.assert_array_equal([o['state'] for o in imported.observations],
                                  [o['state'] for o in online.observations])
    rb = RealmanReplay(FakeActor(), capacity=20, replan_steps=2, delay=1)
    rb.add(imported)
    batch = rb.sample(100)
    starts = np.rint(np.asarray(rb.actor.action_codec.unnormalize(batch['state'], 'state'))[:, 0]).astype(int)
    for i, start in enumerate(starts):
        end = start + batch['window_length'][i]
        assert not any(imported.pauses[start:end-1])
        if imported.pauses[end-1]:
            assert batch['bootstrap_delay'][i] == 0


@pytest.mark.parametrize('last_label', [CONTINUE, SUCCESS, (True, False, 0., 0.)])
@pytest.mark.parametrize('paused', [False, True])
def test_rollout_one_observation_per_transition_and_limit_labels(last_label, paused):
    env, planner = fake_env([CONTINUE, CONTINUE, CONTINUE, CONTINUE, last_label], paused=paused)
    env.get_observation.side_effect = [obs(i) for i in range(3)]
    ep = collect_episode(env, planner, max_steps=2, sleep=lambda _: None)
    assert len(ep.actions) == 2 and len(ep.observations) == 3
    assert env.get_observation.call_count == 3
    assert [o['state'][0] for o in ep.observations] == [0, 1, 2]
    assert [call.args[0]['state'][0] for call in planner.action.call_args_list] == [0, 1]
    assert ep.truncated == (last_label == CONTINUE)
    assert ep.success == (last_label == SUCCESS)
    assert not ep.aborted


def test_truncation_persistence_and_no_false_failure_targets(tmp_path):
    ep = make_episode(6, success=False)
    ep.finish(truncated=True)
    ep.pauses[3] = True
    path = tmp_path/'truncated.npz'
    ep.save(path)
    ep = Episode.load(path)
    assert ep.truncated and not ep.success
    rb = RealmanReplay(FakeActor(), capacity=20, replan_steps=3, delay=1)
    rb.add(ep)
    # Pause before the cutoff still makes start=3 valid, even though 3+r==n.
    assert [t for _, t in rb.eligible()] == [0, 1, 2, 3]
    batch = rb.sample(30)
    assert np.all(batch['masks'] == 1) and np.all(batch['rewards'] == 0)
    assert np.all(batch['valids'] == 1)  # Invalid cutoff rows never reach any optimizer.
    assert not batch['dones'].any()
    ep.finish(False)  # Explicit human failure DOES train terminal windows.
    assert [t for _, t in rb.eligible()] == list(range(6))
    batch = rb.sample(100)
    assert batch['dones'].any() and np.any(batch['masks'] == 0)
    # Older recordings have no truncation field and retain their original labels.
    with np.load(path) as data:
        legacy = {k: data[k] for k in data.files if k != 'truncated'}
    np.savez(tmp_path/'legacy.npz', **legacy)
    assert not Episode.load(tmp_path/'legacy.npz').truncated
    short = make_episode(1, success=False)
    short.finish(truncated=True)
    rb = RealmanReplay(FakeActor(), capacity=20, replan_steps=3, delay=1)
    rb.add(short)
    assert rb.sample(2) is None


def test_edit_loss_masks_q_gradients_after_actual_window(monkeypatch):
    import jax.numpy as jnp
    import expo_ft.agents.alg.realtime_expo_ft as alg
    from expo_ft.realman.actions import critic_window
    monkeypatch.setattr(alg, 'batch_encode', lambda *args, **kwargs: args[2])
    c = codec()
    states = jnp.stack([c.normalize(obs()['state'], 'state')]*2)
    actions = jnp.stack([c.encode(np.array([[.1]*6+[1.]]*3), obs()['state']).reshape(-1)]*2)
    lengths = jnp.array([1, 2])
    # Fixed distribution isolates the Q gradient from entropy and network coupling.
    edit = SimpleNamespace(params=jnp.zeros_like(actions))
    edit.apply_fn = lambda variables, *a, **kw: SimpleNamespace(
        sample=lambda **kw: variables['params'], log_prob=lambda x: jnp.zeros(x.shape[0]))
    edit.apply_gradients = lambda grads: SimpleNamespace(grads=grads)
    native = SimpleNamespace(physical_precache=True,
        critic_actions=lambda a, s, length=None: critic_window(c, a, s, length))
    agent = SimpleNamespace(rng=jax.random.key(42), actor=native, edit_actor=edit,
        batch_encoder=SimpleNamespace(apply_fn=None, params=None),
        data_sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        critic=SimpleNamespace(params=None, apply_fn=lambda variables, o, a, *args, **kw: a.sum(axis=-1)[None]),
        temp=SimpleNamespace(params=None, apply_fn=lambda *args: 1.),
        edit_scale=.1, entropy_scale=0., _apply_edit_xyzg_mask=lambda x: x,
        replace=lambda **kw: SimpleNamespace(**kw))
    updated, info = alg.RealTimeEXPOFTLearner.update_edit_actor(agent, dict(
        observations=jnp.zeros((2, 1)), actions=actions, critic_states=states, window_length=lengths))
    grads = np.asarray(updated.edit_actor.grads).reshape(2, 3, 7)
    assert np.isfinite(float(info['edit_actor_loss']))
    for i, length in enumerate(lengths):
        assert np.any(grads[i, :length] != 0)
        np.testing.assert_array_equal(grads[i, length:], 0)


def test_utd_arguments_and_empty_replay():
    from train_realman_rl import parse_args
    from expo_ft.realman.learning import update_from_replay
    assert parse_args([]).utd_ratio > 0
    assert parse_args(['--utd-ratio', '20']).utd_ratio == 20
    assert parse_args(['--utd-ratio', '1']).utd_ratio == 1
    for value in ('0', '-1'):
        with pytest.raises(SystemExit):
            parse_args(['--utd-ratio', value])
    agent = SimpleNamespace(train_base_actor=False)
    replay = SimpleNamespace(sample=Mock(return_value=None))
    unchanged, metrics = update_from_replay(agent, replay, batch_size=2, utd_ratio=20)
    assert unchanged is agent and metrics is None
    replay.sample.assert_called_once_with(2)


def test_utd_aborts_on_intermediate_nonfinite_metrics():
    from expo_ft.realman.learning import update_from_replay
    agent = SimpleNamespace(train_base_actor=False, data_sharding=None)
    agent.update_critic_only = Mock(return_value=(agent, {'critic_loss': float('nan')}))
    agent.update = Mock()
    replay = SimpleNamespace(sample=Mock(return_value={'actions': np.zeros((2, 7))}))
    with pytest.raises(FloatingPointError, match='UTD step 1/20'):
        update_from_replay(agent, replay, batch_size=2, utd_ratio=20)
    agent.update.assert_not_called()
    replay.sample.assert_called_once_with(2)


def test_filter_candidates_cli_reaches_learner(monkeypatch, tmp_path):
    from train_realman_rl import parse_args
    from expo_ft.realman import build
    assert parse_args([]).filter_candidates == 32
    assert parse_args(['--filter-N', '7']).filter_candidates == 7
    for value in ('0', '-2'):
        with pytest.raises(SystemExit):
            parse_args(['--filter-candidates', value])
    (tmp_path/'_CHECKPOINT_METADATA').write_text('{"commit_timestamp_nsecs": 1}')
    assets = tmp_path/'assets/realman_tape2'
    assets.mkdir(parents=True)
    (assets/'norm_stats.json').write_text('{}')
    # Exercise the real builder's config wiring without loading the full VLA.
    actor = SimpleNamespace(get_params=lambda state: {})
    monkeypatch.setattr(build.NativeRealmanActor, 'initialize', Mock(return_value=(actor, None, None)))
    learner = Mock()
    learner.replace.return_value = learner
    learner.cache_infer_params.return_value = learner
    create = Mock(return_value=learner)
    monkeypatch.setattr(build.RealTimeEXPOFTLearner, 'create', create)
    args = parse_args(['--checkpoint', str(tmp_path), '--filter-candidates', '7',
                      '--candidates', '2', '--edit-candidates', '2'])
    assert build.build_agent(args) is learner
    settings = create.call_args.kwargs
    assert settings['filter_N'] == 7 and settings['filter_n_edit'] == 1
    assert settings['N'] == 2 and settings['n_edit_samples'] == 2
