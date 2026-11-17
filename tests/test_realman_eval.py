"""Evaluation uses an explicit checkpoint and never runs a learner update."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import eval_realman_rl as evaluation


@pytest.fixture
def checkpoint(tmp_path):
    step = tmp_path / 'run' / 'checkpoints' / '20'
    step.mkdir(parents=True)
    (step / '_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs': 1}))
    config = vars(evaluation.training_args([]))
    config.update(replay_semantics_version=3, candidates=2, edit_candidates=2,
                  sft_fingerprint='fp', sdk_debug_dir='logs/old-sdk')
    (step.parent.parent / 'config.json').write_text(json.dumps(config))
    return step


def test_eval_inherits_training_configuration_and_separates_output(checkpoint):
    args = evaluation.evaluation_args(['--rl-checkpoint', str(checkpoint)])
    assert args.candidates == args.edit_candidates == 2
    assert args.episodes == 1 and args.mode == 'eval'
    assert '/evaluations/checkpoint-20-' in args.output_dir
    assert args.sdk_debug_dir == args.output_dir + '/sdk'


def test_incomplete_checkpoint_rejected(checkpoint):
    (checkpoint / '_CHECKPOINT_METADATA').write_text('{}')
    with pytest.raises(SystemExit):
        evaluation.evaluation_args(['--rl-checkpoint', str(checkpoint)])


@pytest.mark.parametrize('benchmark', [False, True])
def test_eval_restores_exact_step_read_only_and_selects_rl(monkeypatch, checkpoint, benchmark):
    agent = SimpleNamespace(critic=SimpleNamespace(step=2000), edit_actor=SimpleNamespace(step=200))
    manager = SimpleNamespace(close=Mock())
    create = Mock(return_value=manager)
    restore = Mock(return_value=agent)
    monkeypatch.setitem(sys.modules, 'expo_ft.realman.checkpoints', SimpleNamespace(
        create_manager=create, restore=restore, fingerprint=lambda _: 'fp'))
    monkeypatch.setitem(sys.modules, 'expo_ft.realman.build', SimpleNamespace(build_agent=Mock(return_value=agent)))
    monkeypatch.setattr(evaluation, 'warmup', lambda a, _: a)
    import realman_inference_tape_trainingrtc as hardware
    load_ros = Mock()
    monkeypatch.setattr(hardware, 'load_ros_client', load_ros)
    env = SimpleNamespace(connect=Mock(), reset=Mock(), close=Mock())
    env_factory = Mock(return_value=env)
    monkeypatch.setitem(sys.modules, 'expo_ft.realman.environment', SimpleNamespace(RealmanEnv=env_factory))
    planner_factory = Mock(return_value=SimpleNamespace(agent=agent))
    episode = SimpleNamespace(save=Mock(), success=True, actions=[1], truncated=False)
    collect = Mock(return_value=episode)
    monkeypatch.setitem(sys.modules, 'expo_ft.realman.rollout', SimpleNamespace(
        RTCPlanner=planner_factory, collect_episode=collect))
    monkeypatch.setattr('builtins.input', lambda _: '')
    argv = ['--rl-checkpoint', str(checkpoint)] + (['--benchmark-only'] if benchmark else [])
    evaluation.main(argv)
    assert create.call_args.kwargs == {'read_only': True}
    assert create.call_args.args[0] == checkpoint.parent
    restore.assert_called_once_with(manager, agent, step=20)
    manager.close.assert_called_once()
    if benchmark:
        env_factory.assert_not_called()
        load_ros.assert_not_called()
    else:
        assert planner_factory.call_args.kwargs == {'only_base_actions': False}
        collect.assert_called_once()
        episode.save.assert_called_once()
        env.close.assert_called_once()


def test_frozen_restore_honors_explicit_step(monkeypatch):
    from expo_ft.realman import checkpoints
    agent = SimpleNamespace(train_base_actor=False, **{key: 0 for key in checkpoints.RL_FIELDS})
    agent.replace = lambda **kw: agent
    agent.cache_infer_params = lambda: agent
    manager = SimpleNamespace(latest_step=Mock(return_value=21),
                              restore=Mock(return_value={'rl': {}}))
    monkeypatch.setattr(checkpoints.ocp.checkpoint_utils, 'construct_restore_args', lambda _: {})
    assert checkpoints.restore(manager, agent, step=20) is agent
    assert manager.restore.call_args.args == (20,)
    manager.latest_step.assert_not_called()
