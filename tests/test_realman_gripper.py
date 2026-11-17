"""Continuous gripper dispatch regression; no hardware/model loading."""
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import realman_inference_tape_trainingrtc as tape
from expo_ft.realman.environment import RealmanEnv


def runner_with_clock(monkeypatch, rtc=True):
    clock = [0.0]
    monkeypatch.setattr(tape, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], time=lambda: clock[0],
        sleep=Mock(side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)),
    ))
    runner = tape.RealmanInference.__new__(tape.RealmanInference)
    runner.running = True
    runner.args = SimpleNamespace(rtc=rtc, gripper_speed=400, gripper_force=800,
        control_hz=30, rtc_delay_steps=5, control_timeout_ms=100, steps_to_execute=8)
    runner.gripper_target_state = 1
    runner.right_arm = SimpleNamespace(
        rm_set_arm_slow_stop=Mock(return_value=0),
        rm_set_gripper_release=Mock(return_value=0),
        rm_set_gripper_pick=Mock(return_value=0),
        rm_get_gripper_state=Mock(side_effect=AssertionError('Must not wait for gripper')),
        rm_get_current_arm_state=Mock(side_effect=AssertionError('Must not wait for arm settling')),
        rm_movej_canfd=Mock(return_value=0),
    )
    return runner


@pytest.mark.parametrize('rtc', [False, True])
@pytest.mark.parametrize('target', [0, 1])
def test_gripper_switch_does_not_stop_sleep_or_poll(monkeypatch, rtc, target):
    runner = runner_with_clock(monkeypatch, rtc)
    runner.gripper_target_state = 1 - target
    runner._execute_right_gripper_action(target)
    assert runner.gripper_target_state == target
    assert not runner._gripper_command_pending
    runner.right_arm.rm_set_arm_slow_stop.assert_not_called()
    runner.right_arm.rm_get_gripper_state.assert_not_called()
    runner.right_arm.rm_get_current_arm_state.assert_not_called()
    tape.time.sleep.assert_not_called()
    if target:
        runner.right_arm.rm_set_gripper_release.assert_called_once_with(400, False, 1)
    else:
        runner.right_arm.rm_set_gripper_pick.assert_called_once_with(400, 800, False, 1)


def test_same_target_not_resent_but_new_target_sent_immediately(monkeypatch):
    runner = runner_with_clock(monkeypatch)
    for target in (0, 0, 1, 1, 0):
        runner._execute_right_gripper_action(target)
    assert runner.right_arm.rm_set_gripper_pick.call_count == 2
    assert runner.right_arm.rm_set_gripper_release.call_count == 1
    runner.right_arm.rm_set_arm_slow_stop.assert_not_called()


@pytest.mark.parametrize('code,message', [(1, 'rejected by controller'), (-2, 'acceptance is unknown')])
def test_ack_failure_does_not_cache_failed_target(monkeypatch, code, message):
    runner = runner_with_clock(monkeypatch)
    runner.right_arm.rm_set_gripper_pick.return_value = code
    with pytest.raises(RuntimeError, match=message):
        runner._execute_right_gripper_action(0)
    assert runner.gripper_target_state == 1
    assert runner._gripper_command_pending
    runner.right_arm.rm_set_gripper_pick.assert_called_once()
    runner.right_arm.rm_get_gripper_state.assert_not_called()


def test_initial_release_only_waits_for_ack(monkeypatch):
    runner = runner_with_clock(monkeypatch)
    runner.gripper_target_state = None
    runner._execute_right_gripper_action(1)
    assert runner.gripper_target_state == 1
    runner.right_arm.rm_get_gripper_state.assert_not_called()
    runner.right_arm.rm_set_arm_slow_stop.assert_not_called()


def test_realman_env_continues_joint_stream_and_returns_no_pause(monkeypatch):
    runner = runner_with_clock(monkeypatch)
    env = RealmanEnv(runner.args)
    env.right_arm = runner.right_arm
    env.gripper_target_state = 1
    env._check_workspace_motion = Mock()
    calls = []
    env.right_arm.rm_movej_canfd.side_effect = lambda *a, **kw: calls.append('joint') or 0
    env.right_arm.rm_set_gripper_pick.side_effect = lambda *a: calls.append('close') or 0
    env.right_arm.rm_set_gripper_release.side_effect = lambda *a: calls.append('open') or 0
    for grip in (0, 0, 1):
        actual, paused = env.step(np.array([0.] * 6 + [grip]))
        assert not paused and actual[6] == grip
    assert calls == ['joint', 'close', 'joint', 'joint', 'open']
    env.right_arm.rm_set_arm_slow_stop.assert_not_called()
    env.right_arm.rm_get_gripper_state.assert_not_called()
    assert env._check_workspace_motion.call_count == 3


def test_gripper_switch_does_not_discard_rtc_chunk(monkeypatch):
    runner = runner_with_clock(monkeypatch)
    runner.cameras = {}
    runner._get_current_state = lambda: {}
    chunk = np.zeros((64, 7), np.float32)
    chunk[1, 0] = .1
    runner._run_inference = Mock(return_value=chunk)
    runner._visualize = lambda: None
    runner._check_workspace_motion = Mock()
    execute = runner._execute_action
    executed = []
    def step(action):
        execute(action)
        executed.append(action.copy())
        if len(executed) == 2:
            runner.running = False
    runner._execute_action = step
    stream = io.StringIO()
    runner._run_rtc(csv.writer(stream), stream)
    np.testing.assert_array_equal(executed, chunk[:2])
    runner._run_inference.assert_called_once()
    runner.right_arm.rm_set_gripper_pick.assert_called_once()
    # One stop from final cleanup, none from the gripper switch.
    runner.right_arm.rm_set_arm_slow_stop.assert_called_once()


def test_trace_distinguishes_ack_from_physical_completion(monkeypatch, tmp_path):
    runner = runner_with_clock(monkeypatch)
    runner._gripper_trace_path = tmp_path / 'events.jsonl'
    runner.diagnostic_episode, runner.diagnostic_step = 3, 42
    runner._execute_right_gripper_action(0)
    rows = [json.loads(s) for s in runner._gripper_trace_path.read_text().splitlines()]
    assert [r['event'] for r in rows] == ['gripper_requested', 'gripper_send', 'gripper_ack', 'gripper_command_accepted']
    assert all(r['episode'] == 3 and r['step'] == 42 for r in rows)
    runner.right_arm.rm_get_current_arm_state.assert_not_called()


def test_diagnostic_write_failure_does_not_change_dispatch(monkeypatch, tmp_path):
    runner = runner_with_clock(monkeypatch)
    runner._gripper_trace_path = tmp_path
    runner._execute_right_gripper_action(0)
    assert runner.gripper_target_state == 0


def test_rl_accepts_sdk_debug_directory():
    from train_realman_rl import parse_args
    assert parse_args(['--sdk-debug-dir', 'logs/diagnostics']).sdk_debug_dir == 'logs/diagnostics'
