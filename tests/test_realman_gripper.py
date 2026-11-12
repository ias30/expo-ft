"""Gripper pause/resume regression without hardware or model loading."""
import csv
import io
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import realman_inference_tape_trainingrtc as tape


def runner_with_clock(monkeypatch, modes):
    clock = [0.0]
    monkeypatch.setattr(tape, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], time=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    ))
    runner = tape.RealmanInference.__new__(tape.RealmanInference)
    runner.running = True
    runner.args = SimpleNamespace(
        rtc=True, gripper_speed=400, gripper_force=800,
        gripper_close_timeout=1, gripper_open_timeout=1,
        control_hz=30, rtc_delay_steps=5, control_timeout_ms=100, steps_to_execute=8,
    )
    sequence = iter(modes)

    def command(*args):
        clock[0] += 1.0  # Reproduce the SDK's nominally nonblocking 1s call.
        return 0

    runner.right_arm = SimpleNamespace(
        rm_set_arm_slow_stop=Mock(return_value=0),
        rm_set_gripper_release=Mock(side_effect=command),
        rm_set_gripper_pick=Mock(side_effect=command),
        rm_get_gripper_state=Mock(side_effect=lambda: (0, {
            'mode': next(sequence), 'status': 1, 'enable_state': 1, 'error': 0,
            'actpos': 64, 'current_force': 100,
        })),
    )
    return runner


@pytest.mark.parametrize('target,modes', [(0, [4, 6, 6]), (1, [5, 1, 1])])
def test_both_operations_pause_and_replan(monkeypatch, target, modes):
    runner = runner_with_clock(monkeypatch, modes)
    runner.gripper_target_state = 1 - target
    runner._execute_right_gripper_action(target)
    assert runner.gripper_target_state == target
    assert runner._rtc_replan_after_gripper
    runner.right_arm.rm_set_arm_slow_stop.assert_called_once()
    assert runner.right_arm.rm_get_gripper_state.call_count == 3


def test_release_does_not_accept_closed_state(monkeypatch):
    runner = runner_with_clock(monkeypatch, [6] * 30)
    runner.gripper_target_state = 0
    with pytest.raises(TimeoutError, match='release completion timed out'):
        runner._execute_right_gripper_action(1)
    assert not getattr(runner, '_rtc_replan_after_gripper', False)


def test_initial_release_does_not_schedule_spurious_replan(monkeypatch):
    runner = runner_with_clock(monkeypatch, [])
    runner.gripper_target_state = None
    runner._execute_right_gripper_action(1)
    runner.right_arm.rm_set_arm_slow_stop.assert_not_called()
    runner.right_arm.rm_get_gripper_state.assert_not_called()
    assert not getattr(runner, '_rtc_replan_after_gripper', False)


def test_one_second_release_resets_control_deadline(monkeypatch):
    runner = runner_with_clock(monkeypatch, [5, 1, 1])
    runner.gripper_target_state = 0
    runner.cameras = {}
    runner._get_current_state = lambda: {}
    fresh = np.ones((64, 7), dtype=np.float32)
    runner._run_inference = Mock(side_effect=[np.zeros((64, 7), dtype=np.float32), fresh])
    runner._visualize = lambda: None
    executed = []

    def execute(action):
        executed.append(action.copy())
        if len(executed) == 1:
            runner._execute_right_gripper_action(1)
        else:
            runner.running = False

    runner._execute_action = execute
    stream = io.StringIO()
    runner._run_rtc(csv.writer(stream), stream)
    assert len(executed) == 2
    np.testing.assert_array_equal(executed[1], fresh[0])
    assert runner._run_inference.call_count == 2
    assert not runner._rtc_replan_after_gripper
