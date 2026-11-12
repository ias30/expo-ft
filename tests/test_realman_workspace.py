"""Workspace guard tests with a simulated SDK; no hardware connections."""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import realman_inference_tape_trainingrtc as tape


def make_runner(fk=None, measured=(0.0, -0.3, 0.0), rtc=True):
    runner = tape.RealmanInference.__new__(tape.RealmanInference)
    runner.args = SimpleNamespace(rtc=rtc, speed=30)
    runner.running = True
    runner.gripper_target_state = 1
    runner.right_arm = SimpleNamespace(
        rm_get_current_work_frame=Mock(return_value=(0, {'pose': [0] * 6})),
        rm_get_current_arm_state=Mock(return_value=(0, {'joint': [0] * 6, 'pose': [*measured, 0, 0, 0]})),
        rm_algo_forward_kinematics=Mock(side_effect=fk or (lambda joints: [*measured, 0, 0, 0])),
        rm_set_arm_slow_stop=Mock(return_value=0),
        rm_movej=Mock(return_value=0),
        rm_movej_canfd=Mock(return_value=0),
    )
    runner._execute_right_gripper_action = Mock()
    return runner


@pytest.mark.parametrize('rtc', [True, False])
@pytest.mark.parametrize('failure', ['target', 'interior', 'feedback', 'frame', 'frame_error', 'feedback_error', 'nan', 'mismatch', 'sdk_error'])
def test_rejects_unsafe_motion_before_any_command(rtc, failure):
    def fk(joints):
        if failure == 'sdk_error':
            raise RuntimeError('FK unavailable')
        x = 0.0
        if (failure == 'target' and joints[0] >= 1) or (failure == 'interior' and 0 < joints[0] < 1):
            x = .115
        if failure == 'nan':
            x = np.nan
        if failure == 'mismatch':
            x = .003
        return [x, -.3, 0, 0, 0, 0]

    runner = make_runner(fk=fk, rtc=rtc)
    if failure == 'feedback':
        runner.right_arm.rm_get_current_arm_state.return_value[1]['pose'][0] = .2
    if failure == 'frame':
        runner.right_arm.rm_get_current_work_frame.return_value[1]['pose'][0] = 1
    if failure == 'frame_error':
        runner.right_arm.rm_get_current_work_frame.return_value = (-1, {'pose': [0] * 6})
    if failure == 'feedback_error':
        runner.right_arm.rm_get_current_arm_state.return_value = (-1, {})
    with pytest.raises(RuntimeError, match='Workspace boundary'):
        runner._execute_action(np.array([np.deg2rad(1), 0, 0, 0, 0, 0, 1]))
    assert not runner.running
    runner.right_arm.rm_set_arm_slow_stop.assert_called_once()
    runner.right_arm.rm_movej.assert_not_called()
    runner.right_arm.rm_movej_canfd.assert_not_called()
    runner._execute_right_gripper_action.assert_not_called()


@pytest.mark.parametrize('axis,coordinate', [(0, -.171), (0, .115), (1, -.501), (1, -.149), (2, -.119), (2, .051)])
def test_all_six_faces_enforced_in_mm(axis, coordinate):
    xyz = [0, -.3, 0]
    xyz[axis] = coordinate
    runner = make_runner(measured=xyz)
    with pytest.raises(RuntimeError, match='outside boundary box'):
        runner._check_workspace_motion()


@pytest.mark.parametrize('rtc', [True, False])
def test_valid_motion_passes_degrees_to_sdk(monkeypatch, rtc):
    monkeypatch.setattr(tape.time, 'sleep', lambda _: None)
    runner = make_runner(rtc=rtc)
    runner._execute_action(np.array([np.deg2rad(1), 0, 0, 0, 0, 0, 1]))
    command = runner.right_arm.rm_movej_canfd if rtc else runner.right_arm.rm_movej
    np.testing.assert_allclose(command.call_args.args[0], [1, 0, 0, 0, 0, 0])
    runner.right_arm.rm_set_arm_slow_stop.assert_not_called()
    assert runner.running


def test_initial_pose_skips_workspace_guard():
    runner = make_runner(fk=lambda joints: [.2 if joints[0] > 1 else 0, -.3, 0, 0, 0, 0])
    runner._check_workspace_motion = Mock(side_effect=AssertionError('Reset must skip workspace checks'))
    runner._move_to_initial_pose()
    runner._check_workspace_motion.assert_not_called()
    runner.right_arm.rm_movej.assert_called_once_with(list(tape.INITIAL_RIGHT_JOINTS_DEG), 5, 0, 0, True)
    runner._execute_right_gripper_action.assert_called_once_with(1)


@pytest.mark.parametrize('xyz', [(-.170, -.500, -.118), (.114, -.150, .050)])
def test_boundary_faces_are_inclusive(xyz):
    runner = make_runner(measured=xyz)
    runner._check_workspace_motion()
    runner.right_arm.rm_set_arm_slow_stop.assert_not_called()
