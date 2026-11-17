"""RealMan joint-space task; both policy and critic use the actual right-wrist slot."""
from ml_collections import ConfigDict


def get_config():
    return ConfigDict(dict(
        env_type='realman', language_instruction='pick up the tape from the rotating disc',
        critic_camera_keys=('base_0_rgb', 'right_wrist_0_rgb'),
        edit_action_xyzg=False, control_hz=30.0, replan_steps=30, delay=5,
        workspace_min=(-170., -500., -122.), workspace_max=(114., -150., 50.),
    ))
