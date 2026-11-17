"""RealMan hardware environment. Construction never loads a VLA or moves a robot."""
import numpy as np
from realman_inference_tape_trainingrtc import RealmanInference
from expo_ft.realman.workspace import WorkspaceMotionRejected


class RealmanEnv(RealmanInference):
    def __init__(self, args):
        # Reuse tested SDK methods without invoking the inference constructor.
        self.args = args
        self.running = True
        self.right_arm = None
        self.cameras = {}
        self.raw_images = {}
        self.gripper_target_state = None
        self.fault = None
        self.recover_workspace_rejection = True
        self.workspace_reset_required = False

    def connect(self):
        try:
            self._initialize_hardware()
        except BaseException:
            self.cleanup()
            raise

    def stop(self):
        if self.right_arm is not None:
            code = self.right_arm.rm_set_arm_slow_stop()
            if code != 0:
                raise RuntimeError(f'Arm stop failed: {code}')

    def reset(self):
        if self.fault:
            raise RuntimeError(f'Environment fault is latched: {self.fault}')
        self.stop()
        self._move_to_initial_pose()  # Startup/reset intentionally bypass workspace guard.
        observation = self.get_observation()
        self.workspace_reset_required = False
        return observation

    def get_observation(self):
        if not self.running or self.fault:
            raise RuntimeError('Environment stopped or faulted')
        try:
            observation = self._get_current_state()
            observation['prompt'] = self.args.prompt
            return observation
        except Exception as exc:
            self.fault = str(exc)
            self.stop()
            raise

    def step(self, action):
        if not self.running or self.fault:
            raise RuntimeError('Environment stopped or faulted')
        if self.workspace_reset_required:
            raise RuntimeError('Workspace rejection requires a confirmed reset before another action')
        try:
            self._execute_action(action)
            actual = np.array(action, np.float32, copy=True)
            actual[6] = self.gripper_target_state
            return actual, False
        except WorkspaceMotionRejected:
            # Guard has stopped successfully and revalidated actual feedback.
            # Keep the connection, but disallow another policy action until reset.
            self.workspace_reset_required = True
            raise
        except BaseException as exc:
            self.fault = str(exc)
            self.stop()
            raise

    def get_info_for_step(self):
        # Detector imports are deferred until an actual hardware rollout.
        from client.real_utils.detector import success_detector_manual
        label = success_detector_manual()
        done, success = label in ('success', 'reset'), label == 'success'
        return done, success, float(success), float(not done)

    def close(self):
        self.running = False
        self.cleanup()
