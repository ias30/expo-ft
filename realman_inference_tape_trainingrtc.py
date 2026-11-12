#!/usr/bin/env python3
"""RealMan tape deployment for the native RTC-SFT checkpoint.

Adapted from openpi-git/examples/realman_tape/realman_inference_tape.py.
Uses clean-prefix clamping (per-token time=0) matching RTC-SFT training.
Run --benchmark-only first to measure inference without hardware connections.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import faulthandler
import logging
import json
import os
import pathlib
import signal
import sys
import threading
import time

import cv2
import numpy as np

# Use exactly the model implementation used to train this checkpoint.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "third_party/openpi_rtc/src"),
    str(PROJECT_ROOT / "third_party/openpi_rtc/packages/openpi-client/src"),
]

from openpi.policies import policy_config
from openpi.policies.training_rtc_buffer import TrainingRTCBuffer
from openpi.shared import normalize
from openpi.training import config as training_config

logger = logging.getLogger(__name__)
PROMPT = "pick up the tape from the rotating disc"
# Same source -> model mapping as convert_realman_to_lerobot_grippers.py.
CAMERA_MAPPING = {
    "camera_high": "cam_high",
    "camera_left_wrist": "cam_right_wrist",
}
# dual_arm_lcx_data_collection.py:499,508: SDK joint targets are in degrees.
INITIAL_RIGHT_JOINTS_DEG = (
    109.89800262451172,
    84.50399780273438,
    52.55799865722656,
    -43.3380012512207,
    68.61699676513672,
    -41.744998931884766,
)
INITIAL_RIGHT_SPEED = 5


def preprocess_image(image_bgr):
    """Camera BGR HWC -> policy RGB uint8 CHW; policy handles resizing."""
    if image_bgr is None:
        raise RuntimeError("Camera has not supplied an image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)


def load_ros_client(path):
    """Load ROS1 before JAX warmup; no node registration or hardware access."""
    os.environ["ROS_MASTER_URI"] = "http://169.254.128.20:11311"
    os.environ["ROS_IP"] = "169.254.128.101"
    os.environ.pop("ROS_HOSTNAME", None)
    ros_path = pathlib.Path(path).expanduser().resolve()
    if not (ros_path / "rospy").is_dir():
        raise FileNotFoundError(f"ROS1 rospy directory missing under {ros_path}")
    if str(ros_path) not in sys.path[:1]:
        sys.path.insert(0, str(ros_path))
    if "rospy" not in sys.modules:
        print("[ROS] Importing rospy before model loading...", flush=True)
        start = time.monotonic()
        # A slow import now emits its actual stack instead of remaining silent.
        faulthandler.dump_traceback_later(10, repeat=False)
        try:
            # Bundled ROS1 findCaller can loop forever at the bottom of the stack
            # under Python 3.11 when INFO logging is enabled. Patch this process
            # only; keep the external ROS installation unchanged.
            from rosgraph import roslogging

            roslogging.RospyLogger.findCaller = logging.Logger.findCaller
            import rospy
        finally:
            faulthandler.cancel_dump_traceback_later()
        print(f"[ROS] rospy imported in {time.monotonic() - start:.3f}s: {rospy.__file__}", flush=True)
    return sys.modules["rospy"]


class ROSImageCamera:
    """Receive BGR/RGB ROS1 images without cv_bridge; reject stale frames."""

    def __init__(self, topic, max_age):
        self.topic = topic
        self.max_age = max_age
        self.lock = threading.Lock()
        self.image = None
        self.received_at = 0.0
        self.subscriber = None

    @staticmethod
    def decode_image(msg):
        if msg.encoding not in ("bgr8", "rgb8"):
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")
        if (msg.width, msg.height) != (640, 480) or msg.step < msg.width * 3:
            raise ValueError("Expected 640x480 RGB/BGR image with valid row stride")
        payload = bytes(msg.data) if isinstance(msg.data, list | tuple) else msg.data
        data = np.frombuffer(payload, dtype=np.uint8)
        if data.size != msg.height * msg.step:
            raise ValueError("Image payload size does not match height * step")
        frame = data.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return frame[:, :, ::-1] if msg.encoding == "rgb8" else frame

    def start_capture(self):
        import rospy
        from sensor_msgs.msg import Image

        self.subscriber = rospy.Subscriber(
            self.topic,
            Image,
            self.receive,
            queue_size=1,
            buff_size=4 * 1024 * 1024,
            tcp_nodelay=True,
        )

    def receive(self, msg):
        received_at = time.monotonic()
        try:
            frame = self.decode_image(msg)
        except ValueError as exc:
            import rospy

            rospy.logwarn_throttle(5.0, f"{self.topic}: {exc}")
            return
        with self.lock:
            self.image = frame.copy()
            self.received_at = received_at

    def get_latest_data(self):
        with self.lock:
            if self.image is None or time.monotonic() - self.received_at > self.max_age:
                raise RuntimeError(f"Missing or stale camera image: {self.topic}")
            return self.received_at, self.image.copy()

    def stop_capture(self):
        if self.subscriber is not None:
            self.subscriber.unregister()
            self.subscriber = None


class RealmanInference:
    """Seven physical dimensions: six right joints in radians + gripper command."""

    def __init__(self, args):
        self.args = args
        self.running = True
        self.right_arm = None
        self.cameras = {}
        self.gripper_target_state = None
        self.raw_images = {}
        config = training_config.get_config(args.config_name)
        self.action_horizon = config.model.action_horizon
        if not config.model.rtc_enabled:
            raise ValueError("This entrypoint requires an RTC-SFT training config")
        if not 1 <= args.rtc_delay_steps <= config.model.rtc_max_delay:
            raise ValueError(f"rtc-delay-steps must be 1..{config.model.rtc_max_delay}, matching training")
        if args.rtc and args.steps_to_execute + args.rtc_delay_steps >= self.action_horizon:
            raise ValueError("RTC requires steps-to-execute + rtc-delay-steps < action horizon")
        checkpoint = pathlib.Path(args.checkpoint).expanduser()
        # Accept the former /params CLI path as well as the policy checkpoint root.
        if checkpoint.name == "params":
            checkpoint = checkpoint.parent
        if not (checkpoint / "params").is_dir() or not (checkpoint / "_CHECKPOINT_METADATA").is_file():
            raise FileNotFoundError(f"Expected a committed checkpoint root: {checkpoint}")
        if not json.loads((checkpoint / "_CHECKPOINT_METADATA").read_text()).get("commit_timestamp_nsecs"):
            raise ValueError(f"Checkpoint has not completed saving: {checkpoint}")
        stats = None
        if args.norm_stats:
            stats_path = pathlib.Path(args.norm_stats).expanduser()
            stats = normalize.load(stats_path.parent if stats_path.name == "norm_stats.json" else stats_path)
        self.policy = policy_config.create_trained_policy(
            config,
            checkpoint,
            norm_stats=stats,
            default_prompt=PROMPT,
            sample_kwargs={"num_steps": args.denoising_steps},
        )
        # Compile the complete policy before opening any hardware connections.
        logger.info("Starting policy warmup/JIT compilation (no hardware connected)")
        self._run_inference(
            {
                "state": np.zeros(7, dtype=np.float32),
                "images": {k: np.zeros((3, 480, 640), dtype=np.uint8) for k in ("cam_high", "cam_right_wrist")},
                "prompt": PROMPT,
            }
        )

    def _initialize_cameras(self):
        rospy = load_ros_client(self.args.ros_python_path)

        if not rospy.core.is_initialized():
            print(
                f"[ROS] Initializing node: master={os.environ['ROS_MASTER_URI']}, local_ip={os.environ['ROS_IP']}",
                flush=True,
            )

            def initialization_timeout(signum, frame):
                raise TimeoutError(
                    "ROS initialization exceeded 30s; check master registration and local ROS_IP connectivity"
                )

            old_handler = signal.signal(signal.SIGALRM, initialization_timeout)
            signal.alarm(30)
            try:
                rospy.init_node("realman_tape_trainingrtc_inference", anonymous=True, disable_signals=True)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        # rospy may reconfigure logging during init_node; restore our progress output.
        logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s")
        logger.disabled = False
        logger.setLevel(logging.INFO)
        print(f"[ROS] Node initialized: {rospy.get_node_uri()}", flush=True)
        rospy.on_shutdown(lambda: setattr(self, "running", False))
        self.cameras = {
            "camera_high": ROSImageCamera(self.args.head_camera_topic, self.args.camera_max_age),
            "camera_left_wrist": ROSImageCamera(self.args.wrist_camera_topic, self.args.camera_max_age),
        }
        for camera in self.cameras.values():
            print(f"[ROS] Subscribing: {camera.topic}", flush=True)
            camera.start_capture()
        print("[ROS] Waiting for fresh images from both topics...", flush=True)
        deadline = time.monotonic() + self.args.camera_connect_timeout
        while self.running:
            try:
                for camera in self.cameras.values():
                    camera.get_latest_data()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for fresh images on both ROS topics") from None
                time.sleep(0.05)
        if not self.running:
            raise RuntimeError("Stopped while waiting for camera topics")
        print("[ROS] Both camera streams ready", flush=True)

    def _check_workspace_motion(self, target_deg=None):
        """Fail closed on invalid feedback or sampled TCP path outside base-frame mm bounds.

        Joint interpolation sampling is a software guard, not a controller safety fence.
        Keep the active work/tool frames unchanged during deployment.
        """
        lower = np.array([-170.0, -500.0, -121.0])
        upper = np.array([114.0, -150.0, 50.0])
        try:
            code, frame = self.right_arm.rm_get_current_work_frame()
            frame_pose = np.asarray(frame["pose"], dtype=float)
            if code != 0 or frame_pose.shape != (6,) or not np.allclose(frame_pose, 0, atol=1e-7):
                raise ValueError("Workspace guard requires an identity/base work frame")
            code, state = self.right_arm.rm_get_current_arm_state()
            if code != 0:
                raise ValueError(f"Workspace feedback failed: {code}")
            current = np.asarray(state["joint"][:6], dtype=float)
            measured = np.asarray(state["pose"][:3], dtype=float) * 1000.0

            def check_xyz(xyz):
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    raise ValueError("Invalid TCP position")
                if np.any(xyz < lower) or np.any(xyz > upper):
                    raise ValueError(f"TCP outside boundary box: xyz_mm={xyz.tolist()}, min={lower}, max={upper}")

            check_xyz(measured)
            target = current if target_deg is None else np.asarray(target_deg, dtype=float)
            if current.shape != (6,) or target.shape != (6,) or not np.isfinite([current, target]).all():
                raise ValueError("Invalid workspace joint values")
            # Compare connected-arm FK with actual TCP to detect frame/tool mismatch.
            fk_current = np.asarray(self.right_arm.rm_algo_forward_kinematics(current.tolist())[:3]) * 1000.0
            check_xyz(fk_current)
            if np.linalg.norm(fk_current - measured) > 2.0:
                raise ValueError("FK and measured TCP differ by more than 2 mm; check work/tool frames")
            # Both endpoints may be inside while a joint-space path leaves the box.
            samples = max(1, int(np.ceil(np.max(np.abs(target - current)) / 0.5)))
            for fraction in np.linspace(0.0, 1.0, samples + 1)[1:]:
                joints = current + fraction * (target - current)
                xyz = np.asarray(self.right_arm.rm_algo_forward_kinematics(joints.tolist())[:3]) * 1000.0
                check_xyz(xyz)
        except Exception as exc:
            self.running = False
            try:
                result = self.right_arm.rm_set_arm_slow_stop()
                if result != 0:
                    logger.error("Workspace stop failed: %s", result)
            except Exception:
                logger.exception("Workspace stop command failed")
            raise RuntimeError(f"Workspace boundary check failed; motion rejected: {exc}") from exc

    def _initialize_hardware(self):
        self._initialize_cameras()
        from Robotic_Arm.rm_robot_interface import RoboticArm
        from Robotic_Arm.rm_robot_interface import rm_thread_mode_e

        self.right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.right_arm.rm_create_robot_arm(self.args.right_robot_ip, 8080)
        if handle.id <= 0:
            raise RuntimeError(f"Right arm connection failed: {handle}")
        logger.info("Right TCP workspace (base frame, mm): min=(-170, -500, -118), max=(114, -150, 50)")
        # Initialization/reset is exempt; policy actions are checked in _execute_action.
        # Explicit startup release establishes the first known command target.
        self._execute_right_gripper_action(1)
        if self.args.reset_pose:
            self._move_to_initial_pose()

    def benchmark_cameras(self):
        """Live images, fixed collection initial state, no robot SDK connections."""
        signal.signal(signal.SIGINT, self._signal_handler)
        last_timestamps = {}
        state = np.asarray([*np.deg2rad(INITIAL_RIGHT_JOINTS_DEG), 1.0], dtype=np.float32)

        def observe():
            deadline = time.monotonic() + self.args.camera_connect_timeout
            while self.running:
                frames = {name: camera.get_latest_data() for name, camera in self.cameras.items()}
                if all(ts > last_timestamps.get(name, -1) for name, (ts, _) in frames.items()):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for new frames from both cameras")
                time.sleep(0.001)
            if not self.running:
                raise InterruptedError("Camera benchmark interrupted")
            now = time.monotonic()
            ages = [now - ts for ts, _ in frames.values()]
            last_timestamps.update({name: ts for name, (ts, _) in frames.items()})
            self.raw_images = {name: frame for name, (_, frame) in frames.items()}
            return {
                "state": state.copy(),
                "images": {dest: preprocess_image(self.raw_images[src]) for src, dest in CAMERA_MAPPING.items()},
                "prompt": PROMPT,
            }, max(ages)

        try:
            self._initialize_cameras()
            logger.info("Camera benchmark: live high/left images; fixed initial-pose state, no robot connection")
            observation, _ = observe()
            previous = self._run_inference(observation)
            overlap = self.action_horizon - self.args.steps_to_execute
            measurements = []
            # Two live-image warmup calls are excluded from statistics.
            for iteration in range(self.args.benchmark_iterations + 2):
                target = np.zeros((self.action_horizon, 7), dtype=np.float32)
                target[:overlap] = previous[self.args.steps_to_execute :]
                start = time.monotonic()
                observation, age = observe()
                acquired = time.monotonic()
                previous = self._run_inference(observation, target, self.args.rtc_delay_steps)
                finished = time.monotonic()
                if iteration >= 2:
                    row = (
                        (acquired - start) * 1000,
                        (finished - acquired) * 1000,
                        (finished - start) * 1000,
                        age * 1000,
                    )
                    measurements.append(row)
                    logger.info(
                        "Camera benchmark %d/%d: acquire=%.2fms policy=%.2fms total=%.2fms received_frame_age=%.2fms",
                        iteration - 1,
                        self.args.benchmark_iterations,
                        *row,
                    )
                self._visualize()
            values = np.asarray(measurements)
            for i, name in enumerate(("acquire", "policy", "total", "received_frame_age")):
                logger.info(
                    "Camera benchmark %s: mean=%.2fms p50=%.2fms p95=%.2fms max=%.2fms",
                    name,
                    values[:, i].mean(),
                    *np.percentile(values[:, i], [50, 95, 100]),
                )
            delay = int(np.ceil(values[:, 2].max() / 1000 * self.args.control_hz))
            logger.info("Camera benchmark observed max total delay=%d ticks at %.1fHz", delay, self.args.control_hz)
            if delay > self.args.rtc_delay_steps:
                raise RuntimeError(
                    f"Measured delay {delay} exceeds clean prefix {self.args.rtc_delay_steps}; lower --control-hz"
                )
        finally:
            self.cleanup()

    def _move_to_initial_pose(self):
        """Match collection startup at speed=5, without the policy workspace guard."""
        logger.info("Moving right arm to collection initial joints (degrees): %s", INITIAL_RIGHT_JOINTS_DEG)
        result = self.right_arm.rm_movej(
            list(INITIAL_RIGHT_JOINTS_DEG),
            INITIAL_RIGHT_SPEED,
            0,
            0,
            True,  # noqa: FBT003
        )
        if result != 0:
            raise RuntimeError(f"Right arm initial pose motion failed: {result}")
        self._execute_right_gripper_action(1)

    def _get_current_state(self):
        sdk_start = time.monotonic()
        code, arm_state = self.right_arm.rm_get_current_arm_state()
        sdk_ms = (time.monotonic() - sdk_start) * 1000
        if sdk_ms > 20:
            logger.warning("SDK rm_get_current_arm_state took %.1fms (result=%s)", sdk_ms, code)
        if code != 0:
            raise RuntimeError(f"Right arm feedback failed: {code}")
        joints = np.asarray(arm_state["joint"][:6], dtype=np.float32)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Invalid right arm joint feedback")
        if self.gripper_target_state not in (0, 1):
            raise RuntimeError("Right gripper target has not been initialized")
        self.raw_images = {name: camera.get_latest_data()[1] for name, camera in self.cameras.items()}
        return {
            "state": np.concatenate([np.deg2rad(joints), [self.gripper_target_state]]).astype(np.float32),
            "images": {
                model_name: preprocess_image(self.raw_images[source]) for source, model_name in CAMERA_MAPPING.items()
            },
            "prompt": PROMPT,
        }

    def _run_inference(self, observation, previous_actions=None, prefix_length=0):
        # Transform absolute targets against THIS observation before clamping.
        # The policy unnormalizes output and adds the observed joints exactly once.
        if previous_actions is None:
            previous_actions = np.zeros((self.action_horizon, 7), dtype=np.float32)
            prefix_length = 0
        actions = np.asarray(self.policy.infer(
            observation,
            rtc_previous_actions=previous_actions,
            rtc_prefix_length=prefix_length,
        )["actions"])
        if actions.shape != (self.action_horizon, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite ({self.action_horizon}, 7) actions, got {actions.shape}")
        if prefix_length:
            np.testing.assert_allclose(
                actions[:prefix_length], previous_actions[:prefix_length], rtol=1e-4, atol=1e-4,
                err_msg="Clean prefix changed across policy transforms",
            )
        return actions

    @staticmethod
    def _map_gripper_value_to_state(value, current_state):
        # Preserve the original hysteresis: 0=pick/closed, 1=release/open.
        return int(value >= 0.7) if current_state == 0 else int(value > 0.3)

    def _execute_right_gripper_action(self, target_state):
        if target_state == self.gripper_target_state:
            return
        if target_state not in (0, 1):
            raise ValueError("Gripper target must be 0=pick or 1=release")
        # Match collection: nonblocking commands, timeout=0, no motor/serial API.
        command = "release" if target_state == 1 else "pick"
        pause_for_gripper = self.gripper_target_state in (0, 1) and getattr(self.args, "rtc", False)
        if pause_for_gripper:
            result = self.right_arm.rm_set_arm_slow_stop()
            if result != 0:
                raise RuntimeError(f"Arm stop before gripper {command} failed: {result}")
            logger.info("RTC paused for gripper %s; waiting for physical completion feedback", command)
        logger.info("SDK gripper %s starting (block=False, timeout=0)", command)
        sdk_start = time.monotonic()
        if target_state == 1:
            result = self.right_arm.rm_set_gripper_release(self.args.gripper_speed, False, 0)  # noqa: FBT003
        else:
            result = self.right_arm.rm_set_gripper_pick(
                self.args.gripper_speed,
                self.args.gripper_force,
                False,  # noqa: FBT003
                0,
            )
        logger.info(
            "SDK gripper %s returned after %.1fms (result=%s)", command, (time.monotonic() - sdk_start) * 1000, result
        )
        if result != 0:
            raise RuntimeError(f"Right gripper command failed: {result}")
        # Accepted command target, NOT measured physical gripper position.
        self.gripper_target_state = target_state
        if pause_for_gripper:
            self._wait_for_gripper_target(target_state)
            self._rtc_replan_after_gripper = True

    def _wait_for_gripper_closed(self):
        """Compatibility wrapper for waiting on a close command."""
        self._wait_for_gripper_target(0)

    def _wait_for_gripper_target(self, target_state):
        """Require two healthy completion samples: open=1, close=2 or 6.

        Force-stop mode 6 does not by itself prove a successful object grasp.
        """
        if target_state not in (0, 1):
            raise ValueError("Gripper target must be 0=pick or 1=release")
        operation = "close" if target_state == 0 else "release"
        completion_modes = (2, 6) if target_state == 0 else (1,)
        timeout = self.args.gripper_close_timeout if target_state == 0 else self.args.gripper_open_timeout
        started = time.monotonic()
        deadline = started + timeout
        stable = 0
        last_state = None
        last_mode = None
        next_log = started
        polls = 0
        while self.running:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Gripper {operation} completion timed out after {time.monotonic() - started:.2f}s; "
                    f"polls={polls}, last_state={last_state}; RTC will not resume"
                )
            query_started = time.monotonic()
            result, state = self.right_arm.rm_get_gripper_state()
            now = time.monotonic()
            polls += 1
            if result != 0:
                raise RuntimeError(f"Gripper completion query failed: {result}")
            last_state = dict(state)
            mode = state.get("mode")
            if mode != last_mode or now >= next_log:
                logger.info(
                    "Gripper %s feedback: elapsed=%.2fs query=%.1fms mode=%s actpos=%s "
                    "force=%s status=%s enabled=%s error=%s stable=%d/2",
                    operation, now - started, (now - query_started) * 1000, mode, state.get("actpos"),
                    state.get("current_force"), state.get("status"), state.get("enable_state"),
                    state.get("error"), stable,
                )
                next_log = now + 1.0
                last_mode = mode
            if state.get("status") != 1 or state.get("enable_state") != 1 or state.get("error") != 0:
                raise RuntimeError(f"Gripper not healthy during {operation}: {state}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Gripper {operation} completion query exceeded timeout; polls={polls}, "
                    f"last_state={last_state}; RTC will not resume"
                )
            stable = stable + 1 if mode in completion_modes else 0
            if stable >= 2:
                logger.info(
                    "Gripper %s completed: mode=%s actpos=%s force=%s; fresh inference may resume",
                    operation, mode,
                    state.get("actpos"),
                    state.get("current_force"),
                )
                return
            time.sleep(0.05)
        raise InterruptedError(f"Stopped while waiting for gripper {operation}")

    def _execute_action(self, action):
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite action values")
        self._check_workspace_motion(np.rad2deg(action[:6]))
        if getattr(self.args, "rtc", False):
            # 30Hz data: low-follow streaming. High-follow requires <=10ms per SDK.
            sdk_start = time.monotonic()
            code = self.right_arm.rm_movej_canfd(np.rad2deg(action[:6]).tolist(), follow=False)
            sdk_ms = (time.monotonic() - sdk_start) * 1000
            if sdk_ms > 20:
                logger.warning("SDK rm_movej_canfd took %.1fms (result=%s)", sdk_ms, code)
        else:
            code = self.right_arm.rm_movej(np.rad2deg(action[:6]).tolist(), self.args.speed, 0, 0, True)  # noqa: FBT003
        if code != 0:
            raise RuntimeError(f"Right arm motion failed: {code}")
        target = self._map_gripper_value_to_state(float(np.clip(action[6], 0, 1)), self.gripper_target_state)
        self._execute_right_gripper_action(target)
        if not getattr(self.args, "rtc", False):
            time.sleep(0.2)  # Legacy synchronous mode only.

    def _visualize(self):
        if self.args.enable_visualization:
            images = [cv2.resize(self.raw_images[k], (320, 240)) for k in CAMERA_MAPPING]
            cv2.imshow("Tape sources: high | left wrist (model: right wrist)", np.hstack(images))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

    def _run_rtc(self, writer, stream):
        """One inference worker; SDK calls and observation snapshots stay on main thread.

        Execute exactly d buffered actions while inference fixes those same d
        actions as clean context. Wait until tick d if inference finishes early;
        stop if it misses that boundary. Never execute an unconditioned gap.
        """
        period = 1.0 / self.args.control_hz
        delay_steps = self.args.rtc_delay_steps
        initial = self._run_inference(self._get_current_state())
        buffer = TrainingRTCBuffer(initial)
        pending = None
        request_origin = 0
        request_time = 0.0
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-inference")
        deadline = time.monotonic()
        try:
            while self.running:
                wait = deadline - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                if not self.running:
                    break
                now = time.monotonic()
                lateness = now - deadline
                tolerance = self.args.control_timeout_ms / 1000.0
                if lateness > tolerance:
                    raise RuntimeError(
                        f"RTC control deadline missed by {lateness * 1000:.1f}ms "
                        f"(limit {self.args.control_timeout_ms:.1f}ms)"
                    )
                if lateness > period:
                    logger.warning("RTC control late by %.1fms; rescheduling without catch-up", lateness * 1000)
                # Anchor the next deadline to this actual tick, avoiding catch-up bursts.
                deadline = now
                if pending is not None and buffer.tick - request_origin >= delay_steps:
                    if not pending.done():
                        raise RuntimeError(
                            f"Inference missed the {delay_steps}-tick clean prefix; lower --control-hz"
                        )
                    elapsed = buffer.install(request_origin, pending.result())
                    logger.info("RTC chunk installed: delay=%d steps, %.3fs", elapsed, time.monotonic() - request_time)
                    pending = None
                if pending is None and buffer.tick - buffer.last_request >= self.args.steps_to_execute:
                    # No execution during this snapshot; the same state anchors target delta conversion.
                    observation = self._get_current_state()
                    request_origin, target, prefix_length = buffer.request(delay_steps)
                    request_time = time.monotonic()
                    pending = executor.submit(self._run_inference, observation, target, prefix_length)
                # Stale streams stop execution even while a chunk is still buffered.
                for camera in self.cameras.values():
                    camera.get_latest_data()
                action = buffer.pop()
                self._execute_action(action)
                writer.writerow([*np.rad2deg(action[:6]), action[6], self.gripper_target_state, time.time()])
                self._visualize()
                if getattr(self, "_rtc_replan_after_gripper", False):
                    # Finish/discard pre-pause inference before reusing the policy RNG.
                    if pending is not None:
                        pending.result()
                        pending = None
                    if not self.running:
                        break
                    fresh = self._run_inference(self._get_current_state())
                    buffer = TrainingRTCBuffer(fresh)
                    self._rtc_replan_after_gripper = False
                    deadline = time.monotonic() + period
                    logger.info("RTC resumed with fresh observation/chunk after gripper operation; old chunk discarded")
                    continue
                deadline += period
        finally:
            self.running = False
            # Stop streaming motion before waiting for an outstanding GPU computation.
            try:
                code = self.right_arm.rm_set_arm_slow_stop()
                if code != 0:
                    logger.error("Arm stop returned %s", code)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
                stream.flush()

    def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        try:
            self._initialize_hardware()
            output = pathlib.Path(self.args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [f"right_joint_{i}_deg" for i in range(6)]
                    + [
                        "right_gripper_prediction",
                        "right_gripper_target_state",
                        "timestamp",
                    ]
                )
                if self.args.rtc:
                    self._run_rtc(writer, stream)
                    return
                while self.running:
                    start = time.monotonic()
                    actions = self._run_inference(self._get_current_state())
                    for action in actions[: self.args.steps_to_execute]:
                        if not self.running:
                            break
                        self._execute_action(action)
                        writer.writerow([*np.rad2deg(action[:6]), action[6], self.gripper_target_state, time.time()])
                    stream.flush()
                    self._visualize()
                    logger.info("Observation/inference/execution cycle: %.3fs", time.monotonic() - start)
        finally:
            self.cleanup()

    def _signal_handler(self, sig, frame):
        self.running = False

    def cleanup(self):
        for camera in self.cameras.values():
            try:
                camera.stop_capture()
            except Exception:
                logger.exception("Camera cleanup failed")
        if self.right_arm is not None:
            try:
                self.right_arm.rm_set_arm_slow_stop()
            finally:
                self.right_arm.rm_delete_robot_arm()
                self.right_arm = None
        if self.args.enable_visualization:
            cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Right-arm RTC-SFT tape inference with clean action prefixes")
    parser.add_argument("--config-name", default="pi05_realman_tape2_rtc")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints/realman_rtc/pi05_realman_tape2_rtc/rtc_native_h64_d5_bs8_acc8/10000"))
    parser.add_argument(
        "--norm-stats", default="/home/ren9/smw/lcx/RTC/expo-ft/third_party/openpi_rtc/assets/pi05_realman_tape2/realman_tape2/norm_stats.json", help="Optional override directory or norm_stats.json; defaults to checkpoint assets"
    )
    parser.add_argument(
        "--ros-python-path",
        default="/home/ren9/zw/realman_deploy5090/ros_python",
        help="Directory containing ROS1 rospy and sensor_msgs Python packages",
    )
    parser.add_argument("--right-robot-ip", default="169.254.128.19")
    parser.add_argument("--gripper-speed", type=int, default=400)
    parser.add_argument("--gripper-force", type=int, default=800)
    parser.add_argument(
        "--gripper-close-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for physical closure feedback after pick command returns",
    )
    parser.add_argument("--gripper-open-timeout", type=float, default=10.0,
                        help="Seconds to wait for physical opening feedback after release")
    parser.add_argument("--head-camera-topic", default="/head_camera/image_raw")
    parser.add_argument(
        "--wrist-camera-topic",
        default="/left_hand_camera/image_raw",
        help="Topic mapped to cam_right_wrist, matching the training camera_left_wrist source",
    )
    parser.add_argument("--camera-max-age", type=float, default=0.5)
    parser.add_argument("--camera-connect-timeout", type=float, default=10.0)
    parser.add_argument("--output", default="realman_tape_trainingrtc.csv")
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument(
        "--steps-to-execute",
        type=int,
        default=30,
        help="RTC minimum interval between inference starts; synchronous execution length otherwise",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Measure warmed-up model inference without ROS or robot connections",
    )
    parser.add_argument(
        "--benchmark-cameras",
        action="store_true",
        help="Benchmark live ROS cameras without robot connections; state is fixed to initial pose",
    )
    parser.add_argument("--benchmark-iterations", type=int, default=30)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--control-timeout-ms",
        type=float,
        default=100.0,
        help="Maximum allowed lateness beyond a control deadline, in milliseconds",
    )
    parser.add_argument("--rtc-delay-steps", type=int, default=5, help="Clean prefix length; trained range 1..5")
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument(
        "--reset-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move right arm to collection initial joints at speed 5 (default); --no-reset-pose skips this",
    )
    parser.add_argument("--enable-visualization", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.speed <= 100 or args.steps_to_execute < 1 or args.denoising_steps < 1:
        parser.error("speed must be 1..100; steps-to-execute and denoising-steps must be positive")
    if not 1 <= args.gripper_speed <= 1000 or not 50 <= args.gripper_force <= 1000:
        parser.error("gripper-speed must be 1..1000; gripper-force must be 50..1000")
    if any(not np.isfinite(x) or x <= 0 for x in (args.camera_max_age, args.camera_connect_timeout)):
        parser.error("Camera timeouts must be finite and positive")
    if not np.isfinite(args.gripper_close_timeout) or args.gripper_close_timeout <= 0:
        parser.error("gripper-close-timeout must be finite and positive")
    if not np.isfinite(args.gripper_open_timeout) or args.gripper_open_timeout <= 0:
        parser.error("gripper-open-timeout must be finite and positive")
    if not np.isfinite(args.control_timeout_ms) or args.control_timeout_ms <= 0:
        parser.error("control-timeout-ms must be finite and positive")
    if args.rtc_delay_steps < 1 or any(not np.isfinite(x) or x <= 0 for x in (args.control_hz,)):
        parser.error("RTC delay must be >=1; control-hz must be finite and positive")
    if args.benchmark_iterations < 1 or (args.benchmark_cameras and args.benchmark_only):
        parser.error("benchmark-iterations must be positive; choose one benchmark mode")
    if args.benchmark_cameras and not args.rtc:
        parser.error("Camera benchmark currently requires --rtc")
    return args


def main():
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logger.info(
        "Starting %s; checkpoint=%s",
        "camera benchmark" if args.benchmark_cameras else ("offline benchmark" if args.benchmark_only else "inference"),
        args.checkpoint,
    )
    if not args.benchmark_only:
        load_ros_client(args.ros_python_path)
    runner = RealmanInference(args)
    logger.info("Model loaded and policy warmup completed")
    if args.benchmark_cameras:
        runner.benchmark_cameras()
        return
    if args.benchmark_only:
        observation = {
            "state": np.zeros(7, dtype=np.float32),
            "images": {name: np.zeros((3, 480, 640), dtype=np.uint8) for name in CAMERA_MAPPING.values()},
            "prompt": PROMPT,
        }
        previous = np.zeros((runner.action_horizon, 7), dtype=np.float32)
        # Warm both boot and streaming paths before measuring.
        previous = runner._run_inference(observation, previous, args.rtc_delay_steps)
        durations = []
        for iteration in range(args.benchmark_iterations):
            logger.info("Benchmark iteration %d/%d starting", iteration + 1, args.benchmark_iterations)
            start = time.monotonic()
            target = np.zeros_like(previous)
            target[:args.rtc_delay_steps] = previous[args.steps_to_execute:args.steps_to_execute + args.rtc_delay_steps]
            previous = runner._run_inference(observation, target, args.rtc_delay_steps)
            durations.append(time.monotonic() - start)
            logger.info("Benchmark iteration %d/%d: %.4f seconds", iteration + 1, args.benchmark_iterations, durations[-1])
        delay = int(np.ceil(max(durations) * args.control_hz))
        logger.info(
            "Model-only RTC timings=%s seconds; conservative delay=%d ticks at %.1fHz",
            durations,
            delay,
            args.control_hz,
        )
        if delay > args.rtc_delay_steps:
            raise RuntimeError(
                f"Model delay {delay} exceeds clean prefix {args.rtc_delay_steps}; lower --control-hz "
                f"below {args.rtc_delay_steps / max(durations):.1f}Hz and allow time for camera/SDK calls"
            )
        logger.info(
            "Benchmark complete: measured delay %d <= clean prefix %d; horizon=%d",
            delay,
            args.rtc_delay_steps,
            runner.action_horizon,
        )
        return
    runner.run()


if __name__ == "__main__":
    main()
