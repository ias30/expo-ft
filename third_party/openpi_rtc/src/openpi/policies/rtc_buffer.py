"""Absolute action timeline shared by the RTC controller and inference result."""

import numpy as np

from openpi.models.rtc import prefix_weights


class RTCBuffer:
    def __init__(self, actions):
        self.actions = np.asarray(actions).copy()
        self.origin = 0
        self.tick = 0
        self.last_request = 0

    def request(self, delay):
        """Snapshot only unconsumed actions; caller pairs this with a fresh observation."""
        offset = self.tick - self.origin
        remaining = self.actions[offset:]
        horizon, dims = self.actions.shape
        if len(remaining) <= delay:
            raise RuntimeError("RTC has insufficient overlap for inference delay; reduce replan interval or latency")
        target = np.zeros((horizon, dims), dtype=np.float32)
        target[: len(remaining)] = remaining
        weights = np.repeat(prefix_weights(horizon, len(remaining), delay)[:, None], dims, axis=1)
        self.last_request = self.tick
        return self.tick, target, weights

    def install(self, origin, actions):
        """Skip actions whose deadlines passed while the model was computing."""
        actions = np.asarray(actions)
        elapsed = self.tick - origin
        if actions.shape != self.actions.shape or not np.isfinite(actions).all():
            raise ValueError("Invalid RTC replacement chunk")
        if origin > self.tick or origin < self.origin or elapsed >= len(actions):
            raise RuntimeError("RTC result arrived too late or out of order")
        self.actions = actions.copy()
        self.origin = origin
        return elapsed

    def pop(self):
        index = self.tick - self.origin
        if index >= len(self.actions):
            raise RuntimeError("RTC action buffer exhausted; stopping instead of replaying old actions")
        action = self.actions[index].copy()
        self.tick += 1
        return action
