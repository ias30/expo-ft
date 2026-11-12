"""Action timeline for RTC-SFT's fixed, clean prefix."""

import numpy as np

from openpi.policies.rtc_buffer import RTCBuffer


class TrainingRTCBuffer(RTCBuffer):
    def request(self, delay):
        offset = self.tick - self.origin
        remaining = self.actions[offset:]
        if not 1 <= delay < len(remaining):
            raise RuntimeError("Insufficient buffered actions for the clean RTC prefix")
        target = np.zeros_like(self.actions, dtype=np.float32)
        target[:delay] = remaining[:delay]
        self.last_request = self.tick
        self.pending_delay = delay
        return self.tick, target, delay

    def install(self, origin, actions):
        # Early replacement would execute predicted suffix actions before their
        # time. Late replacement would condition on fewer actions than executed.
        if origin != self.last_request or self.tick - origin != self.pending_delay:
            raise RuntimeError("RTC result must be installed exactly at the clean-prefix boundary")
        return super().install(origin, actions)
