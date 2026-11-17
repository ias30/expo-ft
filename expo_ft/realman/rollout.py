"""Episode collection: labels precede dispatch, pauses invalidate every old plan."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import logging
import numpy as np
import jax
from expo_ft.realman.replay import Episode
from expo_ft.realman.workspace import WorkspaceMotionRejected


class PlanningTimeout(TimeoutError):
    """Recoverable planning deadline; never used for an SDK/gripper timeout."""


class RTCPlanner:
    def __init__(self, agent, replan_steps, delay, only_base_actions=False):
        self.agent, self.r, self.d = agent, replan_steps, delay
        self.only_base = only_base_actions
        self.queue = deque()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def _precache(self, obs, prefix):
        chunks, updated = self.agent.sample_pre_cache(obs, prefix)
        return jax.block_until_ready(chunks), updated

    def action(self, observation):
        if not self.queue:
            if self.pending is None:
                chunks, updated = self._precache(observation, None)
                delay = 0
            else:
                if not self.pending.done():
                    raise PlanningTimeout('RTC candidates missed the fixed prefix deadline; stop and lower control Hz or N')
                chunks, updated = self.pending.result()
                self.pending = None
                delay = self.d
            self.agent = self.agent.replace(rng=updated.rng)
            actions, self.agent, _ = self.agent.sample_actions(
                observation, chunks, delay=delay, only_base_actions=self.only_base)
            actions = np.asarray(jax.device_get(actions))
            if actions.shape != (self.r, 7) or not np.isfinite(actions).all():
                raise ValueError('Invalid selected action window')
            self.queue.extend(actions)
        if len(self.queue) == self.d and self.d and self.pending is None:
            # Store physical targets; the actor rebases these against this observation.
            prefix = np.stack(self.queue)
            self.pending = self.executor.submit(self._precache, observation, prefix)
        return self.queue.popleft()

    def invalidate(self):
        self.queue.clear()
        if self.pending is not None:
            self.pending.cancel()
            # The robot is stopped at pause/episode boundaries. Drain before a new boot.
            if not self.pending.cancelled():
                self.pending.result()
            self.pending = None

    def close(self):
        try:
            self.invalidate()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)


def collect_episode(env, planner, *, control_hz=30, max_steps=600, timeout_ms=100,
                    clock=time.monotonic, sleep=time.sleep):
    """Caller catches exceptions and persists ``env.last_episode`` as aborted.

    No initial dummy command and no implicit reset. Human must start each episode.
    """
    if max_steps < 1 or control_hz <= 0:
        raise ValueError('max_steps and control_hz must be positive')
    obs = env.get_observation()
    ep = env.last_episode = Episode(obs)
    deadline = None
    pending_transition = None
    try:
        while True:
            if deadline is not None:
                sleep(max(0.0, deadline - clock()))
            if pending_transition is not None:
                # One feedback/image read serves both the previous transition
                # and this tick's policy input. Pause feedback is read only
                # after the old plan is drained, before boot inference.
                obs = env.get_observation()
                actual, paused = pending_transition
                ep.append(actual, obs, reset_plan=paused)
                pending_transition = None
            # Label the last dispatched action BEFORE any new planning/motion.
            done, success, _, _ = env.get_info_for_step()
            if done:
                ep.finish(success)
                break
            if len(ep.actions) >= max_steps:
                ep.finish(truncated=True, end_reason='step_limit')
                break
            was_boot = not planner.queue and planner.pending is None
            action = planner.action(obs)
            # Long JIT/inference must not swallow a queued terminal label.
            done, success, _, _ = env.get_info_for_step()
            if done:
                ep.finish(success)
                break
            if deadline is not None and not was_boot and clock() - deadline > timeout_ms / 1000:
                raise PlanningTimeout('Control dispatch deadline missed')
            dispatched = clock()
            try:
                env.diagnostic_step = len(ep.actions)
                actual, paused = env.step(action)
            except WorkspaceMotionRejected as exc:
                # Rejected command was never sent. Retain only completed
                # transitions; the cutoff window is excluded by replay.
                ep.finish(truncated=True, end_reason='workspace_rejected')
                logging.warning('Episode truncated before dispatch: %s', exc)
                break
            pending_transition = (np.array(actual, copy=True), paused)
            if paused:
                env.stop()
                planner.invalidate()
                deadline = None
            else:
                deadline = dispatched + 1 / control_hz
        return ep
    except BaseException:
        ep.finish(False, aborted=True)
        raise
    finally:
        try:
            try:
                env.stop()
            finally:
                planner.close()
        except BaseException:
            ep.finish(aborted=True, end_reason='cleanup_failed')
            raise
