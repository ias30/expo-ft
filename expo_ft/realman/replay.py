"""Bounded episode replay in physical coordinates, with explicit terminal observations."""
from collections import deque
from pathlib import Path
import os
import numpy as np
import jax
from openpi.shared import image_tools

CAMERAS = ('cam_high', 'cam_right_wrist')


def snapshot(obs):
    return {
        'state': np.array(obs['state'], np.float32, copy=True),
        'prompt': str(obs['prompt']),
        'images': {k: image_tools.resize_with_pad(np.asarray(obs['images'][k]).transpose(1, 2, 0),
                                                224, 224).transpose(2, 0, 1).copy() for k in CAMERAS},
    }


class Episode:
    def __init__(self, observation):
        self.observations = [snapshot(observation)]
        self.actions, self.pauses = [], []
        self.success = False
        self.finished = False
        self.aborted = False
        self.truncated = False
        self.end_reason = ''

    def append(self, action, next_observation, reset_plan=False):
        action = np.asarray(action, np.float32)
        if self.finished or action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Invalid episode append')
        self.actions.append(action.copy())
        self.pauses.append(bool(reset_plan))
        self.observations.append(snapshot(next_observation))

    def finish(self, success=False, aborted=False, truncated=False, end_reason=''):
        if truncated and (success or aborted):
            raise ValueError('A time-limit truncation cannot also be success or aborted')
        self.success, self.aborted, self.finished = bool(success), bool(aborted), True
        self.truncated = bool(truncated)
        self.end_reason = str(end_reason)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            state=np.stack([o['state'] for o in self.observations]),
            actions=np.asarray(self.actions, np.float32).reshape(-1, 7),
            pauses=np.asarray(self.pauses, bool), success=np.asarray(self.success),
            aborted=np.asarray(self.aborted), prompt=np.asarray(self.observations[0]['prompt']),
            truncated=np.asarray(self.truncated),
            end_reason=np.asarray(self.end_reason),
        )
        payload.update({k: np.stack([o['images'][k] for o in self.observations]) for k in CAMERAS})
        with open(str(path) + '.tmp', 'wb') as stream:
            np.savez_compressed(stream, **payload)
        os.replace(str(path) + '.tmp', path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            states, actions = data['state'], data['actions']
            if len(states) != len(actions) + 1 or actions.shape[1:] != (7,):
                raise ValueError(f'Invalid episode dimensions: {path}')
            # NPZ access decompresses a whole array; load each camera once,
            # rather than decompressing the complete episode for every frame.
            images = {k: data[k] for k in CAMERAS}
            prompt = str(data['prompt'])
            obs = [dict(state=s, prompt=prompt, images={k: images[k][i] for k in CAMERAS})
                   for i, s in enumerate(states)]
            ep = cls(obs[0])
            for i, action in enumerate(actions):
                ep.append(action, obs[i+1], bool(data['pauses'][i]))
            ep.finish(bool(data['success']), bool(data['aborted']),
                      truncated=bool(data['truncated']) if 'truncated' in data else False,
                      end_reason=str(data['end_reason']) if 'end_reason' in data else '')
            return ep


class RealmanReplay:
    def __init__(self, actor, capacity=20000, replan_steps=8, delay=5, discount=.99, seed=42):
        if not 0 <= delay <= replan_steps or capacity < actor.model_config.action_horizon:
            raise ValueError('Invalid replay capacity/delay')
        self.actor, self.capacity, self.r, self.delay, self.discount = actor, capacity, replan_steps, delay, discount
        self.episodes = deque()
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, episode):
        if not episode.finished:
            raise ValueError('Replay accepts only labelled, complete episodes')
        if episode.aborted or not episode.actions:
            return
        if len(episode.actions) > self.capacity:
            raise ValueError('Episode exceeds replay capacity; increase capacity or reduce max episode steps')
        while self.size + len(episode.actions) > self.capacity:
            self.size -= len(self.episodes.popleft().actions)
        self.episodes.append(episode)
        self.size += len(episode.actions)

    def eligible(self, success_only=False):
        horizon = self.actor.model_config.action_horizon if success_only else self.r
        candidates = []
        for ep in self.episodes:
            if success_only and not ep.success:
                continue
            for t in range(len(ep.actions)):
                end = min(t + horizon, len(ep.actions))
                if success_only:
                    if end-t != horizon or any(ep.pauses[t:end-1]):
                        continue
                elif ep.truncated and self._window_end(ep, t) == len(ep.actions):
                    # Unknown outcome at a time limit. Exclude these rows from
                    # ALL objectives (critic, filter, edit and temperature).
                    continue
                candidates.append((ep, t))
        return candidates

    def _window_end(self, ep, t):
        end = min(t + self.r, len(ep.actions))
        return next((i + 1 for i in range(t, end) if ep.pauses[i]), end)

    def sample(self, batch_size, success_only=False):
        candidates = self.eligible(success_only)
        if not candidates:
            return None
        H, D = self.actor.model_config.action_horizon, self.actor.model_config.action_dim
        rows = []
        for choice in self.rng.integers(len(candidates), size=batch_size):
            ep, t = candidates[choice]
            n = len(ep.actions)
            nxt = self._window_end(ep, t)
            duration = nxt - t
            boot = ep.pauses[nxt-1] or nxt == n
            delay = 0 if boot else self.delay
            delayed = nxt - delay
            obs, next_obs, delayed_obs = (ep.observations[i] for i in (t, nxt, delayed))
            current, following, earlier = (self.actor.process_raw_inputs(o) for o in (obs, next_obs, delayed_obs))
            row = {k: jax.tree.map(lambda x: np.asarray(x)[0], v) for k, v in current.items() if v is not None}
            for prefix, data in (('next_', following), ('next_delayed_', earlier)):
                for k in ('image', 'image_mask', 'state', 'tokenized_prompt', 'tokenized_prompt_mask'):
                    row[prefix + k] = jax.tree.map(lambda x: np.asarray(x)[0], data[k])
            actions = np.stack(ep.actions[t:min(t+H, n) if success_only else nxt])
            actions = np.pad(actions, ((0, H-len(actions)), (0, 0)), mode='edge')
            encoded = np.asarray(self.actor.action_codec.encode(actions, obs['state']))
            row['actions'] = np.pad(encoded, ((0, 0), (0, D-7)))
            terminal = nxt == n and not ep.truncated
            row['rewards'] = np.float32(self.discount ** (duration-1) if terminal and ep.success else 0)
            row['masks'] = np.float32(not terminal)
            row['dones'] = np.bool_(terminal)
            row['valids'] = np.float32(not (ep.truncated and nxt == n))
            row['window_length'] = np.int32(duration)
            row['bootstrap_discount'] = np.float32(self.discount ** duration)
            row['bootstrap_delay'] = np.int32(delay)
            last_reset = max([0] + [i+1 for i in range(t) if ep.pauses[i]])
            row['episode_step'] = np.int32(t-last_reset)
            rows.append(row)
        return jax.tree.map(lambda *xs: np.stack(xs), *rows)
