"""One action convention for rollout, replay, and Bellman backups."""
import jax.numpy as jnp
import numpy as np
from expo_ft.realman.gripper import OPEN_THRESHOLD, CLOSE_THRESHOLD


class ActionCodec:
    """Six absolute joints (rad) + absolute gripper, relative to a chunk anchor.

    Normalization exactly matches OpenPI (including its epsilon). No clipping:
    clipping normalized deltas would change the physical RTC prefix.
    """
    def __init__(self, stats, quantiles=True):
        if stats is None:
            raise ValueError('RealMan RL requires checkpoint normalization statistics')
        self.stats = stats
        self.quantiles = quantiles
        for key in ('actions', 'state'):
            if key not in stats:
                raise ValueError(f'Missing normalization statistics: {key}')
            s = stats[key]
            for value in ((s.q01, s.q99) if quantiles else (s.mean, s.std)):
                if value is None or len(value) != 7 or not np.isfinite(value).all():
                    raise ValueError(f'Expected seven finite {key} normalization values')

    def normalize(self, x, key):
        s = self.stats[key]
        x = jnp.asarray(x)[..., :7]
        if self.quantiles:
            return (x - jnp.asarray(s.q01)) / (jnp.asarray(s.q99) - jnp.asarray(s.q01) + 1e-6) * 2 - 1
        return (x - jnp.asarray(s.mean)) / (jnp.asarray(s.std) + 1e-6)

    def unnormalize(self, x, key):
        s = self.stats[key]
        x = jnp.asarray(x)[..., :7]
        if self.quantiles:
            return (x + 1) / 2 * (jnp.asarray(s.q99) - jnp.asarray(s.q01) + 1e-6) + jnp.asarray(s.q01)
        return x * (jnp.asarray(s.std) + 1e-6) + jnp.asarray(s.mean)

    @staticmethod
    def anchor(state):
        state = jnp.asarray(state)[..., :7]
        return state.at[..., 6].set(0)[..., None, :]

    def encode(self, absolute, raw_state):
        return self.normalize(jnp.asarray(absolute)[..., :7] - self.anchor(raw_state), 'actions')

    def decode(self, normalized, raw_state):
        return self.unnormalize(normalized, 'actions') + self.anchor(raw_state)

    def rebase(self, actions, old_normalized_state, new_normalized_state):
        old = self.unnormalize(old_normalized_state, 'state')
        new = self.unnormalize(new_normalized_state, 'state')
        return self.encode(self.decode(actions, old), new)


def flat_observation(obs):
    """Preserve training camera mapping and CHW RGB convention."""
    return {
        'observation.images.cam_high': np.asarray(obs['images']['cam_high']),
        'observation.images.cam_right_wrist': np.asarray(obs['images']['cam_right_wrist']),
        'observation.state': np.asarray(obs['state'], dtype=np.float32),
        'prompt': obs['prompt'],
    }


def critic_window(codec, flat_actions, normalized_state, lengths=None):
    """Canonical executed window: binary hysteresis gripper and a marked unused tail.

    Gripper transitions do not interrupt execution. Forward values are
    exactly the SDK mapping; a straight-through derivative lets edit learn the
    continuous gripper proposal. Tail gripper=-10 marks absence (valid binary
    gripper encodings are computed from stats); joints in the absent tail are 0.
    """
    import jax
    actions = flat_actions.reshape(flat_actions.shape[0], -1, 7)
    raw = codec.unnormalize(actions, 'actions')
    initial = codec.unnormalize(normalized_state, 'state')[..., 6] > .5
    def step(previous, command):
        current = jnp.where(previous, command > CLOSE_THRESHOLD, command >= OPEN_THRESHOLD)
        return current, current
    _, bits = jax.lax.scan(step, initial, jnp.swapaxes(raw[..., 6], 0, 1))
    bits = bits.T.astype(raw.dtype)
    horizon = actions.shape[1]
    first = jnp.full((actions.shape[0],), horizon) if lengths is None else lengths
    gripper = raw[..., 6] + jax.lax.stop_gradient(bits - raw[..., 6])
    encoded = codec.normalize(raw.at[..., 6].set(gripper), 'actions')
    valid = jnp.arange(horizon)[None, :] < first[:, None]
    # Explicit sentinel is outside the valid normalized binary gripper range.
    endpoints = codec.normalize(jnp.stack([jnp.zeros(7), jnp.ones(7)]), 'actions')[:, 6]
    sentinel = jnp.minimum(endpoints[0], endpoints[1]) - 10
    tail = jnp.zeros_like(encoded).at[..., 6].set(sentinel)
    return jnp.where(valid[..., None], encoded, tail).reshape(flat_actions.shape)
