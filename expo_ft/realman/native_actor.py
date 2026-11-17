"""Batched native RTC adapter; never imports the separate EXPO OpenPI fork."""
from functools import partial
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_lib
from expo_ft.agents.vla.pi05 import Pi05Agent, _jitted_pack_observation
from expo_ft.realman.actions import ActionCodec, flat_observation, critic_window


def realman_augmentation(rng, images):
    from expo_ft.utils.augmentation import random_crop
    result = dict(images)
    for key in ('base_0_rgb', 'right_wrist_0_rgb'):
        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, images[key].shape[0])
        result[key] = jax.vmap(lambda k, x: random_crop(k, x, 12))(keys, images[key])
    return result


@partial(jax.jit, static_argnames=('num_samples', 'num_steps'))
def native_sample(state, inputs, rng, prefix, *, num_samples, num_steps, noise=None, prefix_lengths=None):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    inputs = jax.tree.map(lambda x: None if x is None else jnp.repeat(x, num_samples, axis=0), inputs)
    # Backup helpers expose physical state dimensions; the model expects padding.
    inputs = dict(inputs)
    inputs['state'] = jnp.pad(inputs['state'], ((0, 0), (0, model.action_dim - inputs['state'].shape[-1])))
    prefix = jnp.repeat(prefix, num_samples, axis=0)
    d = prefix.shape[1]
    target = jnp.pad(prefix, ((0, 0), (0, model.action_horizon - d), (0, model.action_dim - prefix.shape[-1])))
    if noise is not None:
        noise = noise.reshape(-1, model.action_horizon, model.action_dim)
    return model.sample_actions(
        rng, model_lib.Observation.from_dict(inputs), num_steps=num_steps, noise=noise,
        rtc_target=target, rtc_prefix_length=(jnp.asarray(d) if prefix_lengths is None else jnp.repeat(prefix_lengths, num_samples)),
    )


class NativeRealmanActor(Pi05Agent):
    """Physical precache/prefix outside the model; normalized deltas inside it."""
    physical_precache = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = self.state_dim = 7
        self.action_codec = ActionCodec(self.data_config.norm_stats, self.data_config.use_quantile_norm)
        self.num_steps = 10

    def critic_actions(self, actions, state, lengths=None):
        return critic_window(self.action_codec, actions, state, lengths)

    def process_raw_inputs(self, raw_observations, action_dim=7, resize_size=224, normalize=True):
        raw = flat_observation(raw_observations) if 'images' in raw_observations else dict(raw_observations)
        raw['action'] = np.zeros((self.model_config.action_horizon, 7), np.float32)
        data = self.input_model_transforms(self.input_transforms(raw))
        data.pop('actions', None)
        return _jitted_pack_observation(data)

    def sample_training_actions(self, transformed_inputs, train_state, rng, train=False, num_samples=1, noise=None):
        b = transformed_inputs['state'].shape[0]
        empty = jnp.zeros((b, 0, self.model_config.action_dim))
        result = native_sample(train_state, transformed_inputs, rng, empty,
                               num_samples=num_samples, num_steps=self.num_steps, noise=noise)
        return result[..., :7], None

    def sample_actions(self, transformed_inputs, train_state, rng, train=False, num_samples=1):
        return self.sample_training_actions(transformed_inputs, train_state, rng, train, num_samples)

    def sample_training_actions_with_prefix(self, transformed_inputs, train_state, rng, prefix_padded,
                                            num_samples=1, noise=None, prefix_lengths=None):
        result = native_sample(train_state, transformed_inputs, rng, prefix_padded,
                               num_samples=num_samples, num_steps=self.num_steps, noise=noise, prefix_lengths=prefix_lengths)
        return result[..., :7], None

    def sample_actions_with_prefix(self, transformed_inputs, train_state, rng, prefix_padded, num_samples=1):
        # Rollout prefix is absolute, not a delta from the previous observation.
        anchor = self.action_codec.unnormalize(transformed_inputs['state'], 'state')
        prefix = self.action_codec.encode(jnp.asarray(prefix_padded)[None, ..., :7], anchor)
        return self.sample_training_actions_with_prefix(
            transformed_inputs, train_state, rng, prefix, num_samples)[0]

    def process_transformed_outputs(self, transformed_actions, unnormalize=True, *, raw_state=None):
        if raw_state is None:
            raise ValueError('RealMan output requires the observation used as action anchor')
        return self.action_codec.decode(transformed_actions, raw_state)
