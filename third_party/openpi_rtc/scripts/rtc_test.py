"""RTC integration with production Pi0.5 modules at CPU-sized network widths."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.policies.policy import Policy
from openpi.shared import nnx_utils
from scripts.train_accumulation_test import make_small_pi05_state


def test_pi05_guided_sampler_and_zero_mask_regression(monkeypatch):
    config, state = make_small_pi05_state(monkeypatch, dtype="float32")
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    sample = nnx_utils.module_jit(model.sample_actions)
    obs = config.model.fake_obs()
    noise = jax.random.normal(jax.random.key(9), (1, 2, 32))
    key = jax.random.key(10)
    reference = sample(key, obs, num_steps=3, noise=noise)
    zero = sample(
        key, obs, num_steps=3, noise=noise, rtc_target=jnp.zeros_like(noise), rtc_weights=jnp.zeros_like(noise)
    )
    np.testing.assert_allclose(zero, reference, rtol=1e-5, atol=1e-5)
    weights = jnp.zeros_like(noise).at[:, 0, :7].set(1)
    guided = sample(key, obs, num_steps=3, noise=noise, rtc_target=jnp.zeros_like(noise), rtc_weights=weights)
    assert np.isfinite(guided).all()
    assert not np.allclose(guided, reference)
    for before, after in zip(jax.tree.leaves(state.params), jax.tree.leaves(nnx.state(model)), strict=True):
        np.testing.assert_array_equal(before, after)


def test_policy_rebases_absolute_prefix_and_masks_padding(monkeypatch):
    config, state = make_small_pi05_state(monkeypatch, dtype="float32")
    model = nnx.merge(state.model_def, state.params)
    mask = transforms.make_bool_mask(6, -1)
    stats = {"actions": transforms.NormStats(mean=np.ones(7), std=np.ones(7) * 2)}
    policy = Policy(
        model,
        transforms=[transforms.DeltaActions(mask), transforms.Normalize(stats), transforms.PadStatesAndActions(32)],
    )
    captured = {}

    def sample(key, observation, **kwargs):
        captured.update(kwargs)
        return kwargs["rtc_target"]

    monkeypatch.setattr(policy, "_sample_actions", sample)
    # Use pre-tokenized dummy observation, while actions still use production transforms.
    observation = jax.tree.map(lambda x: np.asarray(x[0]), config.model.fake_obs().to_dict())
    observation["state"] = np.arange(7, dtype=np.float32)
    previous = np.full((2, 7), 10, dtype=np.float32)
    policy.infer(observation, rtc_previous_actions=previous, rtc_weights=np.ones((2, 7), dtype=np.float32))
    expected = previous.copy()
    expected[:, :6] -= observation["state"][:6]
    expected = (expected - 1) / (2 + 1e-6)
    np.testing.assert_allclose(captured["rtc_target"][0, :, :7], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(captured["rtc_weights"][0, :, 7:], 0)
    np.testing.assert_array_equal(previous, 10)
