"""RTC training regression checks without loading pretrained weights."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma, pi0_config


def test_adarms_token_conditions_match_broadcast_and_have_gradients():
    norm = gemma.RMSNorm()
    x = jnp.ones((2, 7, 16))
    cond = jax.random.normal(jax.random.key(1), (2, 16))
    params = norm.init(jax.random.key(0), x, cond)
    expected = norm.apply(params, x, cond)
    tokens = jnp.repeat(cond[:, None], 7, axis=1)
    actual = norm.apply(params, x, tokens)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, jnp.broadcast_to(b, a.shape))
    grads = jax.grad(lambda p: norm.apply(p, x, tokens)[0].sum())(params)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(grads))


@pytest.mark.parametrize('delay', [0, 5])
def test_rtc_loss_and_backward_shapes(delay):
    config = pi0_config.Pi0Config(
        pi05=True, paligemma_variant='dummy', action_expert_variant='dummy',
        action_horizon=64, rtc_enabled=True, rtc_max_delay=delay,
    )
    # Abstract execution covers model shape/type checks and autodiff without
    # allocating the production SigLIP encoder or compiling a full train step.
    def forward_backward(rng):
        model = config.create(rng)
        obs, actions = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)
        return nnx.value_and_grad(
            lambda m: m.compute_loss(rng, obs, actions).mean()
        )(model)
    loss, grads = nnx.eval_shape(forward_backward, jax.random.key(0))
    assert loss.shape == ()
    assert len(jax.tree.leaves(grads)) > 0


def test_invalid_prefix_config():
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(pi05=True, rtc_enabled=True, action_horizon=5, rtc_max_delay=5)
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(rtc_enabled=True)


@pytest.mark.parametrize('max_delay', [0, 5])
def test_clean_prefix_and_suffix_loss(monkeypatch, max_delay):
    from types import SimpleNamespace
    from openpi.models import model as model_lib, pi0

    monkeypatch.setattr(model_lib, 'preprocess_observation', lambda rng, obs, train: obs)
    captured = {}
    batch, horizon, dim = 16, 64, 4
    actions = jnp.ones((batch, horizon, dim))

    def embed_suffix(obs, x, time):
        captured.update(x=x, time=time)
        return x, jnp.ones((batch, horizon), bool), jnp.zeros(horizon, bool), None

    def llm(tokens, **kwargs):
        return (tokens[0], jnp.zeros_like(tokens[1])), None

    model = SimpleNamespace(
        rtc_enabled=True, rtc_max_delay=max_delay, action_horizon=horizon,
        embed_prefix=lambda obs: (jnp.zeros((batch, 1, dim)), jnp.ones((batch, 1), bool), jnp.zeros(1, bool)),
        embed_suffix=embed_suffix, PaliGemma=SimpleNamespace(llm=llm), action_out_proj=lambda x: x,
    )
    rng = jax.random.key(12)
    losses = pi0.Pi0.compute_loss(model, rng, None, actions)
    mask = captured['time'] > 0
    np.testing.assert_array_equal(np.asarray(captured['x'])[~mask], np.asarray(actions)[~mask])
    np.testing.assert_array_equal(np.asarray(losses)[~mask], 0)
    _, noise_rng, _, _ = jax.random.split(rng, 4)
    error = jnp.mean((jax.random.normal(noise_rng, actions.shape) - actions) ** 2, axis=-1)
    expected = jnp.sum(error * mask) / jnp.sum(mask)
    np.testing.assert_allclose(losses.mean(), expected, rtol=1e-6)
    if max_delay:
        assert (~mask).any()
    else:
        np.testing.assert_allclose(losses, error)
