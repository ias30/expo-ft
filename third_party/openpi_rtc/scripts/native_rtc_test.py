"""Parity with the frozen original EXPO-FT implementation, not a rewritten oracle."""
import ast
import dataclasses
import functools
import importlib.util
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from expo_ft import rtc_sft
from openpi.models import model as model_lib
from . import train
from .train_accumulation_test import make_small_pi05_state

_fixture = Path(__file__).resolve().parents[3] / 'tests/fixtures/original_rtc_step.py'
_spec = importlib.util.spec_from_file_location('original_rtc_step', _fixture)
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)
# Expose the original nested loss verbatim for direct gradient comparison.
_tree = ast.parse(_fixture.read_text())
_step = next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_step_p1_prefix')
_loss = next(n for n in _step.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_fn')
_loss.args.args.append(ast.arg(arg='delay_spec'))
exec(compile(ast.fix_missing_locations(ast.Module(body=[_loss], type_ignores=[])), str(_fixture), 'exec'), original.__dict__)


def assert_trees_close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), rtol=2e-4, atol=2e-6)


def test_original_loss_gradients_and_update(monkeypatch):
    config, state = make_small_pi05_state(monkeypatch, dtype='float32', ema_decay=0.9)
    config = dataclasses.replace(config, gradient_accumulation_steps=1)
    batch = config.model.fake_obs(batch_size=2), config.model.fake_act(batch_size=2)
    model = nnx.merge(state.model_def, state.params)
    model.train()
    rng = jax.random.key(91)
    for delay in (jnp.asarray(0), jnp.asarray(1), jnp.asarray([0, 1])):
        diff = nnx.DiffState(0, config.trainable_filter)
        native_loss, native_grad = nnx.value_and_grad(original.loss_fn, argnums=diff)(model, rng, *batch, delay)
        shared_loss, shared_grad = nnx.value_and_grad(rtc_sft.prefix_loss, argnums=diff)(model, rng, *batch, delay)
        assert_trees_close((shared_loss, shared_grad), (native_loss, native_grad))
        before, before_info = original.train_step_p1_prefix(config, rng, state, batch, delay)
        after, after_info = rtc_sft.train_step_p1_prefix(config, rng, state, batch, delay)
        assert_trees_close(after, before)
        assert_trees_close(after_info, before_info)
    expected, info = original.train_step_p1_prefix(config, rng, state, batch, jnp.asarray(1))
    actual, actual_info = train.train_step(config, rng, state, batch)
    assert_trees_close(actual, expected)
    np.testing.assert_allclose(actual_info['loss'], info['actor_loss'])


@pytest.mark.parametrize('offload', [False, True])
def test_unequal_suffix_accumulation_matches_native_full_batch(monkeypatch, offload):
    config, state = make_small_pi05_state(monkeypatch, dtype='float32', ema_decay=0.9)
    config = dataclasses.replace(config, gradient_accumulation_offload=offload)
    base_obs, base_actions = config.model.fake_obs(), config.model.fake_act()
    batches = [(dataclasses.replace(base_obs, state=base_obs.state + i * .1), base_actions + i * .2) for i in range(4)]
    full_batch = jax.tree.map(lambda *x: jnp.concatenate(x), *batches)
    preprocess = model_lib.preprocess_observation
    monkeypatch.setattr(model_lib, 'preprocess_observation', lambda rng, obs, **kw: preprocess(None, obs, train=False))
    noise = jax.random.normal(jax.random.key(5), base_actions.shape)
    original_normal = jax.random.normal
    def action_noise(key, shape, *args, **kwargs):
        if len(shape) == 3 and tuple(shape[1:]) == tuple(noise.shape[1:]):
            return jnp.broadcast_to(noise, shape)
        return original_normal(key, shape, *args, **kwargs)
    monkeypatch.setattr(jax.random, 'normal', action_noise)
    monkeypatch.setattr(jax.random, 'beta', lambda key, a, b, shape: jnp.full(shape, .4))
    # H=2: alternating prefixes give suffix counts 2,1,2,1. A simple
    # average of the four normalized gradients must NOT pass this test.
    delays = iter([0, 1, 0, 1])
    monkeypatch.setattr(jax.random, 'randint', lambda key, shape, minval, maxval, **kw: jnp.full(shape, next(delays), jnp.int32))
    actual, info = train.accumulated_train_step(
        config, jax.random.key(7), state, batches[0], iter(batches[1:]),
        functools.partial(train.compute_gradients, config),
        functools.partial(train.apply_accumulated_gradients, config),
        functools.partial(train.accumulate_gradients, config),
    )
    expected, expected_info = original.train_step_p1_prefix(
        config, jax.random.key(7), state, full_batch, jnp.asarray([0, 1, 0, 1]),
    )
    assert_trees_close(actual, expected)
    np.testing.assert_allclose(info['loss'], expected_info['actor_loss'], rtol=1e-5)
    np.testing.assert_allclose(info['grad_norm'], expected_info['actor_grad_norm'], rtol=1e-5)
    assert int(actual.step) == 1
