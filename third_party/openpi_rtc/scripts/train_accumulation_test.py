"""Gradient accumulation uses micro batches but counts optimizer updates."""

import dataclasses
import functools
import os

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import utils

from . import train


class ToyModel(model_lib.BaseModel):
    def __init__(self):
        self.kernel = nnx.Param(jnp.array([[0.3]], dtype=jnp.float32))
        self.frozen = nnx.Param(jnp.array([[0.7]], dtype=jnp.float32))

    def compute_loss(self, rng, observation, actions, *, train):
        return (observation.state @ self.kernel.value + self.frozen.value - actions[:, 0]) ** 2

    def sample_actions(self, rng, observation, **kwargs):
        return (observation.state @ self.kernel.value + self.frozen.value)[:, None, :]


def make_state():
    graph, params = nnx.split(ToyModel())
    tx = optax.chain(optax.clip_by_global_norm(0.2), optax.adam(0.01))
    return utils.TrainState(
        step=jnp.array(0),
        params=params,
        model_def=graph,
        tx=tx,
        opt_state=tx.init(params.filter(nnx.Not(nnx_utils.PathRegex("frozen")))),
        ema_decay=0.9,
        ema_params=params,
    )


def make_batch(values):
    x = jnp.array(values, dtype=jnp.float32).reshape(-1, 1)
    return model_lib.Observation(images={}, image_masks={}, state=x), (2 * x)[:, None, :]


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("count", [1, 4])
def test_accumulation_matches_full_batch(count, offload):
    config = dataclasses.replace(
        config_lib.get_config("debug"),
        gradient_accumulation_steps=count,
        gradient_accumulation_offload=offload,
        freeze_filter=nnx_utils.PathRegex("frozen"),
    )
    micro_batches = [make_batch([i - 3.0, i + 1.0]) for i in range(count)]
    full_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs), *micro_batches)
    reference = make_state()
    accumulated = make_state()
    grad_fn = jax.jit(functools.partial(train.compute_gradients, config))
    update_fn = jax.jit(functools.partial(train.apply_accumulated_gradients, config))
    accum_fn = jax.jit(functools.partial(train.accumulate_gradients, config), donate_argnums=(5,))
    full_fn = jax.jit(functools.partial(train.train_step, config))
    rng = jax.random.key(42)
    # Multiple steps verify optimizer moments, EMA and step counting as well as weights.
    for step in range(3):
        iterator = iter(micro_batches[1:])
        accumulated, info = train.accumulated_train_step(
            config, rng, accumulated, micro_batches[0], iterator, grad_fn, update_fn, accum_fn
        )
        assert list(iterator) == []
        reference, reference_info = full_fn(rng, reference, full_batch)
        for actual, expected in zip(jax.tree.leaves(accumulated), jax.tree.leaves(reference), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(info["loss"], reference_info["loss"], rtol=1e-5)
        np.testing.assert_allclose(info["grad_norm"], reference_info["grad_norm"], rtol=1e-5)
        np.testing.assert_array_equal(accumulated.params["frozen"].value, [[np.float32(0.7)]])
        assert int(accumulated.step) == step + 1


def test_invalid_accumulation_rejected():
    config = dataclasses.replace(config_lib.get_config("debug"), gradient_accumulation_steps=0)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train.main(config)


@dataclasses.dataclass(frozen=True)
class TinyModelConfig(config_lib.pi0_config.Pi0Config):
    action_dim: int = 1
    action_horizon: int = 1
    max_token_len: int = 2

    def create(self, rng):
        return ToyModel()

    def inputs_spec(self, *, batch_size=1):
        return (
            model_lib.Observation(
                images={"base_0_rgb": jnp.zeros((batch_size, 8, 8, 3), jnp.float32)},
                image_masks={"base_0_rgb": jnp.ones((batch_size,), jnp.bool_)},
                state=jnp.zeros((batch_size, 1), jnp.float32),
            ),
            jax.ShapeDtypeStruct((batch_size, 1, 1), jnp.float32),
        )


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("ema", [None, 0.9])
def test_main_accumulation_checkpoint_resume(tmp_path, monkeypatch, ema, offload):
    """Use the real main loop, data loader, sharding, JIT donation and Orbax IO."""
    monkeypatch.setattr(train, "init_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.wandb, "Image", lambda *args, **kwargs: None)
    original_update = jax.config.update

    def update(name, value):
        if name == "jax_compilation_cache_dir":
            value = str(tmp_path / "jax_cache")
        original_update(name, value)

    monkeypatch.setattr(jax.config, "update", update)
    consumed = []
    original_iter = data_loader.TorchDataLoader.__iter__

    def counted_iter(loader):
        for batch in original_iter(loader):
            consumed.append(1)
            yield batch

    monkeypatch.setattr(data_loader.TorchDataLoader, "__iter__", counted_iter)
    saved_steps = []
    original_save = checkpoints.save_state

    def save(manager, state, loader, step):
        assert step == int(state.step)
        saved_steps.append(int(state.step))
        return original_save(manager, state, loader, step)

    monkeypatch.setattr(checkpoints, "save_state", save)
    config = dataclasses.replace(
        config_lib.get_config("debug"),
        model=TinyModelConfig(),
        gradient_accumulation_steps=4,
        gradient_accumulation_offload=offload,
        batch_size=2,
        num_workers=0,
        freeze_filter=nnx_utils.PathRegex("frozen"),
        ema_decay=ema,
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        exp_name="accum",
        num_train_steps=3,
        save_interval=2,
        log_interval=1,
        overwrite=False,
    )
    train.main(config)
    assert saved_steps[-1] == 3
    assert len(consumed) == 3 * 4
    train.main(dataclasses.replace(config, resume=True, num_train_steps=5))
    assert saved_steps[-1] == 5
    assert len(consumed) == 5 * 4


def make_small_pi05_state(monkeypatch, *, dtype="bfloat16", ema_decay=None):
    """Use production Pi0 modules/loss with smaller network dimensions for CPU tests."""
    from openpi.models import gemma
    from openpi.models import lora
    from openpi.models import siglip
    from openpi.training import sharding

    original_gemma = gemma.get_config
    original_vision = siglip.decode_variant

    def gemma_config(variant):
        tiny = original_gemma("dummy")
        if variant == "gemma_2b_lora_rank128":
            return dataclasses.replace(
                tiny,
                lora_configs={
                    "attn": lora.LoRAConfig(rank=128, alpha=128.0),
                    "ffn": lora.LoRAConfig(rank=128, alpha=128.0),
                },
            )
        return tiny

    monkeypatch.setattr(gemma, "get_config", gemma_config)
    monkeypatch.setattr(siglip, "decode_variant", lambda _: original_vision("mu/14"))
    target = config_lib.get_config("pi05_realman_tape2_rtc")
    config = dataclasses.replace(
        target,
        model=dataclasses.replace(target.model, action_horizon=2, rtc_max_delay=1, max_token_len=8, dtype=dtype),
        ema_decay=ema_decay,
        weight_loader=config_lib.weight_loaders.NoOpWeightLoader(),
        batch_size=1,
        gradient_accumulation_steps=4,
    )
    mesh = sharding.make_mesh(1)
    state, _ = train.init_train_state(config, jax.random.key(1), mesh, resume=False)
    return config, state


def test_pi05_loss_with_accumulation(monkeypatch):
    config, state = make_small_pi05_state(monkeypatch)
    batch = (config.model.fake_obs(), config.model.fake_act())
    grad_fn = jax.jit(functools.partial(train.compute_gradients, config))
    update_fn = jax.jit(functools.partial(train.apply_accumulated_gradients, config))
    accum_fn = jax.jit(functools.partial(train.accumulate_gradients, config), donate_argnums=(5,))
    before = {path: np.asarray(v.value).copy() for path, v in state.params.flat_state().items()}
    # Same update, different micro-step keys must change the stochastic loss/gradient.
    loss0, g0 = grad_fn(jax.random.key(2), state, batch, 0)
    loss1, g1 = grad_fn(jax.random.key(2), state, batch, 1)
    assert float(loss0[0] / loss0[1]) != float(loss1[0] / loss1[1])
    del g0, g1
    for _ in range(2):
        state, info = train.accumulated_train_step(
            config, jax.random.key(2), state, batch, iter([batch] * 3), grad_fn, update_fn, accum_fn
        )
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(info))
    assert int(state.step) == 2
    changed = []
    for path, variable in state.params.flat_state().items():
        if config.freeze_filter(path, variable):
            np.testing.assert_array_equal(variable.value, before[path])
        elif not np.array_equal(variable.value, before[path]):
            changed.append("/".join(map(str, path)))
    assert any("PaliGemma/img/" in p for p in changed)
    assert any("lora" in p for p in changed)
    assert any("/llm/" in p and "_1" in p for p in changed)


@pytest.mark.parametrize("offload", [False, True])
def test_pi05_accumulation_matches_full_batch(monkeypatch, offload):
    """Actual Pi0 loss: BS=4 vs BS=1 x 4, with controlled noise/time/augmentation.

    This is a reduced-width, randomly initialized float32 Pi0.5, NOT the full
    pretrained checkpoint. Random controls are test-only and restored by pytest.
    """
    config, initial = make_small_pi05_state(monkeypatch, dtype="float32", ema_decay=0.9)
    config = dataclasses.replace(config, gradient_accumulation_offload=offload)
    full_config = dataclasses.replace(config, batch_size=4, gradient_accumulation_steps=1)
    base_obs = config.model.fake_obs()
    base_actions = config.model.fake_act()
    batches = []
    for i in range(4):
        obs = dataclasses.replace(
            base_obs,
            images={k: jnp.full_like(v, (i - 1.5) / 4) for k, v in base_obs.images.items()},
            state=base_obs.state + i * 0.1,
        )
        batches.append((obs, base_actions + i * 0.2))
    full_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs), *batches)
    # Identical fixed noise and time for every sample, independent of batch shape.
    noise = jax.random.normal(jax.random.key(73), base_actions.shape)
    preprocess = model_lib.preprocess_observation
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, obs, **kw: preprocess(None, obs, train=False))
    original_normal = jax.random.normal

    def fixed_action_noise(key, shape, *args, **kwargs):
        if len(shape) == 3 and tuple(shape[1:]) == tuple(noise.shape[1:]):
            return jnp.broadcast_to(noise, shape)
        return original_normal(key, shape, *args, **kwargs)

    monkeypatch.setattr(jax.random, "normal", fixed_action_noise)
    monkeypatch.setattr(jax.random, "beta", lambda key, a, b, shape: jnp.full(shape, 0.4))
    monkeypatch.setattr(jax.random, "randint", lambda key, shape, minval, maxval, **kw: jnp.ones(shape, jnp.int32))
    grad_fn = jax.jit(functools.partial(train.compute_gradients, config))
    update_fn = jax.jit(functools.partial(train.apply_accumulated_gradients, config))
    accum_fn = jax.jit(functools.partial(train.accumulate_gradients, config), donate_argnums=(5,))
    full_fn = jax.jit(functools.partial(train.train_step, full_config))
    accumulated = reference = initial
    rng = jax.random.key(42)
    for step in range(3):
        accumulated, info = train.accumulated_train_step(
            config, rng, accumulated, batches[0], iter(batches[1:]), grad_fn, update_fn, accum_fn
        )
        reference, full_info = full_fn(rng, reference, full_batch)
        # Compare ALL state leaves: parameters, Adam moments/counter, EMA and step.
        for actual, expected in zip(jax.tree.leaves(accumulated), jax.tree.leaves(reference), strict=True):
            np.testing.assert_allclose(
                np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=1e-4, atol=1e-6
            )
        for name in ("loss", "grad_norm"):
            np.testing.assert_allclose(info[name], full_info[name], rtol=1e-4, atol=1e-6)
        assert int(accumulated.step) == step + 1
        for path, variable in accumulated.params.flat_state().items():
            if config.freeze_filter(path, variable):
                np.testing.assert_array_equal(variable.value, initial.params.flat_state()[path].value)
        max_error = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(jax.tree.leaves(accumulated.params), jax.tree.leaves(reference.params), strict=True)
        )
        print(f"offload={offload} update={step + 1} max_parameter_error={max_error:.3e}")
