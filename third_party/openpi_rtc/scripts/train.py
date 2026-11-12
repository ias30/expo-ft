import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

from expo_ft.rtc_sft import prefix_loss_terms, train_step_p1_prefix

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def compute_gradients(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    micro_step: int | at.Int[at.Array, ""] = 0,
) -> tuple[at.Array, Any]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    if config.gradient_accumulation_steps > 1:
        train_rng = jax.random.fold_in(train_rng, micro_step)
    observation, actions = batch

    # RTC accumulation differentiates the unnormalized numerator, carrying
    # its denominator separately across all microbatches.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    if getattr(config.model, "rtc_enabled", False):
        def numerator_fn(model):
            masked, count = prefix_loss_terms(
                model, train_rng, observation, actions, jnp.asarray(config.model.rtc_max_delay),
            )
            return masked.sum(), count

        (numerator, count), grads = nnx.value_and_grad(
            numerator_fn, argnums=diff_state, has_aux=True,
        )(model)
        return jnp.stack([numerator, count]), grads
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    return loss, grads


@at.typecheck
def apply_gradients(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    loss: at.Array,
    grads: Any,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Clip averaged gradients and update optimizer, parameters, step and EMA once."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def train_step(config, rng, state, batch):
    """Original single-batch path when accumulation is disabled."""
    if getattr(config.model, "rtc_enabled", False):
        state, info = train_step_p1_prefix(
            config, rng, state, batch, jnp.asarray(config.model.rtc_max_delay),
        )
        return state, {
            "loss": info["actor_loss"], "grad_norm": info["actor_grad_norm"],
            "param_norm": info["actor_param_norm"],
        }
    loss, grads = compute_gradients(config, rng, state, batch)
    return apply_gradients(config, state, loss, grads)


def accumulate_gradients(config, rng, state, batch, micro_step, total_loss, total_grads):
    """Fuse backward and accumulation so XLA can reuse the donated gradient buffer."""
    loss, grads = compute_gradients(config, rng, state, batch, micro_step)
    return total_loss + loss, jax.tree.map(lambda a, b: a + b, total_grads, grads)


def accumulated_train_step(config, rng, state, first_batch, data_iter, gradient_fn, update_fn, accumulate_fn):
    """Consume K batches at fixed parameters; return one optimizer update."""
    total_loss, total_grads = gradient_fn(rng, state, first_batch, 0)
    if config.gradient_accumulation_offload:
        # Copy onto writable host arrays and drop device gradient references BEFORE
        # launching the next backward pass. No accumulated gradient stays on GPU.
        total_loss = np.asarray(jax.device_get(total_loss)).copy()
        total_grads = jax.tree.map(lambda g: np.array(g, copy=True), jax.device_get(total_grads))
        for micro_step in range(1, config.gradient_accumulation_steps):
            loss, grads = gradient_fn(rng, state, next(data_iter), micro_step)
            host_grads = jax.device_get(grads)
            total_loss += np.asarray(jax.device_get(loss))
            del grads, loss

            def add_in_place(accumulated, gradient):
                np.add(accumulated, gradient, out=accumulated)
                return accumulated

            total_grads = jax.tree.map(add_in_place, total_grads, host_grads)
            del host_grads
        # Restore each gradient to the corresponding parameter's device/sharding.
        params = state.params.filter(config.trainable_filter)
        total_grads = jax.tree.map(lambda g, p: jax.device_put(g, p.sharding), total_grads, params)
        total_loss = jax.device_put(total_loss, state.step.sharding)
    else:
        for micro_step in range(1, config.gradient_accumulation_steps):
            total_loss, total_grads = accumulate_fn(rng, state, next(data_iter), micro_step, total_loss, total_grads)
    return update_fn(state, total_loss, total_grads)


def apply_accumulated_gradients(config, state, total_loss, total_grads):
    if getattr(config.model, "rtc_enabled", False):
        # Sum all suffix errors / sum all suffix counts, including when random
        # prefix lengths differ between microbatches. Clip only after dividing.
        numerator, count = total_loss[0], jnp.maximum(total_loss[1], 1.0)
        grads = jax.tree.map(lambda g: g / count, total_grads)
        return apply_gradients(config, state, numerator / count, grads)
    count = config.gradient_accumulation_steps
    grads = jax.tree.map(lambda g: g / count, total_grads)
    return apply_gradients(config, state, total_loss / count, grads)


def main(config: _config.TrainConfig):
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    init_logging()
    logging.info(
        "Micro batch=%d, accumulation=%d, effective batch=%d; steps count optimizer updates",
        config.batch_size,
        config.gradient_accumulation_steps,
        config.batch_size * config.gradient_accumulation_steps,
    )
    logging.info("Gradient accumulation offload to CPU: %s", config.gradient_accumulation_offload)
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pgradient_step = jax.jit(
        functools.partial(compute_gradients, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
    )
    paccumulate_step = jax.jit(
        functools.partial(accumulate_gradients, config),
        donate_argnums=(5,),
    )
    pupdate_step = jax.jit(
        functools.partial(apply_accumulated_gradients, config),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(0, 2),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            if config.gradient_accumulation_steps == 1:
                train_state, info = ptrain_step(train_rng, train_state, batch)
            else:
                train_state, info = accumulated_train_step(
                    config, train_rng, train_state, batch, data_iter, pgradient_step, pupdate_step, paccumulate_step
                )
        infos.append(info)
        completed_step = step + 1
        if completed_step == start_step + 1 or completed_step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {completed_step}: {info_str}")
            wandb.log(reduced_info, step=completed_step)
            infos = []
        if completed_step < config.num_train_steps:
            batch = next(data_iter)

        if completed_step % config.save_interval == 0 or completed_step == config.num_train_steps:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, completed_step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
