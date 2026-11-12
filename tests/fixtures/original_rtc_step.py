"""Frozen pre-refactor EXPO-FT RTC function for numerical parity tests."""
import dataclasses
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.utils as training_utils

def _forward_get_velocity(model, observation, x_t, time):
    """v_t for a pre-processed observation: the velocity half of Pi0.compute_loss."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    return model.action_out_proj(suffix_out[:, -model.action_horizon:])

def train_step_p1_prefix(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple,
    delay_spec: jnp.ndarray,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Prefix-conditioned flow-matching step (RTC-SFT, arXiv 2512.05964).

    `batch = (observation, padded_actions)` with padded_actions (B, H, D_pad). `delay_spec`
    sets the per-example prefix length d: a scalar draws d ~ Unif{0..delay_spec} (offline
    RTC-SFT); a (B,) array is used as d directly (online fine-tuning passes the deployed
    delay, 0 for the boot chunk). The first d positions of x_t are the clean actions
    (per-token t=0) and only the remaining H-d positions are noised and supervised. d=0 is
    vanilla pi05 BC; d=replan_steps is the prefix-inpainted regime.
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        obs = _model.preprocess_observation(preprocess_rng, observation, train=True)
        B, H, _ = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)
        # Same τ distribution as pi05's default compute_loss.
        time_b = jax.random.beta(time_rng, 1.5, 1, (B,)) * 0.999 + 0.001  # (B,)
        if delay_spec.ndim == 0:
            d = jax.random.randint(delay_rng, (B,), 0, delay_spec + 1)    # (B,)
        else:
            d = delay_spec                                                # (B,)
        # postfix_mask: (B, H) — 0 on first d_i positions (prefix), 1 on the rest.
        postfix_mask = (jnp.arange(H)[None, :] >= d[:, None]).astype(jnp.float32)
        # Per-token τ: prefix = 0 (clean), postfix = per-batch scalar.
        time_per_pos = time_b[:, None] * postfix_mask                     # (B, H)
        t_exp = time_per_pos[..., None]                                   # (B, H, 1)
        x_t = t_exp * noise + (1 - t_exp) * actions                       # prefix slots -> actions
        # u_t on prefix slots is unused: the loss is masked to the postfix.
        u_t = noise - actions
        v_t = _forward_get_velocity(model, obs, x_t, time_per_pos)        # (B, H, D)
        per_pos = jnp.mean(jnp.square(v_t - u_t), axis=-1)                # (B, H)
        masked = per_pos * postfix_mask                                   # (B, H)
        denom = jnp.maximum(postfix_mask.sum(), 1.0)
        return masked.sum() / denom

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    lr_schedule_fn = config.lr_schedule.create()
    current_lr = lr_schedule_fn(state.step)
    info = {
        "actor_loss": loss,
        "actor_grad_norm": optax.global_norm(grads),
        "actor_param_norm": optax.global_norm(kernel_params),
        "actor_lr": current_lr,
        "actor_state_step": state.step,
    }
    return new_state, info


