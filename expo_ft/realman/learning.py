"""Stream UTD minibatches so only one critic batch is transferred at a time."""
import jax
import numpy as np


def update_from_replay(agent, replay, *, batch_size, utd_ratio):
    """U critic/filter steps followed by one policy step on the last minibatch.

    Base-policy BC uses its separate successful-window batch, when available.
    Returns (agent, None) if replay has no eligible critic windows. Metrics are
    from the last minibatch; every intermediate update is checked for finiteness.
    """
    if batch_size < 1 or utd_ratio < 1:
        raise ValueError('batch_size and utd_ratio must be positive')
    train_base_actor = agent.train_base_actor
    for index in range(utd_ratio):
        batch = replay.sample(batch_size)
        if batch is None:
            if index:
                raise RuntimeError('Replay became empty during a UTD call')
            return agent, None
        batch = jax.device_put(batch, agent.data_sharding)
        if index + 1 < utd_ratio:
            agent, metrics = agent.update_critic_only(batch)
        else:
            actor_batch = replay.sample(batch_size, success_only=True) if train_base_actor else None
            active = agent.replace(train_base_actor=train_base_actor and actor_batch is not None)
            active, metrics = active.update(active, batch, 1,
                None if actor_batch is None else jax.device_put(actor_batch, agent.data_sharding))
            agent = active.replace(train_base_actor=train_base_actor)
        jax.block_until_ready(metrics)
        if not all(np.isfinite(np.asarray(v)).all() for v in metrics.values()):
            raise FloatingPointError(f'Nonfinite RL metrics at UTD step {index + 1}/{utd_ratio}')
        del batch
    return agent, {**metrics, 'utd_ratio': utd_ratio, 'critic_steps_this_call': utd_ratio}
