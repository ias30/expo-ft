"""Frozen-VLA checkpoints contain only trainable RL state and its targets."""
import hashlib
from pathlib import Path
import orbax.checkpoint as ocp
from expo_ft.agents.alg import realtime_expo_ft as algorithm

RL_FIELDS = ('batch_encoder', 'critic', 'target_critic', 'edit_actor', 'filter_critic', 'temp', 'rng')


def fingerprint(checkpoint):
    root = Path(checkpoint)
    digest = hashlib.sha256()
    for relative in ('_CHECKPOINT_METADATA', 'assets/realman_tape2/norm_stats.json'):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def create_manager(path, train_base_actor, *, read_only=False):
    keys = ('agent', 'params') if train_base_actor else ('rl',)
    return ocp.CheckpointManager(path,
        item_handlers={k: ocp.PyTreeCheckpointHandler(save_concurrent_gb=4, restore_concurrent_gb=4) for k in keys},
        options=ocp.CheckpointManagerOptions(max_to_keep=2, enable_async_checkpointing=False,
                                             read_only=read_only, create=not read_only))


def save(manager, agent, step):
    agent = agent.replace(_infer_cache=None)
    if agent.train_base_actor:
        algorithm.save_checkpoint(manager, agent, step)
    else:
        manager.save(step, {'rl': {k: getattr(agent, k) for k in RL_FIELDS}})
    manager.wait_until_finished()


def restore(manager, agent, step=None):
    agent = agent.replace(_infer_cache=None)
    if agent.train_base_actor:
        return algorithm.restore_checkpoint(manager, agent, step=step).cache_infer_params()
    template = {k: getattr(agent, k) for k in RL_FIELDS}
    restored = manager.restore(manager.latest_step() if step is None else step, items={'rl': template}, restore_kwargs={
        'rl': {'restore_args': ocp.checkpoint_utils.construct_restore_args(template)}})
    return agent.replace(**restored['rl']).cache_infer_params()
