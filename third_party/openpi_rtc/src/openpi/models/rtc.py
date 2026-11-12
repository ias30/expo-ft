"""Inference-time RTC, arXiv:2506.07339v1 equations (2)-(5).

OpenPI integrates noise-time t from 1 to 0 (paper tau = 1-t).
"""

import jax
import jax.numpy as jnp
import numpy as np


def prefix_weights(horizon: int, overlap: int, delay: int) -> np.ndarray:
    """Full weight for committed prefix, exponential soft overlap, zero new tail."""
    if not 0 <= delay <= overlap <= horizon:
        raise ValueError("RTC requires 0 <= delay <= overlap <= horizon")
    i = np.arange(horizon)
    c = np.clip((overlap - i) / (overlap - delay + 1), 0, 1)
    return np.where(i < delay, 1.0, np.where(i < overlap, c * np.expm1(c) / np.expm1(1.0), 0)).astype(np.float32)


def guided_velocity(velocity_fn, x, time, target, weights, max_guidance):
    """PiGDM VJP guidance; negative sign accounts for OpenPI's reverse time."""

    def denoise(z):
        v = velocity_fn(z)
        return z - time * v, v

    clean, pullback, velocity = jax.vjp(denoise, x, has_aux=True)
    correction = pullback((target - clean) * weights)[0]
    # t / ((1-t) * r^2) = (t^2 + (1-t)^2) / (t*(1-t)).
    scale = jnp.minimum(max_guidance, (time**2 + (1 - time) ** 2) / jnp.maximum(time * (1 - time), 1e-6))
    return velocity - scale * correction
