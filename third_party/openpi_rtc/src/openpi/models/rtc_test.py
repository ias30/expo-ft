import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rtc import guided_velocity
from openpi.models.rtc import prefix_weights
from openpi.policies.rtc_buffer import RTCBuffer


def test_soft_mask_equation():
    w = prefix_weights(25, 17, 4)
    np.testing.assert_array_equal(w[:4], 1)
    np.testing.assert_array_equal(w[17:], 0)
    assert np.all(np.diff(w[4:17]) < 0)
    c = 13 / 14
    np.testing.assert_allclose(w[4], c * np.expm1(c) / np.expm1(1))
    with pytest.raises(ValueError, match="requires"):
        prefix_weights(25, 3, 4)


def test_vjp_sign_and_cross_time_jacobian():
    # Coupled velocity: guidance must differentiate through v, not just clamp x.
    matrix = jnp.array([[0.2, 0.3], [-0.4, 0.1]])
    x = jnp.array([0.1, -0.2])
    target = jnp.array([0.8, 0.6])
    weights = jnp.array([1.0, 0.0])
    t = 0.6
    v = matrix @ x
    jacobian = jnp.eye(2) - t * matrix
    scale = min(5.0, (t * t + (1 - t) ** 2) / (t * (1 - t)))
    expected = v - scale * jacobian.T @ ((target - (x - t * v)) * weights)
    actual = jax.jit(guided_velocity, static_argnums=0)(lambda z: matrix @ z, x, t, target, weights, 5.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    # At noise endpoint, clipped guidance stays finite and moves toward the target.
    guided = guided_velocity(jnp.zeros_like, x, 1.0, target, weights, 5.0)
    assert np.isfinite(guided).all()
    assert (x - 0.1 * guided)[0] > x[0]
    np.testing.assert_array_equal(guided_velocity(lambda z: z, x, t, target, jnp.zeros(2), 5.0), x)


def test_timeline_drops_elapsed_prefix_and_never_replays():
    old = np.repeat(np.arange(25)[:, None], 7, axis=1)
    buffer = RTCBuffer(old)
    for _ in range(8):
        buffer.pop()
    origin, target, weights = buffer.request(4)
    assert origin == 8
    np.testing.assert_array_equal(target[:17], old[8:])
    np.testing.assert_array_equal(weights[:4], 1)
    for _ in range(4):
        buffer.pop()
    new = old + 100
    assert buffer.install(origin, new) == 4
    np.testing.assert_array_equal(buffer.pop(), new[4])
    for _ in range(20):
        buffer.pop()
    with pytest.raises(RuntimeError, match="exhausted"):
        buffer.pop()
    with pytest.raises(RuntimeError, match="too late"):
        buffer.install(origin, new)
