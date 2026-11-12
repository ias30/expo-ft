"""Native RTC sampling, physical prefix transforms, and execution boundaries."""

from types import SimpleNamespace
import csv
import io

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as model_lib, pi0
from openpi.policies.policy import Policy
from openpi.policies.realman_policy import RealmanOutputs
from openpi.policies.training_rtc_buffer import TrainingRTCBuffer
from openpi.shared import nnx_utils
from scripts.train_accumulation_test import make_small_pi05_state
from expo_ft.rtc_sft import _forward_get_velocity


def test_native_sampler_with_production_modules(monkeypatch):
    config, state = make_small_pi05_state(monkeypatch, dtype="float32")
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    sample = nnx_utils.module_jit(model.sample_actions)
    obs = config.model.fake_obs()
    key = jax.random.key(10)
    noise = jax.random.normal(key, (1, 2, 32))
    target = jnp.full_like(noise, 0.25)
    original = sample(key, obs, num_steps=3, noise=noise)
    boot = sample(key, obs, num_steps=3, noise=noise, rtc_target=target, rtc_prefix_length=jnp.array(0))
    np.testing.assert_allclose(boot, original, rtol=1e-5, atol=1e-5)
    native = sample(key, obs, num_steps=3, noise=noise, rtc_target=target, rtc_prefix_length=jnp.array(1))
    np.testing.assert_array_equal(native[:, :1], target[:, :1])
    assert np.isfinite(native).all()

    @jax.jit
    def uncached_reference(params, observation, noise, target):
        reference_model = nnx.merge(state.model_def, params)
        reference_model.eval()
        observation = model_lib.preprocess_observation(None, observation, train=False)
        prefix = jnp.arange(2)[None, :, None] < 1
        x = jnp.where(prefix, target, noise)

        def step(i, x):
            times = jnp.where(prefix[..., 0], 0.0, 1.0 - i / 3.0)
            v = _forward_get_velocity(reference_model, observation, x, times)
            return jnp.where(prefix, target, x - v / 3.0)

        return jax.lax.fori_loop(0, 3, step, x)

    # Compare cached inference to the full attention forward used by training.
    expected = uncached_reference(state.params, obs, noise, target)
    np.testing.assert_allclose(native, expected, rtol=1e-5, atol=1e-5)


def test_every_denoising_step_uses_clean_prefix_and_zero_time(monkeypatch):
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, obs, train: obs)
    observed = []
    target = jnp.arange(24, dtype=jnp.float32).reshape(1, 6, 4)

    def embed_suffix(obs, actions, time):
        jax.debug.callback(lambda x, t: observed.append((np.array(x), np.array(t))), actions, time)
        return actions, jnp.ones((1, 6), bool), jnp.zeros(6, bool), None

    def llm(tokens, **kwargs):
        return (tokens[0], None if tokens[1] is None else jnp.ones_like(tokens[1])), None

    model = SimpleNamespace(
        rtc_enabled=True, action_horizon=6, action_dim=4,
        embed_prefix=lambda obs: (jnp.zeros((1, 1, 4)), jnp.ones((1, 1), bool), jnp.zeros(1, bool)),
        embed_suffix=embed_suffix, PaliGemma=SimpleNamespace(llm=llm), action_out_proj=lambda x: x,
    )
    obs = model_lib.Observation(images={}, image_masks={}, state=jnp.zeros((1, 4)))
    result = pi0.Pi0.sample_actions(
        model, jax.random.key(0), obs, num_steps=4, noise=jnp.zeros_like(target),
        rtc_target=target, rtc_prefix_length=jnp.array(5),
    )
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert len(observed) == 4
    for actions, time in observed:
        np.testing.assert_array_equal(actions[:, :5], target[:, :5])
        np.testing.assert_array_equal(time[:, :5], 0)
        assert (time[:, 5:] > 0).all()
    np.testing.assert_array_equal(result[:, :5], target[:, :5])
    np.testing.assert_allclose(result[:, 5:], -1)


def test_policy_prefix_rebased_normalized_and_restored(monkeypatch):
    config, state = make_small_pi05_state(monkeypatch, dtype="float32")
    model = nnx.merge(state.model_def, state.params)
    mask = transforms.make_bool_mask(6, -1)
    stats = {"actions": transforms.NormStats(mean=np.ones(7), std=np.ones(7) * 2)}
    policy = Policy(
        model,
        transforms=[transforms.DeltaActions(mask), transforms.Normalize(stats), transforms.PadStatesAndActions(32)],
        output_transforms=[transforms.Unnormalize(stats), transforms.AbsoluteActions(mask), RealmanOutputs()],
    )
    captured = {}

    def sample(key, obs, **kwargs):
        captured.update(kwargs)
        return kwargs["rtc_target"]

    monkeypatch.setattr(policy, "_sample_actions", sample)
    observation = jax.tree.map(lambda x: np.asarray(x[0]), config.model.fake_obs().to_dict())
    observation['state'] = np.arange(7, dtype=np.float32)
    previous = np.full((2, 7), 10, dtype=np.float32)
    output = policy.infer(observation, rtc_previous_actions=previous, rtc_prefix_length=1)
    expected = previous.copy()
    expected[:, :6] -= observation['state'][:6]
    expected = (expected - 1) / (2 + 1e-6)
    np.testing.assert_allclose(captured['rtc_target'][0, :, :7], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(captured['rtc_target'][0, :, 7:], 0)
    np.testing.assert_allclose(output['actions'], previous, atol=1e-5)
    np.testing.assert_array_equal(previous, 10)
    assert 'rtc_weights' not in captured
    assert int(captured['rtc_prefix_length']) == 1
    for length in (-1, 2, 1.5):
        with pytest.raises(ValueError, match="trained delay"):
            policy.infer(observation, rtc_previous_actions=previous, rtc_prefix_length=length)
    with pytest.raises(ValueError, match="without weights"):
        policy.infer(observation, rtc_previous_actions=previous, rtc_prefix_length=1, rtc_weights=np.ones((2, 7)))


def test_buffer_preserves_executed_prefix_and_rejects_early_late_install():
    actions = np.repeat(np.arange(64)[:, None], 7, axis=1)
    buffer = TrainingRTCBuffer(actions)
    for _ in range(8):
        buffer.pop()
    origin, target, length = buffer.request(5)
    assert origin == 8 and length == 5
    np.testing.assert_array_equal(target[:5], actions[8:13])
    replacement = actions + 100
    replacement[:5] = target[:5]
    with pytest.raises(RuntimeError, match="boundary"):
        buffer.install(origin, replacement)
    for expected in target[:5]:
        np.testing.assert_array_equal(buffer.pop(), expected)
    buffer.install(origin, replacement)
    np.testing.assert_array_equal(buffer.pop(), replacement[5])
    with pytest.raises(RuntimeError, match="boundary"):
        buffer.install(origin, replacement)


def test_entrypoint_defaults_and_initial_pose():
    import realman_inference_tape_trainingrtc as entry
    args = entry.parse_args(['--benchmark-only'])
    assert args.config_name == 'pi05_realman_tape2_rtc'
    assert args.checkpoint.endswith('rtc_native_h64_d5_bs8_acc8/10000')
    assert args.rtc and args.rtc_delay_steps == 5 and args.reset_pose
    assert entry.INITIAL_RIGHT_SPEED == 5
    np.testing.assert_allclose(
        entry.INITIAL_RIGHT_JOINTS_DEG,
        [109.89800262451172, 84.50399780273438, 52.55799865722656,
         -43.3380012512207, 68.61699676513672, -41.744998931884766],
        rtol=0, atol=0,
    )


@pytest.mark.parametrize('ready', [True, False])
def test_controller_waits_for_prefix_boundary_or_stops(monkeypatch, ready):
    import realman_inference_tape_trainingrtc as entry
    clock = [0.0]
    monkeypatch.setattr(entry, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], time=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    ))
    old = np.repeat(np.arange(64, dtype=np.float32)[:, None], 7, axis=1)
    new = old + 100
    submitted = []

    class Executor:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, observation, target, length):
            submitted.append((target.copy(), length))
            new[:length] = target[:length]
            return SimpleNamespace(done=lambda: ready, result=lambda: new)

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(entry, 'ThreadPoolExecutor', Executor)
    runner = entry.RealmanInference.__new__(entry.RealmanInference)
    runner.args = SimpleNamespace(control_hz=30, rtc_delay_steps=5, control_timeout_ms=100, steps_to_execute=8)
    runner.running = True
    runner.cameras = {}
    runner.gripper_target_state = 1
    runner._get_current_state = lambda: {}
    runner._run_inference = lambda obs: old
    runner._visualize = lambda: None
    executed, stopped = [], []
    runner.right_arm = SimpleNamespace(rm_set_arm_slow_stop=lambda: stopped.append(True) or 0)

    def execute(action):
        executed.append(action.copy())
        if len(executed) == 14:
            runner.running = False

    runner._execute_action = execute
    stream = io.StringIO()
    if ready:
        runner._run_rtc(csv.writer(stream), stream)
        np.testing.assert_array_equal(executed[:13], old[:13])
        np.testing.assert_array_equal(executed[13], new[5])
    else:
        with pytest.raises(RuntimeError, match='missed the 5-tick'):
            runner._run_rtc(csv.writer(stream), stream)
        np.testing.assert_array_equal(executed, old[:13])
    assert stopped
    assert len(submitted) == 1 and submitted[0][1] == 5
    np.testing.assert_array_equal(submitted[0][0][:5], old[8:13])
