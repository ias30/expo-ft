from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rtc_previous_actions: np.ndarray | None = None,
        rtc_weights: np.ndarray | None = None,
        rtc_max_guidance: float = 5.0,
        rtc_prefix_length: int | None = None,
    ) -> dict:  # type: ignore[misc]
        """Infer actions, optionally conditioned on absolute physical RTC prefix targets.

        rtc_weights has the same shape as rtc_previous_actions. The normal input
        transforms rebase targets against obs and normalize them; padded model
        dimensions receive zero guidance. This extension supports JAX Pi0 only.

        For an RTC-SFT model, rtc_prefix_length selects clean-prefix clamping
        instead of guidance. It must be within the trained delay range, and
        rtc_weights must be omitted. Prefix targets use the same transforms.
        """
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if rtc_previous_actions is not None:
            if self._is_pytorch_model or not isinstance(self._model, Pi0):
                raise ValueError("RTC requires a JAX Pi0/Pi0.5 policy")
            previous = np.asarray(rtc_previous_actions, dtype=np.float32)
            native_rtc = rtc_prefix_length is not None
            if native_rtc:
                if (
                    not self._model.rtc_enabled
                    or not isinstance(rtc_prefix_length, (int, np.integer))
                    or not 0 <= rtc_prefix_length <= self._model.rtc_max_delay
                    or rtc_weights is not None
                ):
                    raise ValueError("Native RTC prefix must be within the trained delay range, without weights")
                weights = np.zeros_like(previous)
            else:
                weights = np.asarray(rtc_weights, dtype=np.float32)
            if (
                previous.ndim != 2
                or previous.shape != weights.shape
                or previous.shape[0] != self._model.action_horizon
                or not 0 < previous.shape[1] <= self._model.action_dim
            ):
                raise ValueError("RTC actions and weights must have matching (horizon, physical_dims) shapes")
            if (
                not np.isfinite(previous).all()
                or not np.isfinite(weights).all()
                or np.any((weights < 0) | (weights > 1))
            ):
                raise ValueError("RTC inputs must be finite with weights in [0, 1]")
            if not np.isfinite(rtc_max_guidance) or rtc_max_guidance <= 0:
                raise ValueError("RTC guidance must be finite and positive")
            # Absolute physical targets -> deltas relative to THIS observation -> normalization -> padding.
            inputs["actions"] = previous.copy()
        elif rtc_weights is not None or rtc_prefix_length is not None:
            raise ValueError("RTC weights/prefix length require previous actions")
        inputs = self._input_transform(inputs)
        rtc_target = inputs.pop("actions") if rtc_previous_actions is not None else None
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if rtc_target is not None:
            sample_kwargs['rtc_target'] = jnp.asarray(rtc_target)[None]
            if rtc_prefix_length is not None:
                # Dynamic scalar keeps boot (0) and streaming (d) on one JIT.
                sample_kwargs['rtc_prefix_length'] = jnp.asarray(rtc_prefix_length, dtype=jnp.int32)
            else:
                padded_weights = np.zeros_like(rtc_target, dtype=np.float32)
                padded_weights[:, : weights.shape[1]] = weights
                sample_kwargs.update(
                    rtc_weights=jnp.asarray(padded_weights)[None],
                    rtc_max_guidance=rtc_max_guidance,
                )
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
