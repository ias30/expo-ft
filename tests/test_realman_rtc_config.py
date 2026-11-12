"""Verify the deployed recipe and trainable parameter groups."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/openpi_rtc/src'))

import flax.nnx as nnx
import jax
from openpi.models import gemma
from openpi.training.config import get_config


def test_recipe():
    config = get_config('pi05_realman_tape2_rtc')
    assert (config.batch_size, config.gradient_accumulation_steps) == (16, 4)
    assert config.model.pi05 and config.model.rtc_enabled
    assert (config.model.action_horizon, config.model.rtc_max_delay) == (64, 5)
    assert config.model.action_expert_variant == 'gemma_300m'
    lora = gemma.get_config(config.model.paligemma_variant).lora_configs
    assert all(c.rank == 128 and c.alpha == 128 for c in lora.values())
    assert config.data.repo_id == 'realman_tape2'
    assert config.data.episodes == tuple(range(80))


def test_trainable_groups():
    config = get_config('pi05_realman_tape2_rtc')
    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    groups = set()
    for path, value in nnx.state(model, nnx.Param).flat_state().items():
        name = '/'.join(map(str, path))
        frozen = config.freeze_filter(path, value)
        if 'lora' in name:
            assert not frozen
            groups.add('lora')
        elif 'PaliGemma/img/' in name:
            assert not frozen
            groups.add('vision')
        elif '/llm/' in name and '_1' in name:
            assert not frozen
            groups.add('expert')
        elif '/llm/' in name:
            assert frozen
            groups.add('language_base')
        else:
            assert not frozen
    assert groups == {'lora', 'vision', 'expert', 'language_base'}
