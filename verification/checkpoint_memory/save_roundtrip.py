"""Restore a real training checkpoint and verify an isolated save round trip."""
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/openpi_rtc/src'),
                str(ROOT / 'third_party/openpi_rtc/packages/openpi-client/src')]

import jax
import numpy as np
from openpi.training import checkpoints, config, sharding


def digest(state):
    result = []
    for leaf in jax.tree.leaves(state):
        array = np.asarray(jax.device_get(leaf))
        result.append((str(array.dtype), array.shape, hashlib.sha256(array.tobytes()).hexdigest()))
    return result


def main():
    trainer = runpy.run_path(str(ROOT / 'third_party/openpi_rtc/scripts/train.py'))
    trainer['init_logging']()
    cfg = config.get_config('pi05_realman_tape2_rtc')
    source = ROOT / 'checkpoints/realman_rtc/pi05_realman_tape2_rtc/rtc_native_h64_d5_bs8_acc8'
    output = ROOT / 'verification/checkpoint_memory' / ('roundtrip_' + time.strftime('%Y%m%d_%H%M%S'))
    print('OUTPUT', output, flush=True)
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    shape, _ = trainer['init_train_state'](cfg, jax.random.key(cfg.seed), mesh, resume=True)
    src, _ = checkpoints.initialize_checkpoint_dir(source, keep_period=cfg.keep_period, overwrite=False, resume=True)
    dst, _ = checkpoints.initialize_checkpoint_dir(output, keep_period=None, overwrite=False, resume=False)
    try:
        print('RESTORE source step 3333', flush=True)
        state = checkpoints.restore_state(src, shape, None, step=3333)
        jax.block_until_ready(state)
        assert int(state.step) == 3333
        expected = digest(state)
        norm = checkpoints.load_norm_stats(source / '3333/assets', 'realman_tape2')
        loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=norm, asset_id='realman_tape2'))
        started = time.monotonic()
        checkpoints.save_state(dst, state, loader, 3333)
        save_seconds = time.monotonic() - started
        assert (output / '3333').is_dir()
        assert json.loads((output / '3333/_CHECKPOINT_METADATA').read_text())['commit_timestamp_nsecs']
        del state
        gc.collect()
        print('RESTORE saved checkpoint for full array comparison', flush=True)
        restored = checkpoints.restore_state(dst, shape, loader, step=3333)
        assert digest(restored) == expected
        assert (output / '3333/assets/realman_tape2/norm_stats.json').read_bytes() == (source / '3333/assets/realman_tape2/norm_stats.json').read_bytes()
        print(json.dumps({'result': 'PASS', 'step': int(restored.step), 'array_leaves_checked': len(expected), 'save_seconds': save_seconds, 'output': str(output)}), flush=True)
    finally:
        dst.close()
        src.close()


if __name__ == '__main__':
    main()
