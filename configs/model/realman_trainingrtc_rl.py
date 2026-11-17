"""Small initial RealMan online RL configuration; profile before increasing N/batch."""
from configs.model.realtime_expo_ft_pi_config import get_config as base_config


def get_config():
    config = base_config()
    config.pi05_config_name = 'pi05_realman_tape2_rtc'
    config.pi05_asset_id = 'realman_tape2'
    config.train_base_actor = False
    config.N = 2
    config.n_edit_samples = 2
    config.filter_N = 32
    config.filter_n_edit = 1
    config.edit_scale = .05
    config.use_full_augmentation = False
    config.num_qs = 10
    config.num_min_qs = 2
    return config
