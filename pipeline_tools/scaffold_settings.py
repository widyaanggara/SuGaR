"""Shared Scaffold-GS controls for the dashboard and training launcher."""
import math

DEFAULTS = dict(voxel_size=0.001, feat_dim=32, n_offsets=10, update_depth=3,
    update_init_factor=16, update_hierachy_factor=4, start_stat=500,
    update_from=1500, update_until=15000, update_interval=100,
    densify_grad_threshold=0.0002, min_opacity=0.005,
    success_threshold=0.8, lambda_dssim=0.2)


def scaffold_settings(data):
    result = {}
    for key, default in DEFAULTS.items():
        value = data.get('scaffold_' + key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Scaffold {key} harus berupa angka finite.')
        if isinstance(default, int) and not isinstance(value, int):
            raise ValueError(f'Scaffold {key} harus berupa bilangan bulat.')
        if key in ('min_opacity', 'success_threshold', 'lambda_dssim'):
            valid = 0 <= value <= 1
        elif key in ('start_stat', 'update_from', 'densify_grad_threshold'):
            valid = value >= 0
        else:
            valid = value > 0
        if not valid:
            raise ValueError(f'Nilai Scaffold {key} di luar rentang.')
        result['scaffold_' + key] = value
    if not result['scaffold_start_stat'] <= result['scaffold_update_from'] < result['scaffold_update_until']:
        raise ValueError('Scaffold memerlukan start_stat <= update_from < update_until.')
    depth = result['scaffold_update_depth']
    factor = result['scaffold_update_hierachy_factor']
    # Avoid zero-sized growth voxels and unbounded exponentiation.
    if depth > 32 or factor ** (depth - 1) > result['scaffold_update_init_factor']:
        raise ValueError('Hierarki Scaffold menghasilkan ukuran voxel nol; sesuaikan depth dan faktor pertumbuhan.')
    return result


def scaffold_arguments(data, upstream=False):
    return [arg for key, value in scaffold_settings(data).items()
            for arg in ('--' + (key[len('scaffold_'):] if upstream else key), str(value))]


def add_scaffold_arguments(parser):
    for key, default in DEFAULTS.items():
        parser.add_argument('--scaffold_' + key, type=type(default), default=default)
