"""Artifact locations shared by the dashboard and external method runner."""
import json
import math
from pathlib import Path

METHODS = {'2dgs': ('2DGS', '2DGS'), 'radegs': ('RaDe-GS', 'RaDe-GS')}


def output_base(root, experiment='baseline'):
    if experiment not in ('baseline', 'scaffold_anchor'):
        raise ValueError('Eksperimen tidak dikenal.')
    return Path(root) / ('outputs/ablations/scaffold_anchor' if experiment == 'scaffold_anchor' else 'outputs')


def checkpoint(output):
    return Path(output) / 'point_cloud/iteration_30000/point_cloud.ply'


def trained(output):
    path = checkpoint(output)
    return path.is_file() and path.stat().st_size > 0 and (Path(output) / 'cfg_args').is_file()


def visual_metrics(output):
    try:
        values = json.loads((Path(output) / 'results.json').read_text(encoding='utf-8'))['ours_30000']
        if not all(isinstance(values[k], (int, float)) and not isinstance(values[k], bool)
                   and math.isfinite(values[k]) for k in ('PSNR', 'SSIM', 'LPIPS')):
            return None
        return values
    except (OSError, ValueError, KeyError, TypeError):
        return None


def mesh_path(output, method):
    return Path(output) / ('train/ours_30000/fuse_post.ply' if method == '2dgs' else 'recon_post.ply')
