"""Anchor-only Scaffold-GS initialization experiment (not Gaussian distillation)."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_tools.dataset_registry import has_colmap_model, file_signature
from pipeline_tools.recovery_state import atomic_json, scene_identity, code_identity, fingerprint, StageJournal
from pipeline_tools.gpu_lock import gpu_lock


def export_anchors(source, scene, destination):
    import numpy as np
    from scipy.spatial import cKDTree
    from plyfile import PlyData, PlyElement
    vertex = PlyData.read(str(source), mmap=False)['vertex'].data
    if not any(name.startswith('f_anchor_feat_') for name in vertex.dtype.names):
        raise ValueError('Expected Scaffold-GS anchor PLY, not a vanilla Gaussian PLY.')
    points = np.column_stack([vertex[k] for k in ('x', 'y', 'z')])
    if not np.isfinite(points).all():
        raise ValueError('Scaffold anchors contain non-finite coordinates.')
    points = np.unique(points, axis=0)
    if len(points) < 2:
        raise ValueError('At least two distinct anchors are required.')
    spec = importlib.util.spec_from_file_location('colmap_points', ROOT / 'gaussian_splatting/scene/colmap_loader.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    sparse = Path(scene) / 'sparse/0'
    xyz, rgb, _ = (loader.read_points3D_binary(str(sparse / 'points3D.bin'))
                   if (sparse / 'points3D.bin').exists() else loader.read_points3D_text(str(sparse / 'points3D.txt')))
    if not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise ValueError('COLMAP points/colors are empty or invalid.')
    _, indices = cKDTree(xyz).query(points)
    data = np.zeros(len(points), dtype=[(k, 'f4') for k in ('x', 'y', 'z', 'nx', 'ny', 'nz')] +
                    [(k, 'u1') for k in ('red', 'green', 'blue')])
    for i, key in enumerate(('x', 'y', 'z')):
        data[key] = points[:, i]
    for i, key in enumerate(('red', 'green', 'blue')):
        data[key] = np.clip(rgb[indices, i], 0, 255).astype('u1')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.ply')
    PlyData([PlyElement.describe(data, 'vertex')]).write(str(temporary))
    temporary.replace(destination)
    return {'point_count': len(points), 'representation': 'anchors', 'color_source': 'nearest_colmap_point',
            'offsets_and_view_dependent_attributes_transferred': False}


def prepare(scene, output, repo, python=sys.executable, iterations=30000, resolution=1, gpu=0, resume=True):
    scene, output, repo = Path(scene).resolve(), Path(output).resolve(), Path(repo).resolve()
    if not has_colmap_model(scene):
        raise ValueError('Complete COLMAP images/ and sparse/0 are required.')
    for name in ('train.py', 'scene/gaussian_model.py', 'arguments/__init__.py'):
        if not (repo / name).is_file():
            raise FileNotFoundError(f'Scaffold-GS repository incomplete: {repo / name}')
    if iterations < 1 or resolution < 1 or gpu < 0:
        raise ValueError('Invalid Scaffold iterations, resolution, or GPU.')
    identity = {'scene': scene_identity(scene), 'iterations': iterations, 'resolution': resolution,
                'repo': str(repo), 'python': str(python), 'variant': 'anchors_nearest_colmap_rgb_v1',
                'code': code_identity(repo, [str(p.relative_to(repo)) for p in repo.rglob('*.py') if '.git' not in p.parts]),
                'adapter': code_identity(ROOT, ['pipeline_tools/scaffold_init.py', 'pipeline_tools/scaffold_train.py'])}
    run = output / 'runs' / fingerprint(identity)
    if not resume:
        import uuid
        run = run / uuid.uuid4().hex
    journal = StageJournal(run / 'stages', identity, enabled=resume)
    trained = journal.get('training', {})
    if not trained:
        run.mkdir(parents=True, exist_ok=True)
        command = [str(python), '-u', str(ROOT / 'pipeline_tools/scaffold_train.py'),
                   '--repo', str(repo), '-s', str(scene), '-m', str(run),
                   '--iterations', str(iterations), '-r', str(resolution), '--eval']
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        started = time.monotonic()
        with gpu_lock(gpu):
            subprocess.run(command, cwd=str(repo), env=env, check=True)
        trained = run / f'point_cloud/iteration_{iterations}/point_cloud.ply'
        journal.complete('training', {}, trained)
        atomic_json(run / 'training_receipt.json', {'identity': identity, 'seconds': time.monotonic() - started})
    inputs = {'anchor_ply': file_signature(trained)}
    converted = journal.get('conversion', inputs)
    if not converted:
        converted = run / 'initial_points.ply'
        details = export_anchors(trained, scene, converted)
        atomic_json(run / 'initialization.json', dict(details, identity=identity, anchor_ply=str(trained)))
        journal.complete('conversion', inputs, converted)
    return str(converted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--resolution', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    print(prepare(**vars(args)))
