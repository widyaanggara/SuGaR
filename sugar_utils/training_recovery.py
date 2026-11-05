"""Full optimizer/RNG recovery used only inside training workers."""
import random
import os
import math
from pathlib import Path
import numpy as np
import torch

from pipeline_tools.recovery_state import CheckpointStore


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


class SugarRecovery:
    def __init__(self, args):
        options = getattr(args, 'recovery', None)
        self.store = CheckpointStore(options['directory'], {**options['identity'],
            'torch': str(getattr(torch, '__version__', 'unknown')), 'numpy': str(np.__version__)}) if options else None
        self.resume = bool(options and options['resume'])
        self.stage = options['identity'].get('stage', 'SuGaR') if options else 'SuGaR'
        self.state, self.iteration = None, 0
        self.order_pending = False

    def restore_model(self, model):
        if not self.store or not self.resume:
            return
        def validate(state, iteration):
            required = ('model', 'optimizer', 'rng', 'training', 'camera_order', 'camera_offset', 'buffers')
            if not all(key in state for key in required) or iteration < 0:
                raise ValueError('Checkpoint training SuGaR tidak lengkap.')
            saved, current = state['model'], model.state_dict()
            if saved.keys() != current.keys():
                raise ValueError('Parameter model SuGaR berbeda.')
            points = saved['_points']
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                raise ValueError('Checkpoint tidak memiliki titik 3D yang valid.')
            if not math.isfinite(state['training']['loss']):
                raise ValueError('Loss checkpoint tidak finite.')
            for name, value in saved.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f'Parameter {name} tidak finite.')
                if value.shape != current[name].shape:
                    if (model.binded_to_surface_mesh or '.' in name or value.ndim != current[name].ndim
                            or value.shape[1:] != current[name].shape[1:] or value.shape[0] != len(points)):
                        raise ValueError(f'Bentuk parameter {name} tidak cocok.')
            if not state['optimizer'].get('param_groups'):
                raise ValueError('State optimizer SuGaR kosong.')
        self.state, self.iteration = self.store.load(
            lambda path: torch.load(str(path), map_location='cpu', weights_only=False), validate)
        if self.state:
            parameters = dict(model.named_parameters())
            for name, value in self.state['model'].items():
                if name in parameters and parameters[name].shape != value.shape:
                    setattr(model, name, torch.nn.Parameter(value.to(model.device), requires_grad=parameters[name].requires_grad))
            model.load_state_dict(self.state['model'])
            for name in ('knn_idx', 'knn_dists'):
                if name in self.state['buffers']:
                    setattr(model, name, self.state['buffers'][name].to(model.device))
        else:
            print(f'[RESUME] Belum ada checkpoint berkala yang cocok untuk {self.stage}; '
                  'tahap ini mulai dari awal. File lama tanpa metadata tidak diasumsikan aman.', flush=True)

    def restore_training(self, optimizer, densifier=None):
        if not self.state:
            return
        actual = optimizer.state_dict()['param_groups']
        saved = self.state['optimizer']['param_groups']
        if [group['name'] for group in actual] != [group['name'] for group in saved]:
            raise ValueError('Kelompok optimizer tidak cocok dengan checkpoint.')
        for current, group in zip(optimizer.optimizer.param_groups, saved):
            for parameter, parameter_id in zip(current['params'], group['params']):
                for key, value in self.state['optimizer']['state'].get(parameter_id, {}).items():
                    if key in ('exp_avg', 'exp_avg_sq') and value.shape != parameter.shape:
                        raise ValueError('Ukuran state optimizer tidak cocok.')
        optimizer.load_state_dict(self.state['optimizer'])
        optimizer.current_iteration = self.state['optimizer_iteration']
        if densifier is not None:
            for name in ('points_gradient_accum', 'denom', 'max_radii2D'):
                if name in self.state['buffers']:
                    setattr(densifier, name, self.state['buffers'][name].to(densifier.model.device))
        restore_rng(self.state['rng'])
        self.order_pending = True
        for key in ('model', 'optimizer', 'buffers', 'rng'):
            self.state.pop(key, None)

    def camera_order(self, count):
        if self.order_pending:
            self.order_pending = False
            order, offset = self.state['camera_order'].cpu(), self.state['camera_offset']
            if len(order) != count or sorted(order.tolist()) != list(range(count)) or not 0 <= offset <= count:
                raise ValueError('Urutan kamera checkpoint tidak cocok dengan dataset.')
            return order, offset
        return torch.randperm(count), 0

    def save(self, model, optimizer, densifier, iteration, order, offset, training, force=False):
        if not self.store or (not force and iteration % 500):
            return
        buffers = {name: getattr(model, name) for name in ('knn_idx', 'knn_dists') if hasattr(model, name)}
        if densifier is not None:
            buffers.update({name: getattr(densifier, name) for name in ('points_gradient_accum', 'denom', 'max_radii2D')})
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'optimizer_iteration': optimizer.current_iteration, 'rng': rng_state(),
            'training': training, 'camera_order': order.cpu(), 'camera_offset': offset, 'buffers': buffers}
        self.store.save(state, iteration, torch.save)
        print(f'[CHECKPOINT] SuGaR iterasi {iteration} tersimpan atomik (dua slot).', flush=True)
