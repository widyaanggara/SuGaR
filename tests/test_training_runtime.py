"""Exercise the real recovery helper using a tiny NumPy optimizer adapter.

The adapter replaces unavailable PyTorch/CUDA only; serialization, optimizer
restoration, RNG restoration and camera-cursor logic are the production code.
"""
import importlib.util
import pickle
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


class Tensor:
    def __init__(self, value, requires_grad=True):
        self.value = np.array(value.value if isinstance(value, Tensor) else value, copy=True)
        self.requires_grad = requires_grad
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim
    def __len__(self): return len(self.value)
    def to(self, device): return self
    def cpu(self): return self
    def tolist(self): return self.value.tolist()


class RNG:
    def __init__(self, state): self.state = state
    def cpu(self): return self


class Model:
    device = 'cpu'
    binded_to_surface_mesh = False
    def __init__(self): self._points = Tensor([[.1, .2, .3]])
    def state_dict(self): return {'_points': self._points}
    def named_parameters(self): return [('_points', self._points)]
    def load_state_dict(self, state): self._points.value[:] = state['_points'].value


class Optimizer:
    def __init__(self, model, tensor_rng):
        self.model, self.tensor_rng = model, tensor_rng
        self.current_iteration = 0
        self.values = {'state': {0: {'exp_avg': Tensor(np.zeros((1, 3))),
            'exp_avg_sq': Tensor(np.zeros((1, 3)))}}, 'param_groups': [{'name': 'points', 'params': [0]}]}
        self.optimizer = SimpleNamespace(param_groups=[{'name': 'points', 'params': [model._points]}])
    def state_dict(self): return self.values
    def load_state_dict(self, state): self.values = state
    def step(self, camera):
        # Momentum plus three independent random streams expose missing RNG/Adam restoration.
        noise = random.random() + np.random.random() + self.tensor_rng.random_sample()
        gradient = self.model._points.value - camera + noise * .01
        state = self.values['state'][0]
        state['exp_avg'].value = .9 * state['exp_avg'].value + .1 * gradient
        state['exp_avg_sq'].value = .99 * state['exp_avg_sq'].value + .01 * gradient ** 2
        self.current_iteration += 1
        self.model._points.value -= .01 * state['exp_avg'].value / (np.sqrt(state['exp_avg_sq'].value) + 1e-8)


class RuntimeTests(unittest.TestCase):
    def test_interrupted_optimizer_rng_and_camera_cursor_match_uninterrupted_steps(self):
        tensor_rng = np.random.RandomState(33)
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            get_rng_state=lambda: RNG(tensor_rng.get_state()),
            set_rng_state=lambda state: tensor_rng.set_state(state.state),
            randperm=lambda count: Tensor(tensor_rng.permutation(count)),
            isfinite=lambda value: np.isfinite(value.value), nn=SimpleNamespace(Parameter=Tensor),
            save=pickle.dump, load=lambda path, **kwargs: pickle.loads(Path(path).read_bytes()))
        path = Path(__file__).resolve().parents[1] / 'sugar_utils/training_recovery.py'
        spec = importlib.util.spec_from_file_location('training_recovery_cpu_test', path)
        runtime = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'torch': fake_torch}):
            spec.loader.exec_module(runtime)
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(recovery=dict(directory=folder, identity={'stage': 'coarse'}, resume=True))
            random.seed(31)
            np.random.seed(32)
            model = Model()
            optimizer = Optimizer(model, tensor_rng)
            recovery = runtime.SugarRecovery(args)
            order, _ = recovery.camera_order(4)
            for camera in order.tolist()[:3]: optimizer.step(camera)
            recovery.save(model, optimizer, None, 3, order, 3,
                {'loss': .5, 'epoch': 0, 'train_losses': [], 'current_sh_levels': 4,
                 'current_resolution_factor': 1}, force=True)
            def continue_steps(helper, opt, camera_order, offset, steps):
                visited = []
                while len(visited) < steps:
                    for camera in camera_order.tolist()[offset:]:
                        opt.step(camera)
                        visited.append(camera)
                        if len(visited) == steps: return visited
                    camera_order, offset = helper.camera_order(4)
            expected_cameras = continue_steps(recovery, optimizer, order, 3, 7)
            expected_weights = model._points.value.copy()
            expected_momentum = optimizer.values['state'][0]['exp_avg'].value.copy()
            # Simulate a fresh process consuming RNG during initialization.
            random.random(); np.random.random(); tensor_rng.random_sample()
            recovered_model = Model()
            recovered = runtime.SugarRecovery(args)
            recovered.restore_model(recovered_model)
            recovered_optimizer = Optimizer(recovered_model, tensor_rng)
            recovered.restore_training(recovered_optimizer)
            restored_order, offset = recovered.camera_order(4)
            self.assertEqual(offset, 3)
            actual_cameras = continue_steps(recovered, recovered_optimizer, restored_order, offset, 7)
            self.assertEqual(actual_cameras, expected_cameras)
            np.testing.assert_array_equal(recovered_model._points.value, expected_weights)
            np.testing.assert_array_equal(recovered_optimizer.values['state'][0]['exp_avg'].value, expected_momentum)
            self.assertEqual(recovered_optimizer.current_iteration, 10)


if __name__ == '__main__':
    unittest.main()
