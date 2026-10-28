import importlib.util
import ast
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from gaussian_splatting.utils.resolution_utils import scaled_resolution
from pipeline_tools.run_settings import (resolution_settings, colmap_matches, colmap_signature,
    training_matches, evaluation_matches, colmap_settings)
from pipeline_tools.colmap_capabilities import inspect_colmap, resolve_device
from web_dashboard.server import build_command, prepare_jobs


def write(path, content='data'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def load_colmap():
    path = Path(__file__).resolve().parents[1] / 'pipeline_tools/02_run_colmap.py'
    spec = importlib.util.spec_from_file_location('colmap_wrapper_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResolutionTests(unittest.TestCase):
    def test_3dgs_loader_uses_shared_size_and_accepts_legacy_config_without_new_fields(self):
        path = Path(__file__).resolve().parents[1] / 'gaussian_splatting/utils/camera_utils.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'loadCam')
        sizes = []
        class ImageTensor:
            shape = (3, 1200, 1600)
            def __getitem__(self, key): return self
        def resize(image, size):
            sizes.append(size)
            return ImageTensor()
        namespace = dict(scaled_resolution=scaled_resolution, PILtoTorch=resize,
                         Camera=lambda **kwargs: kwargs, WARNED=True)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        info = SimpleNamespace(image=SimpleNamespace(size=(3000, 4000)), uid=1, R=None, T=None,
                               FovX=1., FovY=1., image_name='test')
        args = SimpleNamespace(image_downscale=2, image_max_size=1600, resolution=-1, data_device='cpu')
        namespace['loadCam'](args, 0, info, 1)
        self.assertEqual(sizes[-1], (1200, 1600))
        args.image_max_size = None  # render.py loading a checkpoint written before this setting existed
        args.image_downscale = None
        namespace['loadCam'](args, 0, info, 1)
        self.assertEqual(sizes[-1], (1600, 2133))

    def test_same_aspect_preserving_size_for_landscape_portrait_and_small_photos(self):
        self.assertEqual(scaled_resolution(4000, 3000, 1, 1600), (1600, 1200))
        self.assertEqual(scaled_resolution(3000, 4000, 1, 1600), (1200, 1600))
        self.assertEqual(scaled_resolution(4000, 3000, 4, 1600), (1000, 750))
        self.assertEqual(scaled_resolution(4000, 3000, 2, 0), (2000, 1500))
        self.assertEqual(scaled_resolution(800, 600, 1, 1600), (800, 600))
        self.assertEqual(scaled_resolution(800, 600, 2, 1600), (400, 300))
        self.assertEqual(scaled_resolution(4000, 3000, 1, 1600, 2), (800, 600))

    def test_api_validates_and_persists_settings_and_commands_pass_them(self):
        settings = {'colmap_max_image_size': 2000, 'image_downscale': 4, 'image_max_size': 1280}
        for action in ('colmap', '3dgs', 'sugar', 'evaluate_visual'):
            jobs, _ = prepare_jobs(dict(action=action, class_name='Dewa', model_id='model_01', **settings))
            job = jobs[0]
            self.assertEqual({key: job[key] for key in settings}, settings)
            command = build_command(job)
            if action == 'colmap':
                self.assertEqual(command[command.index('--colmap_max_image_size') + 1], '2000')
            else:
                self.assertEqual(command[command.index('--image_downscale') + 1], '4')
                self.assertEqual(command[command.index('--image_max_size') + 1], '1280')
        for values in ({'image_downscale': 3}, {'image_downscale': True}, {'image_max_size': -1},
                       {'colmap_max_image_size': 0}, {'image_max_size': 1024.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                resolution_settings(values)

    def test_changed_resolution_or_colmap_model_invalidates_cached_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, output = Path(temporary) / 'scene', Path(temporary) / 'output'
            write(scene / 'images/a.jpg')
            for name in ('cameras', 'images', 'points3D'):
                write(scene / 'sparse/0' / (name + '.bin'))
            signature = colmap_signature(scene)
            write(scene / 'colmap_settings.json', json.dumps(dict(max_image_size=2000, model_signature=signature)))
            self.assertTrue(colmap_matches(scene, 2000))
            self.assertFalse(colmap_matches(scene, 3200))
            write(output / 'training_settings.json', json.dumps(dict(image_downscale=2, image_max_size=1280, colmap_signature=signature)))
            self.assertTrue(training_matches(output, scene, 2, 1280))
            self.assertFalse(training_matches(output, scene, 4, 1280))
            write(scene / 'sparse/0/cameras.bin', 'new SfM model')
            self.assertFalse(training_matches(output, scene, 2, 1280))
            self.assertFalse(colmap_matches(scene, 2000))
            write(scene / 'colmap_settings.json', '{invalid')
            self.assertFalse(colmap_matches(scene, 3200))
            write(output / 'evaluation/metrics.json', json.dumps(dict(image_downscale=2, image_max_size=1280)))
            self.assertTrue(evaluation_matches(output, 2, 1280))
            self.assertFalse(evaluation_matches(output, 2, 1600))


class ColmapTests(unittest.TestCase):
    def test_capability_check_and_explicit_cuda_failure_without_silent_fallback(self):
        for header, cuda in [('COLMAP 3.7\nCommit abc without CUDA', False),
                             ('COLMAP 3.8\nCommit abc with CUDA', True)]:
            with patch('pipeline_tools.colmap_capabilities.subprocess.run',
                       return_value=SimpleNamespace(stdout=header)):
                self.assertEqual(inspect_colmap()['cuda'], cuda)
                self.assertEqual(resolve_device('auto'), cuda)
                if cuda:
                    self.assertTrue(resolve_device('cuda'))
                else:
                    with self.assertRaisesRegex(RuntimeError, 'build CUDA'):
                        resolve_device('cuda')
        with patch('pipeline_tools.colmap_capabilities.subprocess.run', side_effect=FileNotFoundError('colmap')):
            self.assertFalse(inspect_colmap()['available'])
            with self.assertRaisesRegex(RuntimeError, 'tidak dapat diperiksa'):
                resolve_device('auto')

    def test_gpu_matching_options_reach_single_and_full_pipeline_jobs(self):
        settings = dict(colmap_device='cuda', colmap_matcher='sequential',
                        colmap_max_features=4096, colmap_sequential_overlap=15)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'datasets/Dewa/model_01/input/a.jpg')
            for action in ('colmap', 'full_pipeline'):
                jobs, _ = prepare_jobs(dict(action=action, class_name='Dewa', model_id='model_01', **settings), root)
                command = build_command(jobs[0], root)
                for key, value in settings.items():
                    self.assertEqual(jobs[0][key], value)
                    self.assertEqual(command[command.index('--' + key) + 1], str(value))
        for invalid in ({'colmap_device': 'gpu'}, {'colmap_matcher': 'bad'},
                        {'colmap_max_features': True}, {'colmap_max_features': 32769},
                        {'colmap_sequential_overlap': 1}, {'colmap_sequential_overlap': 2.5}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare_jobs(dict(action='colmap', class_name='Dewa', model_id='model_01', **invalid))

    def test_output_is_streamed_before_child_finishes(self):
        module = load_colmap()
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary) / 'release'
            messages = []
            errors = []
            class Logger:
                def log(self, message, **kwargs): messages.append(message)
            code = (f'import pathlib,time; p=pathlib.Path({str(release)!r}); '
                    'print("Matching block 1/10", flush=True); '
                    'exec("while not p.exists():\\n time.sleep(0.02)"); print("done", flush=True)')
            def run():
                try: module.run_command([sys.executable, '-u', '-c', code], Logger())
                except Exception as exc: errors.append(exc)
            thread = threading.Thread(target=run)
            thread.start()
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and not any(line.startswith('[COLMAP] Matching') for line in messages):
                    time.sleep(.02)
                self.assertTrue(any(line.startswith('[COLMAP] Matching') for line in messages))
                self.assertTrue(thread.is_alive(), 'Progress must reach the dashboard before the process exits')
            finally:
                write(release)
                thread.join(timeout=8)
            self.assertFalse(errors)

    def test_nonzero_command_keeps_error_details(self):
        module = load_colmap()
        with self.assertRaisesRegex(RuntimeError, 'feature matching failed'):
            module.run_command([sys.executable, '-c', 'import sys; print("feature matching failed"); sys.exit(2)'])

    def test_extraction_limit_changes_database_and_does_not_touch_original_photos(self):
        module = load_colmap()
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            write(scene / 'input/a.jpg', 'original image')
            commands = []
            def fake_command(command, logger=None):
                commands.append(command)
                if command[1] == 'mapper':
                    folder = Path(command[command.index('--output_path') + 1]) / '0'
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / 'images.bin').write_bytes((2).to_bytes(8, 'little'))
                if command[1] == 'image_undistorter':
                    folder = Path(command[command.index('--output_path') + 1])
                    write(folder / 'images/a.jpg', 'undistorted image')
                    for name in ('cameras', 'images', 'points3D'):
                        write(folder / 'sparse' / (name + '.bin'))
            with patch.object(module, 'run_command', side_effect=fake_command):
                module.run_colmap_pipeline(scene, max_image_size=2000)
            extractor = commands[0]
            self.assertEqual(extractor[extractor.index('--SiftExtraction.max_image_size') + 1], '2000')
            self.assertIn('maxsize_2000', extractor[extractor.index('--database_path') + 1])
            self.assertTrue(colmap_matches(scene, 2000))
            self.assertEqual((scene / 'input/a.jpg').read_text(), 'original image')
            self.assertEqual((scene / 'images/a.jpg').read_text(), 'undistorted image')

    def test_cuda_sequential_uses_logical_gpu_and_separate_database_and_cache(self):
        module = load_colmap()
        settings = dict(colmap_device='cuda', colmap_matcher='sequential',
                        colmap_max_features=4096, colmap_sequential_overlap=15)
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            write(scene / 'input/a.jpg')
            commands = []
            def fake_command(command, logger=None):
                commands.append(command)
                if command[1] == 'mapper':
                    folder = Path(command[command.index('--output_path') + 1]) / '0'
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / 'images.bin').write_bytes((2).to_bytes(8, 'little'))
                elif command[1] == 'image_undistorter':
                    folder = Path(command[command.index('--output_path') + 1])
                    write(folder / 'images/a.jpg')
                    for name in ('cameras', 'images', 'points3D'):
                        write(folder / 'sparse' / (name + '.bin'))
            with patch.object(module, 'run_command', side_effect=fake_command), \
                 patch.object(module, 'resolve_device', return_value=True):
                module.run_colmap_pipeline(scene, **settings)
            extraction, matching = commands[:2]
            self.assertEqual(extraction[extraction.index('--SiftExtraction.gpu_index') + 1], '0')
            self.assertEqual(extraction[extraction.index('--SiftExtraction.max_num_features') + 1], '4096')
            self.assertEqual(matching[1], 'sequential_matcher')
            self.assertEqual(matching[matching.index('--SiftMatching.gpu_index') + 1], '0')
            self.assertEqual(matching[matching.index('--SequentialMatching.overlap') + 1], '15')
            self.assertNotEqual(Path(extraction[extraction.index('--database_path') + 1]).parent.name, 'maxsize_3200')
            self.assertTrue(colmap_matches(scene, 3200, settings))
            self.assertFalse(colmap_matches(scene, 3200, dict(settings, colmap_device='cpu')))
            self.assertFalse(colmap_matches(scene, 3200, dict(settings, colmap_max_features=8192)))


if __name__ == '__main__':
    unittest.main()
