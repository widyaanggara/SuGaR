"""Verify dashboard parameters reach upstream training calls, without a GPU."""
import ast
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline_tools.run_settings import research_presets, sugar_settings, training_matches, colmap_signature
from web_dashboard.server import prepare_jobs, build_command

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('settings_contract_' + name, ROOT / 'pipeline_tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, content='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


class SettingsContractTests(unittest.TestCase):
    def test_fresh_html_controls_match_the_declared_default_official_preset(self):
        class Defaults(HTMLParser):
            def __init__(self):
                super().__init__()
                self.values, self.current = {}, None
            def handle_starttag(self, tag, items):
                attrs = dict(items)
                if tag == 'input' and 'id' in attrs:
                    self.values[attrs['id']] = attrs.get('value')
                if tag == 'select':
                    self.current = attrs.get('id')
                if tag == 'option' and self.current:
                    self.values.setdefault(self.current, attrs.get('value'))
                    if 'selected' in attrs:
                        self.values[self.current] = attrs.get('value')
            def handle_endtag(self, tag):
                if tag == 'select': self.current = None
        parser = Defaults()
        parser.feed((ROOT / 'web_dashboard/index.html').read_text(encoding='utf-8'))
        reference = research_presets()
        self.assertEqual(parser.values['settingsPreset'], reference['default'])
        for field, control in dict(vertices='meshTriangles', gaussians_per_triangle='gaussiansPerTriangle',
                                   refinement_time='refinementTime', regularization='regularization').items():
            self.assertEqual(parser.values[control], str(reference['presets'][reference['default']]['values'][field]))

    def test_official_presets_agree_with_upstream_train_py_config_branches(self):
        tree = ast.parse((ROOT / 'train.py').read_text(encoding='utf-8'))
        for key, branch in (('official_high', 'high_poly'), ('official_low', 'low_poly')):
            values = research_presets()['presets'][key]['values']
            args = types.SimpleNamespace(n_vertices_in_mesh=None, gaussians_per_triangle=None)
            node = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Attribute) and node.test.attr == branch)
            exec(compile(ast.Module(body=node.body, type_ignores=[]), 'upstream_config', 'exec'), {'args': args})
            self.assertEqual(values['vertices'], args.n_vertices_in_mesh)
            self.assertEqual(values['gaussians_per_triangle'], args.gaussians_per_triangle)

    def test_api_and_full_batch_forward_all_sugar_controls_and_validate_bad_types(self):
        from pipeline_tools import batch_run_all as batch
        values = dict(vertices=200000, gaussians_per_triangle=6, refinement_time='medium', regularization='density')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / 'datasets/Dewa/model_01'
            write(scene / 'input/a.jpg')
            for action in ('sugar', 'full_pipeline'):
                jobs, _ = prepare_jobs(dict(action=action, class_name='Dewa', model_id='model_01', **values), root)
                command = build_command(jobs[0], root)
                for key, value in values.items():
                    self.assertEqual(command[command.index('--' + key) + 1], str(value))
            class StopAtTraining(Exception): pass
            def stage(script, args, root):
                if script == '04_train_sugar.py':
                    for key, value in values.items():
                        self.assertEqual(str(args[args.index('--' + key) + 1]), str(value))
                    raise StopAtTraining
            with patch.object(batch, 'run_stage', side_effect=stage), patch.object(batch, 'has_colmap_model', return_value=True):
                with self.assertRaises(StopAtTraining):
                    batch.process_single_statue(scene, 'Dewa', 'model_01', root=root, **values)
        for invalid in ({'gaussians_per_triangle': 2}, {'vertices': 200000.5}, {'vertices': True},
                        {'gaussians_per_triangle': False}, {'regularization': 'bad'}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sugar_settings(invalid)

    def test_baseline_resolution_reaches_real_training_command_and_receipt(self):
        module = load('03_train_3dgs')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            process = types.SimpleNamespace(stdout=[], returncode=0, pid=1, wait=lambda: 0)
            with patch.object(module, 'has_colmap_model', return_value=True), \
                 patch.object(module.subprocess, 'Popen', return_value=process) as popen:
                module.train_vanilla_3dgs(temporary, output, gpu=2, image_downscale=4, image_max_size=1280)
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index('--image_downscale') + 1], '4')
            self.assertEqual(command[command.index('--image_max_size') + 1], '1280')
            self.assertIn('--eval', command)
            self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            receipt = json.loads((output / 'training_settings.json').read_text())
            self.assertEqual(receipt['image_max_size'], 1280)
            self.assertTrue(receipt['eval_split'])

    def test_sugar_controls_reach_coarse_mesh_refinement_export_and_manifest(self):
        module = load('04_train_sugar')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'sugar'
            artifacts = [output / name for name in ('coarse/test.pt', 'mesh/test.ply', 'refined/test.pt', 'textured_mesh/test.obj')]
            for path in artifacts: write(path)
            coarse, mesh, refined, textured = [Mock(return_value=value) for value in
                (str(artifacts[0]), [str(artifacts[1])], str(artifacts[2]), str(artifacts[3]))]
            functions = {
                'sugar_trainers.coarse_density': ('coarse_training_with_density_regularization', coarse),
                'sugar_trainers.coarse_sdf': ('coarse_training_with_sdf_regularization', Mock()),
                'sugar_trainers.coarse_density_and_dn_consistency': ('coarse_training_with_density_regularization_and_dn_consistency', Mock()),
                'sugar_extractors.coarse_mesh': ('extract_mesh_from_coarse_sugar', mesh),
                'sugar_trainers.refine': ('refined_training', refined),
                'sugar_extractors.refined_mesh': ('extract_mesh_and_texture_from_refined_sugar', textured)}
            fake_modules = {}
            for package in ('sugar_trainers', 'sugar_extractors'):
                fake_modules[package] = types.ModuleType(package)
                fake_modules[package].__path__ = []
            for name, (function, value) in functions.items():
                fake_modules[name] = types.ModuleType(name)
                setattr(fake_modules[name], function, value)
            fake_modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(
                is_available=lambda: True, set_device=Mock(), reset_peak_memory_stats=Mock(), empty_cache=Mock()))
            baseline = Path(temporary) / 'baseline'
            write(baseline / 'point_cloud/iteration_7000/point_cloud.ply')
            with patch.dict(sys.modules, fake_modules), patch.dict(os.environ, {}), \
                 patch.object(module, 'has_colmap_model', return_value=True), \
                 patch.object(module, 'has_checkpoint', return_value=True), \
                 patch.object(module, 'training_matches', return_value=True):
                module.train_sugar_pipeline(temporary, str(baseline), str(output),
                    regularization_type='density', refinement_time='medium', n_vertices=200000,
                    gaussians_per_triangle=6, gpu=2, image_downscale=4, image_max_size=1280)
                self.assertEqual(os.environ['SUGAR_IMAGE_DOWNSCALE'], '4')
                self.assertEqual(os.environ['SUGAR_IMAGE_MAX_SIZE'], '1280')
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(mesh.call_args.args[0].decimation_target, 200000)
            self.assertEqual(refined.call_args.args[0].gaussians_per_triangle, 6)
            self.assertEqual(refined.call_args.args[0].refinement_iterations, 7000)
            self.assertEqual(textured.call_args.args[0].n_gaussians_per_surface_triangle, 6)
            self.assertTrue(coarse.call_args.args[0].eval)
            report = json.loads((output / 'pipeline_result.json').read_text())
            self.assertEqual(report['gaussians_per_triangle'], 6)
            self.assertEqual(report['regularization'], 'density')
            self.assertEqual(report['image_max_size'], 1280)
            self.assertEqual(report['decimation_target'], 200000)

    def test_cached_training_cannot_hide_a_different_background_or_non_holdout_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / 'scene'
            output = Path(temporary) / 'output'
            for name in ('cameras', 'images', 'points3D'): write(scene / 'sparse/0' / (name + '.bin'))
            write(output / 'training_settings.json', json.dumps(dict(image_downscale=1, image_max_size=1600,
                  colmap_signature=colmap_signature(scene))))
            write(output / 'cfg_args', 'Namespace(eval=False, white_background=False)')
            self.assertFalse(training_matches(output, scene, 1, 1600))
            write(output / 'cfg_args', 'Namespace(eval=True, white_background=True)')
            self.assertFalse(training_matches(output, scene, 1, 1600))
