"""Normal SuGaR training meshes only the statue, reusing an earlier coarse model (no GPU)."""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('statue_pipeline_04', ROOT / 'pipeline_tools/04_train_sugar.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, content='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


FOCUS = dict(region={'min': [-1., -1., -1.], 'max': [1., 1., 1.], 'min_triangles': 50}, foreground_box='upstream',
             poisson_depth=10, statue_poisson_depth=10, foreground_triangle_target=200000,
             background_triangle_target=200000, project_mesh_on_surface_points=False,
             statue_focus=True, up=[0., 0., 1.])


class StatuePipelineTests(unittest.TestCase):
    def test_statue_focus_reuses_coarse_and_meshes_only_the_statue(self):
        module = load()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'sugar'
            artifacts = [output / name for name in ('coarse/test.pt', 'mesh/test.ply', 'refined/test.pt', 'textured_mesh/test.obj')]
            for path in artifacts:
                write(path)
            coarse = Mock(return_value=str(artifacts[0]))
            mesh = Mock(return_value=[str(artifacts[1])])
            refined = Mock(return_value=str(artifacts[2]))
            textured = Mock(return_value=str(artifacts[3]))
            functions = {
                'sugar_trainers.coarse_density': ('coarse_training_with_density_regularization', Mock()),
                'sugar_trainers.coarse_sdf': ('coarse_training_with_sdf_regularization', Mock()),
                'sugar_trainers.coarse_density_and_dn_consistency': (
                    'coarse_training_with_density_regularization_and_dn_consistency', coarse),
                'sugar_extractors.coarse_mesh': ('extract_mesh_from_coarse_sugar', mesh),
                'sugar_trainers.refine': ('refined_training', refined),
                'sugar_extractors.refined_mesh': ('extract_mesh_and_texture_from_refined_sugar', textured)}
            fake = {}
            for package in ('sugar_trainers', 'sugar_extractors'):
                fake[package] = types.ModuleType(package)
                fake[package].__path__ = []
            for name, (function, value) in functions.items():
                fake[name] = types.ModuleType(name)
                setattr(fake[name], function, value)
            fake['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(
                is_available=lambda: True, set_device=Mock(), reset_peak_memory_stats=Mock(), empty_cache=Mock()))
            fake['open3d'] = types.SimpleNamespace(io=types.SimpleNamespace(
                read_triangle_mesh=lambda path: types.SimpleNamespace(vertices=[0] * 4, triangles=[0] * 2)))
            baseline = Path(temporary) / 'baseline'
            write(baseline / 'point_cloud/iteration_7000/point_cloud.ply')
            run = dict(regularization_type='dn_consistency', refinement_time='long', n_vertices=200000,
                       gaussians_per_triangle=3, image_downscale=1, image_max_size=1600)
            with patch.dict(sys.modules, fake), patch.dict(os.environ, {}), \
                 patch.object(module, 'has_colmap_model', return_value=True), \
                 patch.object(module, 'has_checkpoint', return_value=True), \
                 patch.object(module, 'training_matches', return_value=True), \
                 patch.object(module, 'statue_focus_options', return_value=FOCUS) as options:
                # Old behaviour: whole-scene mesh, later cropped.
                module.train_sugar_pipeline(temporary, str(baseline), str(output), **run)
                self.assertFalse(json.loads((output / 'pipeline_result.json').read_text())['statue_focus'])
                options.assert_not_called()
                module.train_sugar_pipeline(temporary, str(baseline), str(output), statue_focus=True, **run)
            self.assertEqual(coarse.call_count, 1, 'Statue focus must reuse the finished coarse model')
            self.assertEqual(mesh.call_count, 2)
            self.assertEqual(refined.call_count, 2)
            args = mesh.call_args.args[0]
            self.assertTrue(args.statue_focus)
            self.assertFalse(args.project_mesh_on_surface_points)
            # SuGaR's own foreground/background boxes stay; only the statue is split out of the foreground.
            self.assertTrue(args.center_bbox)
            self.assertIsNone(args.bboxmin)
            self.assertEqual(args.statue_box, ((-1., -1., -1.), (1., 1., 1.)))
            self.assertEqual(args.statue_up, [0., 0., 1.])
            self.assertEqual((args.foreground_triangle_target, args.background_triangle_target), (200000, 200000))
            self.assertEqual((args.poisson_depth, args.statue_poisson_depth), (10, 10))
            manifest = json.loads((output / 'pipeline_result.json').read_text())
            self.assertTrue(manifest['statue_focus'])
            self.assertTrue(manifest['coarse_checkpoint'].startswith('coarse/reused/'))

    def test_statue_focus_falls_back_to_scene_mesh_when_cameras_are_unusable(self):
        module = load()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises((OSError, ValueError)):
                module.statue_focus_options(temporary, 1000)


if __name__ == '__main__':
    unittest.main()
