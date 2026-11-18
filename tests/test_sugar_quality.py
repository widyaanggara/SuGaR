import copy
import json
import os
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import Mock, patch

from pipeline_tools import optimize_sugar as quality
from pipeline_tools.dataset_registry import file_signature, has_sugar_result
from test_sugar_focus import cameras_at


def write(path, content='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


class SugarQualityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.source, self.output = root / 'source', root / 'quality'
        self.scene, self.baseline = root / 'scene', root / '3dgs'
        self.coarse = write(self.source / 'coarse/15000.pt')
        baseline_ply = write(self.baseline / 'point_cloud/iteration_7000/point_cloud.ply')
        write(self.baseline / 'cameras.json', json.dumps(cameras_at([8, 3, -7])))
        self.manifest = dict(scene_path=str(self.scene), coarse_checkpoint='coarse/15000.pt',
                             regularization='dn_consistency', decimation_target=1000000,
                             refinement_iterations=15000, white_background=False,
                             image_downscale=1, image_max_size=1600,
                             baseline_signature=file_signature(baseline_ply))
        write(self.source / 'pipeline_result.json', json.dumps(self.manifest))
        self.addCleanup(patch.stopall)
        patch.object(quality, 'training_matches', return_value=True).start()
        patch.object(quality, 'scene_identity', return_value={'fixture': 1}).start()

    def prepare(self, destination=None):
        return quality.prepare(self.source, destination or self.output, self.scene, self.baseline)

    def test_plan_preserves_protocol_and_binds_coarse_and_focused_region(self):
        plan = self.prepare()
        self.assertEqual(plan['settings']['refinement_iterations'], 15000)
        self.assertEqual(plan['gaussians_per_triangle'], 3)
        self.assertEqual(plan['coarse_signature'], file_signature(self.coarse))
        self.assertEqual((plan['statue_focus'], plan['statue_poisson_depth'], plan['poisson_depth'],
                          plan['background_triangle_target']), (True, 11, 11, 2000000))
        self.assertAlmostEqual(sum(v * v for v in plan['up']), 1.0)
        self.assertLess(plan['region']['min'][0], 8)
        self.assertGreater(plan['region']['max'][0], 8)
        self.assertFalse(self.output.exists())

    def test_short_source_refinement_is_extended_not_inherited(self):
        self.manifest['refinement_iterations'] = 2000
        write(self.source / 'pipeline_result.json', json.dumps(self.manifest))
        plan = self.prepare()
        self.assertEqual(plan['settings']['refinement_iterations'], 15000)
        self.assertEqual(plan['source_refinement_iterations'], 2000)
        explicit = quality.prepare(self.source, self.output, self.scene, self.baseline,
                                   refinement_iterations=7000)
        self.assertEqual(explicit['settings']['refinement_iterations'], 7000)
        with self.assertRaisesRegex(ValueError, 'refinement'):
            quality.prepare(self.source, self.output, self.scene, self.baseline,
                            refinement_iterations=3000)

    def test_dashboard_job_reaches_cli_with_separate_output_per_configuration(self):
        from web_dashboard import server
        root = self.source.parent
        values = dict(gaussians_per_triangle=6, refinement_iterations=7000, foreground_triangles=1000000,
                      background_triangles=0, poisson_depth=12, statue_poisson_depth=10)
        request = dict(action='sugar_quality', class_name='Dewi', model_id='model_01',
                       sugar_quality=dict(values, statue_focus=False))
        with patch.object(server, 'has_checkpoint', return_value=True), \
                patch.object(server, 'has_sugar_result', return_value=True):
            jobs, skipped = server.prepare_jobs(request, root)
            default_jobs, _ = server.prepare_jobs(dict(request, sugar_quality=None), root)
        self.assertEqual((len(jobs), skipped), (1, []))
        command = server.build_command(jobs[0], root)
        self.assertTrue(command[2].endswith('optimize_sugar.py'))
        for key, value in values.items():
            self.assertEqual(command[command.index('--' + key) + 1], str(value))
        self.assertIn('--no-statue_focus', command)
        self.assertEqual(Path(command[command.index('--output_dir') + 1]),
                         root / 'outputs/quality/sugar/Dewi/model_01/g6_it7000_fg1000000_bg0_d12_snone')
        default = server.build_command(default_jobs[0], root)
        self.assertIn('--statue_focus', default)
        self.assertNotIn('--background_mode', default)
        self.assertTrue(default[default.index('--output_dir') + 1].endswith('g3_it15000_fg0_bg2000000_d11_s11'))
        # The real CLI parser must accept exactly what the dashboard builds.
        with patch.object(sys, 'argv', default[2:] + ['--plan_only']), \
                patch.object(quality, 'prepare', side_effect=SystemExit(0)) as prepare, \
                patch.object(quality, 'gpu_lock'), patch.object(quality, 'scene_pairs'):
            with self.assertRaises(SystemExit):
                quality.main()
        self.assertEqual(prepare.call_args.args[4:], (3, 11, 0, 2000000, 15000, True, 11))
        with patch.object(server, 'has_checkpoint', return_value=True), \
                patch.object(server, 'has_sugar_result', return_value=False):
            jobs, skipped = server.prepare_jobs(request, root)
        self.assertEqual(jobs, [])
        self.assertIn('SuGaR baseline', skipped[0])
        for invalid in ({'gaussians_per_triangle': 2}, {'refinement_iterations': 3000},
                        {'foreground_triangles': -1}, {'poisson_depth': 13}, {'background_triangles': True},
                        {'unknown': 1}, 'bad', {'background_mode': 'free'}, {'statue_focus': 1},
                        {'statue_poisson_depth': 13}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                server.prepare_jobs(dict(request, sugar_quality=invalid), root)

    def test_protected_outputs_and_changed_inputs_are_rejected(self):
        for destination in (self.source, self.source / 'child', self.scene, self.baseline):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.prepare(destination)
        plan = self.prepare()
        write(self.output / 'optimization_plan.json', json.dumps(plan))
        self.assertEqual(self.prepare(), plan)
        self.coarse.write_text('changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'berubah'):
            self.prepare()

    def test_stale_baseline_and_escaping_checkpoint_are_rejected(self):
        self.manifest['coarse_checkpoint'] = '../outside.pt'
        write(self.source.parent / 'outside.pt')
        write(self.source / 'pipeline_result.json', json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'coarse'):
            self.prepare()
        self.manifest['coarse_checkpoint'] = 'coarse/15000.pt'
        self.manifest['baseline_signature'] = {}
        write(self.source / 'pipeline_result.json', json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'baseline'):
            self.prepare()

    def test_worker_uses_existing_coarse_full_scene_and_isolated_summary(self):
        plan = self.prepare()
        trainer, logger = Mock(), Mock()
        args = types.SimpleNamespace(statue_class='Dewi', model_id='model_01', gpu=0)
        with patch.object(quality, 'module', return_value=trainer), \
                patch.object(quality, 'ThesisLogger', return_value=logger), \
                patch('pipeline_tools.object_mesh.export_object') as export:
            quality.train_worker(args, plan, self.scene, self.baseline, self.output)
        kwargs = trainer.train_sugar_pipeline.call_args.kwargs
        self.assertEqual(kwargs['reuse_coarse'], str(self.coarse.resolve()))
        self.assertEqual(kwargs['gaussians_per_triangle'], 3)
        self.assertEqual(kwargs['quality_options']['region'], plan['region'])
        self.assertEqual(export.call_args.args[4], 'baseline')
        self.assertEqual(export.call_args.kwargs['output_override'], self.output)
        logger.finalize.assert_called_once_with(True, export_summary=False)

    def test_nonstandard_output_validates_against_explicit_baseline(self):
        for key in ('coarse_mesh', 'refined_checkpoint', 'surface_mesh'):
            self.manifest[key] = key + '.ply'
            write(self.output / self.manifest[key])
        write(self.output / 'pipeline_result.json', json.dumps(self.manifest))
        self.assertTrue(has_sugar_result(self.output, baseline=self.baseline))

    def test_pipeline_reuses_coarse_and_passes_world_bounds_to_extraction(self):
        pipeline = quality.module('04_train_sugar')
        mesh_path = write(self.output / 'mesh/new.ply')
        refined_path = write(self.output / 'refined/new.pt')
        surface_path = self.output / 'surface/refined_surface.ply'
        mesh, refine = Mock(return_value=[str(mesh_path)]), Mock(return_value=str(refined_path))
        coarse = Mock(side_effect=AssertionError('Coarse training must not run'))
        functions = {
            'sugar_trainers.coarse_density': ('coarse_training_with_density_regularization', coarse),
            'sugar_trainers.coarse_sdf': ('coarse_training_with_sdf_regularization', coarse),
            'sugar_trainers.coarse_density_and_dn_consistency': ('coarse_training_with_density_regularization_and_dn_consistency', coarse),
            'sugar_extractors.coarse_mesh': ('extract_mesh_from_coarse_sugar', mesh),
            'sugar_trainers.refine': ('refined_training', refine),
            'sugar_extractors.refined_mesh': ('extract_mesh_and_texture_from_refined_sugar', Mock()),
        }
        modules = {}
        for package in ('sugar_trainers', 'sugar_extractors'):
            modules[package] = types.ModuleType(package)
            modules[package].__path__ = []
        for name, (function, implementation) in functions.items():
            modules[name] = types.ModuleType(name)
            setattr(modules[name], function, implementation)
        modules['torch'] = types.SimpleNamespace(cuda=Mock(is_available=lambda: True))
        modules['open3d'] = types.SimpleNamespace(io=types.SimpleNamespace(
            read_triangle_mesh=lambda path: types.SimpleNamespace(vertices=[0]*8, triangles=[0]*12)))
        options = dict(region={'min': [6., 1., -9.], 'max': [10., 5., -5.]}, poisson_depth=10,
                       foreground_triangle_target=0, background_triangle_target=1000000,
                       project_mesh_on_surface_points=False,
                       statue_focus=True, statue_poisson_depth=11, up=[0., -1., 0.])
        source_bytes = self.coarse.read_bytes()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}), \
                patch.object(pipeline, 'has_colmap_model', return_value=True), \
                patch.object(pipeline, 'has_checkpoint', return_value=True), \
                patch.object(pipeline, 'training_matches', return_value=True), \
                patch('pipeline_tools.sugar_surface.export_refined_surface',
                      side_effect=lambda checkpoint, path: str(write(path))):
            for _ in range(2):
                pipeline.train_sugar_pipeline(str(self.scene), str(self.baseline), str(self.output),
                    refinement_time='long', n_vertices=1000000, gaussians_per_triangle=3,
                    export_obj=False, reuse_coarse=str(self.coarse), quality_options=options)
        coarse.assert_not_called()
        self.assertEqual(mesh.call_count, 1)
        self.assertEqual(refine.call_count, 1)
        extraction = mesh.call_args.args[0]
        self.assertFalse(extraction.center_bbox)
        self.assertEqual(extraction.bboxmin, '(6.0, 1.0, -9.0)')
        self.assertTrue(extraction.save_predecimation)
        self.assertEqual(extraction.foreground_triangle_target, 0)
        self.assertEqual(extraction.background_triangle_target, 1000000)
        self.assertFalse(extraction.project_mesh_on_surface_points)
        self.assertTrue(extraction.statue_focus)
        self.assertFalse(hasattr(extraction, 'skip_background'))
        self.assertEqual((extraction.statue_up, extraction.statue_poisson_depth), ([0., -1., 0.], 11))
        self.assertFalse(hasattr(refine.call_args.args[0], 'free_background'))
        counts = json.loads((self.output / 'refinement_geometry.json').read_text())
        self.assertEqual(counts, dict(vertices=8, triangles=12, gaussians=36))
        self.assertEqual(refine.call_args.args[0].gaussians_per_triangle, 3)
        self.assertEqual(self.coarse.read_bytes(), source_bytes)
        manifest = json.loads((self.output / 'pipeline_result.json').read_text())
        self.assertEqual(manifest['coarse_checkpoint'], 'coarse/reused/15000.pt')
        self.assertEqual(manifest['surface_mesh'], surface_path.relative_to(self.output).as_posix())

    def test_comparison_matches_camera_names_and_rejects_protocol_mismatch(self):
        before = dict(visual_protocol={'version': 'test'}, white_background=False,
                      image_downscale=1, image_max_size=1600, eval_split_interval=8,
                      psnr=24., ssim=.83, lpips=.31,
                      per_view=[dict(image='a', psnr=22., ssim=.8, lpips=.32),
                                dict(image='b', psnr=26., ssim=.86, lpips=.30)])
        after = copy.deepcopy(before)
        after['per_view'].reverse()
        after['psnr'] = 25.
        after['per_view'][1]['psnr'] = 24.
        report = quality.compare_reports(before, after)
        self.assertEqual(report['delta']['psnr'], 1.)
        self.assertEqual(report['per_view'][-1]['image'], 'a')
        self.assertEqual(report['per_view'][-1]['psnr_delta'], 2.)
        after['image_max_size'] = 800
        with self.assertRaisesRegex(ValueError, 'Protokol'):
            quality.compare_reports(before, after)


if __name__ == '__main__':
    unittest.main()
