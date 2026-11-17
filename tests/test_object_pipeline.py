import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline_tools.object_region import save_region, read_region, validate_region, object_arguments
from pipeline_tools.object_mesh import clean_mesh, export_object, export_optional, object_status
from pipeline_tools.dataset_registry import has_sugar_result, file_signature
from pipeline_tools.batch_run_all import process_single_statue
from pipeline_tools.geometry_config import geometry_model_path
from pipeline_tools.logger_utils import ThesisLogger
from web_dashboard.server import prepare_jobs, build_command


REGION = {'min': [-1, -2, -3], 'max': [1, 2, 3], 'min_triangles': 50}


def write(path, content='data'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


class ObjectPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root / 'datasets/Dewa/model_01'
        write(self.scene / 'images/a.jpg')
        for name in ('cameras', 'images', 'points3D'):
            write(self.scene / 'sparse/0' / (name + '.bin'))

    def test_region_validation_and_save_requires_existing_colmap(self):
        for value in ({}, {**REGION, 'min': [True, 0, 0]}, {**REGION, 'max': [-2, 2, 3]},
                      {**REGION, 'min': [float('nan'), 0, 0]}, {**REGION, 'min_triangles': -1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_region(value)
        region = save_region(self.scene, REGION)
        self.assertIn('colmap_signature', region)
        self.assertEqual(read_region(self.scene), region)
        with self.assertRaisesRegex(ValueError, 'COLMAP'):
            save_region(self.root / 'empty_scene', REGION)
        save_region(self.scene, None)
        self.assertIsNone(read_region(self.scene))

    def test_queue_snapshots_bounds_and_routes_all_new_flags(self):
        saved = save_region(self.scene, REGION)
        jobs, _ = prepare_jobs(dict(action='full_pipeline', class_name='Dewa', model_id='model_01',
            sugar_mesh_export='surface', sugar_diagnostics=True), self.root)
        save_region(self.scene, {**REGION, 'max': [2, 3, 4]})
        saved = {'mode': 'camera_focus_v1'}
        self.assertEqual(jobs[0]['object_region'], saved)
        command = build_command(jobs[0], self.root)
        self.assertEqual(json.loads(command[command.index('--object_region_json') + 1]), saved)
        self.assertEqual(command[command.index('--sugar_mesh_export') + 1], 'surface')
        self.assertIn('--sugar_diagnostics', command)
        stages = []
        with patch('pipeline_tools.batch_run_all.run_stage', side_effect=lambda script, args, root: stages.append((script, args))):
            process_single_statue(self.scene, 'Dewa', 'model_01', root=self.root, skip_colmap=True,
                include_external=True, object_region=saved, sugar_mesh_export='surface', sugar_diagnostics=True)
        sugar = next(args for script, args in stages if script == '04_train_sugar.py')
        self.assertEqual(sugar[sugar.index('--mesh_export') + 1], 'surface')
        self.assertEqual(sum(script == 'object_mesh.py' for script, _ in stages), 1)
        self.assertEqual(sum('--sugar_stage' in args for _, args in stages), 1)
        post = [args for script, args in stages if script == 'run_external_method.py' and '--postprocess_only' in args]
        self.assertEqual(len(post), 2)
        for args in post:
            self.assertNotIn('--object_region_json', args)

    def test_surface_only_result_is_ready_for_rendering_and_geometry(self):
        output = self.root / 'outputs/sugar/Dewa/model_01'
        for name in ('mesh/coarse.ply', 'refined/test.pt', 'surface/refined_surface.ply'):
            write(output / name)
        manifest = dict(coarse_mesh='mesh/coarse.ply', refined_checkpoint='refined/test.pt',
                        textured_mesh=None, surface_mesh='surface/refined_surface.ply')
        write(output / 'pipeline_result.json', json.dumps(manifest))
        self.assertTrue(has_sugar_result(output))
        self.assertEqual(geometry_model_path(self.root, 'Dewa', 'model_01', 'SuGaR'), output / manifest['surface_mesh'])
        (output / manifest['surface_mesh']).unlink()
        self.assertFalse(has_sugar_result(output))

    def test_crop_uses_bounds_and_keeps_disconnected_substantial_parts(self):
        native, crop = MagicMock(), MagicMock()
        crop.has_triangles.return_value = True
        crop.cluster_connected_triangles.return_value = ([0, 1, 2], [500, 100, 2], [])
        native.crop.return_value = crop
        o3d = MagicMock()
        self.assertIs(clean_mesh(native, REGION, o3d), crop)
        o3d.geometry.AxisAlignedBoundingBox.assert_called_once_with(REGION['min'], REGION['max'])
        self.assertEqual(crop.remove_triangles_by_mask.call_args.args[0].tolist(), [False, False, True])
        native.remove_triangles_by_mask.assert_not_called()
        crop.has_triangles.return_value = False
        with self.assertRaisesRegex(ValueError, 'segitiga'):
            clean_mesh(native, REGION, o3d)

    def test_export_preserves_source_caches_and_invalidates_when_bounds_change(self):
        output = self.root / 'outputs/radegs/Dewa/model_01'
        source = write(output / 'recon_post.ply', 'original mesh')
        o3d, mesh = MagicMock(), MagicMock()
        mesh.vertices, mesh.triangles = [1, 2, 3], [[0, 1, 2]]
        o3d.io.read_triangle_mesh.return_value = mesh
        def save(path, result):
            write(Path(path), 'object mesh')
            return True
        o3d.io.write_triangle_mesh.side_effect = save
        with patch.dict(sys.modules, {'open3d': o3d}), patch('pipeline_tools.object_mesh.clean_mesh', return_value=mesh) as cleaner:
            target = export_object(self.root, 'Dewa', 'model_01', 'radegs', 'baseline', REGION)
            self.assertEqual(source.read_text(), 'original mesh')
            self.assertEqual(target.read_text(), 'object mesh')
            export_object(self.root, 'Dewa', 'model_01', 'radegs', 'baseline', REGION)
            self.assertEqual(cleaner.call_count, 1)
            self.assertEqual(object_status(output, REGION), 'ready')
            self.assertEqual(object_status(output, {**REGION, 'max': [2, 3, 4]}), 'stale')
            source.write_text('new original mesh')
            export_object(self.root, 'Dewa', 'model_01', 'radegs', 'baseline', REGION)
            self.assertEqual(cleaner.call_count, 2)

    def test_changed_colmap_warns_without_overwriting_original_or_old_object(self):
        saved = save_region(self.scene, REGION)
        output = self.root / 'outputs/radegs/Dewa/model_01'
        source = write(output / 'recon_post.ply', 'original')
        target = write(output / 'object_mesh/object.ply', 'previous object')
        write(self.scene / 'sparse/0/images.bin', 'different camera reconstruction')
        self.assertIsNone(export_optional(self.root, 'Dewa', 'model_01', 'radegs', 'baseline', saved))
        self.assertEqual(source.read_text(), 'original')
        self.assertEqual(target.read_text(), 'previous object')
        self.assertEqual(object_status(output, saved), 'error')
        self.assertIn('COLMAP berubah', (output / 'object_mesh/error.json').read_text())

    def test_diagnostic_logger_does_not_create_or_change_summary(self):
        logger = ThesisLogger('Dewa', 'model_01', 'SuGaR_diagnostic_coarse', base_dir=self.root)
        def close_handlers():
            for handler in list(logger.logger.handlers):
                handler.close()
                logger.logger.removeHandler(handler)
        self.addCleanup(close_handlers)
        logger.set_metrics(psnr=20, ssim=0.8, lpips=0.3)
        logger.finalize(True, export_summary=False)
        self.assertTrue(logger.json_file.is_file())
        self.assertFalse((self.root / 'outputs/logs/evaluation_reports/summary_comparison.csv').exists())

    def test_surface_export_uses_refined_vertices_and_faces_without_uv_baking(self):
        from pipeline_tools.sugar_surface import export_refined_surface
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2]])
        def tensor(value):
            mock = MagicMock()
            mock.detach.return_value.cpu.return_value.numpy.return_value = value
            return mock
        torch, o3d = MagicMock(), MagicMock()
        torch.load.return_value = {'state_dict': {'_points': tensor(vertices), '_surface_mesh_faces': tensor(faces)}}
        o3d.utility.Vector3dVector.side_effect = lambda value: value
        o3d.utility.Vector3iVector.side_effect = lambda value: value
        o3d.io.write_triangle_mesh.side_effect = lambda path, mesh: bool(write(Path(path), 'surface'))
        target = self.root / 'surface/refined.ply'
        with patch.dict(sys.modules, {'torch': torch, 'open3d': o3d}):
            export_refined_surface(self.root / 'refined.pt', target)
        self.assertTrue(target.is_file())
        np.testing.assert_array_equal(o3d.geometry.TriangleMesh.call_args.args[0], vertices)
        np.testing.assert_array_equal(o3d.geometry.TriangleMesh.call_args.args[1], faces)


if __name__ == '__main__':
    unittest.main()
