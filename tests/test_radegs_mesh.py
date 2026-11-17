import unittest
import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, patch

from pipeline_tools.radegs_mesh import main, post_process_mesh


class MeshPostprocessTests(unittest.TestCase):
    def mesh(self, counts):
        mesh = MagicMock()
        mesh.vertices = [None] * 3
        mesh.triangles = [None] * sum(counts)
        labels = [index for index, count in enumerate(counts) for _ in range(count)]
        mesh.cluster_connected_triangles.return_value = (labels, counts, [])
        return mesh

    def test_fewer_than_eight_clusters_keeps_valid_components(self):
        mesh = self.mesh([120, 80, 10])
        self.assertIs(post_process_mesh(mesh, 8), mesh)
        mask = mesh.remove_triangles_by_mask.call_args.args[0].tolist()
        self.assertEqual(sum(mask), 10)
        self.assertEqual(mask[:200], [False] * 200)

    def test_native_threshold_preserved_with_many_clusters(self):
        mesh = self.mesh([100, 90, 80, 70, 60, 55, 52, 51, 50])
        post_process_mesh(mesh, 8)
        self.assertEqual(sum(mesh.remove_triangles_by_mask.call_args.args[0]), 50)

    def test_empty_or_only_tiny_components_fail_clearly(self):
        for counts in ([], [10, 20]):
            with self.subTest(counts=counts), self.assertRaises(RuntimeError):
                post_process_mesh(self.mesh(counts), 8)

    def test_launcher_forwards_native_flags_and_validates_both_meshes(self):
        native = MagicMock()
        native.ArgumentParser = argparse.ArgumentParser
        def model_params(parser, sentinel):
            parser.add_argument('-m', dest='model_path')
            parser.add_argument('-r', type=int)
            parser.add_argument('--depth_ratio', type=float)
            return MagicMock()
        native.ModelParams.side_effect = model_params
        native.get_combined_args.side_effect = lambda parser: parser.parse_args()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('recon.ply', 'recon_post.ply'):
                (root / name).write_text('mesh')
            argv = ['radegs_mesh.py', '--repo', str(root), '-m', str(root),
                    '--iteration', '30000', '-r', '2', '--depth_ratio', '0.5',
                    '--voxel_size', '0.009', '--max_depth', '8.0', '--num_cluster', '8']
            with patch('pipeline_tools.radegs_mesh.importlib.util.spec_from_file_location'), \
                 patch('pipeline_tools.radegs_mesh.importlib.util.module_from_spec', return_value=native), \
                 patch.object(sys, 'argv', argv), patch.object(sys, 'path', list(sys.path)):
                main()
        self.assertIs(native.post_process_mesh, post_process_mesh)
        native.render_sets.assert_called_once()
        self.assertEqual(native.render_sets.call_args.args[1], 30000)
        self.assertEqual(native.render_sets.call_args.args[3:], (8.0, 0.009, 8, False))
        self.assertEqual(native.o3d.io.read_triangle_mesh.call_count, 2)


if __name__ == '__main__':
    unittest.main()
