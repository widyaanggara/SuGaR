import tempfile
import unittest
from pathlib import Path

import numpy as np

from sugar_utils.statue_focus import camera_up, extract_statue, load_region, region_contains, save_region


def rotation_to(up):
    """Orthonormal basis whose third column is `up`."""
    up = up / np.linalg.norm(up)
    helper = np.array([1., 0., 0.]) if abs(up[0]) < 0.9 else np.array([0., 1., 0.])
    x = np.cross(helper, up)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up], axis=1)


class StatueFocusTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.up = np.array([0.2, -0.95, 0.1])
        self.up /= np.linalg.norm(self.up)
        basis = rotation_to(self.up)
        self.origin = np.array([3., -1., 2.])
        to_world = lambda local: local @ basis.T + self.origin
        ground = np.c_[rng.uniform(-3, 3, (60000, 2)), rng.normal(0, 0.003, 60000)]
        angle = rng.uniform(0, 2 * np.pi, 30000)
        statue = np.c_[0.3 * np.cos(angle), 0.3 * np.sin(angle), rng.uniform(0.05, 2.0, 30000)]
        neighbor = np.c_[rng.uniform(1.6, 2.2, (8000, 2)), rng.uniform(0.05, 0.8, 8000)]
        floater = np.c_[rng.normal(0, 0.01, (20, 2)), rng.normal(2.6, 0.01, 20)]
        self.parts = {name: to_world(value) for name, value in
                      dict(ground=ground, statue=statue, neighbor=neighbor, floater=floater).items()}
        self.points = np.concatenate(list(self.parts.values()))
        self.box_min, self.box_max = self.origin - 3.2, self.origin + 3.2
        self.center = to_world(np.array([0., 0., 1.]))

    def extract(self):
        return extract_statue(self.points, self.up, self.center, self.box_min, self.box_max)

    def test_keeps_statue_and_drops_ground_neighbor_and_floater(self):
        region = self.extract()
        self.assertTrue(region['summary']['ground_found'])
        self.assertGreater(region_contains(region, self.parts['statue']).mean(), 0.95)
        for name in ('ground', 'neighbor', 'floater'):
            with self.subTest(part=name):
                self.assertLess(region_contains(region, self.parts[name]).mean(), 0.02)

    def test_region_round_trips_and_rejects_points_outside_grid(self):
        region = self.extract()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'statue_region.npz'
            save_region(path, region)
            loaded = load_region(path)
        np.testing.assert_array_equal(region_contains(loaded, self.points), region_contains(region, self.points))
        self.assertEqual(loaded['summary'], region['summary'])
        self.assertFalse(region_contains(loaded, self.origin[None] + 50).any())

    def test_without_ground_the_closest_component_is_still_isolated(self):
        self.points = np.concatenate([self.parts['statue'], self.parts['neighbor']])
        region = self.extract()
        self.assertFalse(region['summary']['ground_found'])
        self.assertGreater(region_contains(region, self.parts['statue']).mean(), 0.95)
        self.assertLess(region_contains(region, self.parts['neighbor']).mean(), 0.02)

    def test_ground_is_found_when_up_is_given_with_the_wrong_sign(self):
        region = extract_statue(self.points, -self.up, self.center, self.box_min, self.box_max)
        self.assertTrue(region['summary']['ground_found'])
        self.assertGreater(region_contains(region, self.parts['statue']).mean(), 0.95)
        self.assertLess(region_contains(region, self.parts['ground']).mean(), 0.02)

    def test_camera_up_handles_portrait_photos_on_an_orbit(self):
        up = np.array([0., 0., 1.])
        rotations = []
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            look = np.array([-np.cos(angle), -np.sin(angle), -0.3])
            look /= np.linalg.norm(look)
            right = np.cross(look, up)
            right /= np.linalg.norm(right)
            down = np.cross(look, right)
            # Phone turned 90 degrees: image x points down the world, image y points sideways.
            rotations.append(np.stack([down, -right, look], axis=1))
        estimate = camera_up(np.stack(rotations))
        self.assertGreater(abs(estimate @ up), 0.95)

    def test_camera_up_uses_negative_camera_y(self):
        rotations = np.stack([rotation_to(np.array([0., 0., 1.]))[:, [0, 2, 1]] * [1, -1, 1]] * 3)
        np.testing.assert_allclose(camera_up(rotations), -rotations[0][:, 1])


class FakeMesh:
    """Just enough of open3d.geometry.TriangleMesh for statue_submesh."""
    def __init__(self, vertices, triangles):
        self.vertices, self.triangles = np.asarray(vertices), np.asarray(triangles)

    def remove_unreferenced_vertices(self):
        used = np.unique(self.triangles)
        remap = np.full(len(self.vertices), -1)
        remap[used] = np.arange(len(used))
        self.vertices, self.triangles = self.vertices[used], remap[self.triangles]


FAKE_O3D = type('o3d', (), {'geometry': type('g', (), {'TriangleMesh': FakeMesh}),
                            'utility': type('u', (), {'Vector3dVector': np.asarray, 'Vector3iVector': np.asarray})})


class StatueSubmeshTests(unittest.TestCase):
    def test_exports_exactly_the_statue_part_of_the_merged_refined_mesh(self):
        from pipeline_tools.object_mesh import statue_submesh
        statue_vertices = np.arange(12, dtype=float).reshape(4, 3)
        background_vertices = 100 + np.arange(9, dtype=float).reshape(3, 3)
        merged = FakeMesh(np.concatenate([statue_vertices, background_vertices]),
                          [[0, 1, 2], [0, 2, 3], [4, 5, 6]])
        split = dict(statue_vertices=4, statue_triangles=2, total_vertices=7, total_triangles=3)
        statue = statue_submesh(merged, split, FAKE_O3D)
        np.testing.assert_array_equal(statue.vertices, statue_vertices)
        np.testing.assert_array_equal(statue.triangles, [[0, 1, 2], [0, 2, 3]])
        with self.assertRaisesRegex(ValueError, 'tidak cocok'):
            statue_submesh(merged, dict(split, total_triangles=4), FAKE_O3D)
        with self.assertRaisesRegex(ValueError, 'tidak valid'):
            statue_submesh(merged, dict(split, statue_vertices=2), FAKE_O3D)


if __name__ == '__main__':
    unittest.main()
