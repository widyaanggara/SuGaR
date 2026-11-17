import unittest
import numpy as np
from pipeline_tools.sugar_focus import camera_focus_region


def cameras_at(center, scale=1.):
    result = []
    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        position = np.asarray(center) + scale * np.array([4 * np.cos(angle), 4 * np.sin(angle), 1.])
        direction = np.asarray(center) - position
        direction /= np.linalg.norm(direction)
        right = np.cross(direction, [0, 0, 1.])
        right /= np.linalg.norm(right)
        rotation = np.column_stack((right, np.cross(direction, right), direction))
        result.append(dict(position=position.tolist(), rotation=rotation.tolist(),
                           width=1600, height=1200, fx=2000, fy=2000))
    return result


class FocusTests(unittest.TestCase):
    def test_converging_views_bound_target_and_exclude_distant_scene(self):
        center = np.array([10., -4., 2.])
        region = camera_focus_region(cameras_at(center))
        np.testing.assert_allclose((np.array(region['min']) + region['max']) / 2, center)
        self.assertLess(region['max'][0], center[0] + 4)
        self.assertGreater(region['max'][2], center[2] + 1.5)

    def test_colmap_scale_and_translation_are_respected(self):
        a = camera_focus_region(cameras_at([0, 0, 0]))
        b = camera_focus_region(cameras_at([8, 3, -7], 5))
        np.testing.assert_allclose(b['min'], np.array(a['min']) * 5 + [8, 3, -7])
        np.testing.assert_allclose(b['max'], np.array(a['max']) * 5 + [8, 3, -7])

    def test_parallel_and_invalid_views_fail_instead_of_guessing(self):
        cameras = cameras_at([0, 0, 0])
        for camera in cameras:
            camera['rotation'] = np.eye(3).tolist()
        with self.assertRaises(ValueError):
            camera_focus_region(cameras)
        with self.assertRaises(ValueError):
            camera_focus_region(cameras[:2])


if __name__ == '__main__':
    unittest.main()
