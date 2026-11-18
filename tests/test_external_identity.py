import copy
import unittest

from pipeline_tools.run_external_method import training_identities_match
from pipeline_tools.visual_protocol import specification


class ExternalIdentityTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(scene={'images': {'a.png': {'size': 42}}}, resolution=1,
                        experiment='baseline', initial_sha256=None,
                        visual_protocol=specification(1, 1600))

    def test_shared_native_resolution_change_preserves_reuse(self):
        new = copy.deepcopy(self.old)
        new['resolution'] = -1
        self.assertTrue(training_identities_match(self.old, new))
        self.assertEqual(self.old['resolution'], 1)
        self.assertEqual(new['resolution'], -1)

    def test_effective_changes_still_block_reuse(self):
        for key, value in (
            ('scene', {'images': {}}), ('visual_protocol', specification(1, 1920)),
            ('initial_sha256', 'changed'), ('experiment', 'scaffold_anchor'),
            ('radegs_yaml', {'training': 'changed'}), ('visual_protocol', None),
        ):
            with self.subTest(key=key):
                new = {**self.old, key: value, 'resolution': -1}
                self.assertFalse(training_identities_match(self.old, new))

    def test_moved_or_copied_project_still_reuses_the_same_photos(self):
        old = {**self.old, 'scene': {'scene': '/home/wisnu/SuGaR/datasets/Dewa/model_01', 'colmap': {'c': 1},
                                     'images': {'a.png': {'size': 42, 'mtime_ns': 1}}}}
        moved = {**self.old, 'resolution': -1, 'scene': {'scene': '/home/anggara/SuGaR/datasets/Dewa/model_01',
                 'colmap': {'c': 1}, 'images': {'a.png': {'size': 42, 'mtime_ns': 99}}}}
        self.assertTrue(training_identities_match(old, moved))
        for scene in ({**moved['scene'], 'images': {'a.png': {'size': 43, 'mtime_ns': 99}}},
                      {**moved['scene'], 'images': {'b.png': {'size': 42, 'mtime_ns': 99}}},
                      {**moved['scene'], 'colmap': {'c': 2}}):
            with self.subTest(scene=scene):
                self.assertFalse(training_identities_match(old, {**moved, 'scene': scene}))

    def test_legacy_resolution_remains_significant(self):
        old = {k: v for k, v in self.old.items() if k != 'visual_protocol'}
        self.assertFalse(training_identities_match(old, {**old, 'resolution': -1}))
        self.assertTrue(training_identities_match(old, old))
        self.assertFalse(training_identities_match(None, old))


if __name__ == '__main__':
    unittest.main()
