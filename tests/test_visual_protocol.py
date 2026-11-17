import argparse
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

from gaussian_splatting.utils.resolution_utils import scaled_resolution
from pipeline_tools.visual_protocol import install_camera_sizing, install_metrics, specification
from web_dashboard.server import build_command


class VisualProtocolTests(unittest.TestCase):
    def test_native_camera_loader_uses_sugar_sizes_for_all_orientations(self):
        for width, height, expected in ((4000, 3000, (1600, 1200)),
                                        (3000, 4000, (1200, 1600)),
                                        (800, 600, (800, 600))):
            image = types.SimpleNamespace(size=(width, height))
            converter = lambda image, size: size
            module = types.SimpleNamespace(PILtoTorch=converter)
            module.loadCam = lambda args, index, info, scale: module.PILtoTorch(info.image, (1920, 2560))
            install_camera_sizing(module, 1, 1600)
            self.assertEqual(module.loadCam(None, 0, types.SimpleNamespace(image=image), 1), expected)
            self.assertIs(module.PILtoTorch, converter)
            self.assertEqual(expected, scaled_resolution(width, height, 1, 1600))

    def test_camera_override_restored_after_failure_and_honors_downscale(self):
        converter = MagicMock()
        module = types.SimpleNamespace(PILtoTorch=converter)
        def fail(*args):
            module.PILtoTorch(types.SimpleNamespace(size=(800, 600)), None)
            raise ValueError('camera failed')
        module.loadCam = fail
        install_camera_sizing(module, 2, 1600)
        with self.assertRaises(ValueError):
            module.loadCam(None, 0, types.SimpleNamespace(image=types.SimpleNamespace(size=(800, 600))), 1)
        self.assertIs(module.PILtoTorch, converter)
        self.assertEqual(converter.call_args.args[1], (400, 300))

    def test_metric_adapter_uses_common_functions_and_signed_lpips_input(self):
        names = ('lpipsPyTorch', 'utils', 'utils.loss_utils', 'utils.image_utils',
                 'gaussian_splatting.utils.loss_utils', 'gaussian_splatting.utils.image_utils',
                 'gaussian_splatting.lpipsPyTorch', 'gaussian_splatting.lpipsPyTorch.modules',
                 'gaussian_splatting.lpipsPyTorch.modules.lpips')
        modules = {name: types.ModuleType(name) for name in names}
        modules['utils'].loss_utils = modules['utils.loss_utils']
        modules['utils'].image_utils = modules['utils.image_utils']
        ssim, psnr, criterion = MagicMock(), MagicMock(), MagicMock()
        modules['gaussian_splatting.utils.loss_utils'].ssim = ssim
        modules['gaussian_splatting.utils.image_utils'].psnr = psnr
        factory = MagicMock()
        factory.return_value.to.return_value.eval.return_value = criterion
        modules['gaussian_splatting.lpipsPyTorch.modules.lpips'].LPIPS = factory
        class Pixel(float):
            device = 'test-device'
        with patch.dict(sys.modules, modules):
            install_metrics()
            modules['lpipsPyTorch'].lpips(Pixel(.25), Pixel(.75))
            criterion.assert_called_once_with(-.5, .5)
            self.assertIs(modules['utils.loss_utils'].ssim, ssim)
            self.assertIs(modules['utils.image_utils'].psnr, psnr)

    def test_external_jobs_receive_same_sizing_as_sugar(self):
        for method in ('2dgs', 'radegs'):
            command = build_command(dict(action=method, class_name='Dewa', model_id='model_01',
                visual_protocol='shared', image_downscale=1, image_max_size=1600))
            self.assertEqual(command[command.index('--visual_protocol') + 1], 'shared')
            self.assertEqual(command[command.index('--image_max_size') + 1], '1600')
            self.assertEqual(command[command.index('--image_downscale') + 1], '1')


if __name__ == '__main__':
    unittest.main()
