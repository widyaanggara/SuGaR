"""Exercise native circular imports in an isolated process without CUDA."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExternalVisualLauncherTests(unittest.TestCase):
    def test_scene_initialization_and_shared_sizing_before_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            files = {
                'scene/__init__.py': 'from utils.camera_utils import cameraList_from_camInfos\n',
                'scene/cameras.py': 'class Camera: pass\n',
                'utils/__init__.py': '',
                'utils/camera_utils.py': (
                    'from scene.cameras import Camera\n'
                    'def PILtoTorch(image, size): return size\n'
                    'def loadCam(args, index, info, scale):\n'
                    '    return PILtoTorch(info.image, (999, 999))\n'
                    'def cameraList_from_camInfos(info):\n'
                    '    return loadCam(None, 0, info, 1)\n'
                ),
                'utils/render_utils.py': 'save_img_u8 = None\n',
                'metrics.py': '',
            }
            entrypoint = (
                'from types import SimpleNamespace\n'
                'from scene import cameraList_from_camInfos\n'
                'info = SimpleNamespace(image=SimpleNamespace(size=(4000, 3000)))\n'
                'assert cameraList_from_camInfos(info) == (1600, 1200)\n'
            )
            files['train.py'] = entrypoint
            files['render.py'] = entrypoint + (
                'from utils.render_utils import save_img_u8\n'
                'from pipeline_tools.visual_protocol import save_rgb_png\n'
                'assert save_img_u8 is save_rgb_png\n'
            )
            for name, content in files.items():
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
            for script in ('train.py', 'render.py'):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / 'pipeline_tools/external_visual_launcher.py'),
                         '--repo', str(repo), script],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
