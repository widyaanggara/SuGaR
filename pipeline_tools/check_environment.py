"""Run on the target Ubuntu server before submitting the 18-model queue."""
import argparse
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_tools.gpu_lock import gpu_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip_smoke', action='store_true', help='Only inspect versions/imports, without CUDA kernel tests.')
    args = parser.parse_args()
    errors = []
    report = {'platform': platform.platform(), 'python': platform.python_version(), 'gpu_id': args.gpu}
    if sys.platform != 'linux':
        errors.append('Jalankan perintah ini di Ubuntu WSL, bukan Python native Windows.')
    if sys.version_info[:2] != (3, 9):
        errors.append('Environment proyek memerlukan Python 3.9. Jalankan conda activate sugar.')
    os_release = Path('/etc/os-release')
    if os_release.is_file():
        report['os_release'] = os_release.read_text().strip()
        if 'VERSION_ID="22.04"' not in report['os_release']:
            errors.append('Ubuntu target belum teridentifikasi sebagai versi 22.04.')
    for executable in ('colmap', 'nvcc', 'g++', 'ffmpeg', 'nvidia-smi'):
        report[executable] = shutil.which(executable)
        if not report[executable]:
            errors.append(f'{executable} tidak ditemukan di PATH.')
    if report['nvcc']:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        report['nvcc_version'] = result.stdout.strip()
        if result.returncode or 'release 11.8' not in result.stdout:
            errors.append('nvcc harus berasal dari CUDA Toolkit 11.8.')
    if report['g++']:
        result = subprocess.run(['g++', '-dumpversion'], capture_output=True, text=True)
        report['g++_version'] = result.stdout.strip()
        if not result.stdout.strip().isdigit() or int(result.stdout.strip()) > 11:
            errors.append('Gunakan GCC/G++ 11 untuk kombinasi CUDA 11.8 proyek ini.')
    if report['nvidia-smi']:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.free',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        report['gpu_inventory'] = result.stdout.strip()
        if result.returncode:
            errors.append('nvidia-smi gagal. Periksa driver NVIDIA Windows dan WSL 2.')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    expected = {'torch': '2.0.1', 'torchvision': '0.15.2', 'pytorch3d': '0.7.4'}
    for module in ('torch', 'torchvision', 'pytorch3d', 'open3d', 'cv2', 'numpy', 'scipy',
                   'rich', 'plyfile', 'mcubes', 'diff_gaussian_rasterization', 'simple_knn._C'):
        try:
            loaded = importlib.import_module(module)
            version = getattr(loaded, '__version__', 'import OK')
            report[module] = version
            if module in expected and version.split('+')[0] != expected[module]:
                errors.append(f'{module}: perlu {expected[module]}, ditemukan {version}.')
        except Exception as exc:
            errors.append(f'{module}: {exc}')
    torch = sys.modules.get('torch')
    if torch:
        report['torch_cuda'] = torch.version.cuda
        if torch.version.cuda != '11.8' or not torch.cuda.is_available():
            errors.append('PyTorch harus memakai CUDA 11.8 dan torch.cuda.is_available() harus True.')
        elif not errors:
            with gpu_lock(args.gpu):
                torch.cuda.set_device(0)
                report['selected_gpu'] = torch.cuda.get_device_name(0)
                report['compute_capability'] = torch.cuda.get_device_capability(0)
                if not args.skip_smoke:
                    try:
                        from pytorch3d.ops import knn_points
                        from simple_knn._C import distCUDA2
                        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
                        points = torch.rand(16, 3, device='cuda')
                        assert torch.isfinite(distCUDA2(points)).all()
                        assert torch.isfinite(knn_points(points[None], points[None], K=2).dists).all()
                        settings = GaussianRasterizationSettings(image_height=8, image_width=8,
                            tanfovx=1., tanfovy=1., bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
                            viewmatrix=torch.eye(4, device='cuda'), projmatrix=torch.eye(4, device='cuda'),
                            sh_degree=0, campos=torch.zeros(3, device='cuda'), prefiltered=False, debug=False)
                        image, radii = GaussianRasterizer(settings)(means3D=torch.tensor([[0., 0., 1.]], device='cuda'),
                            means2D=torch.zeros(1, 3, device='cuda'), colors_precomp=torch.ones(1, 3, device='cuda'),
                            opacities=torch.full((1, 1), .5, device='cuda'), scales=torch.full((1, 3), .1, device='cuda'),
                            rotations=torch.tensor([[1., 0., 0., 0.]], device='cuda'))
                        torch.cuda.synchronize()
                        assert image.shape == (3, 8, 8) and torch.isfinite(image).all()
                        report['cuda_smoke'] = 'PASS: simple-knn, PyTorch3D KNN, Gaussian rasterizer'
                    except Exception as exc:
                        errors.append(f'CUDA smoke test gagal: {exc}')
    report['errors'] = errors
    report['status'] = 'FAILED' if errors else 'PASSED'
    folder = PROJECT_ROOT / 'outputs/logs'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'environment_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
