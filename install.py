"""Install the pinned Ubuntu environment and fail immediately on setup errors."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*arguments):
    subprocess.run(list(arguments), cwd=str(ROOT), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--with_nvdiffrast', action='store_true', help='Install pinned optional nvdiffrast v0.3.3.')
    options.add_argument('--no_nvdiffrast', action='store_true', help='Compatibility alias; nvdiffrast is skipped by default.')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Jalankan installer dari Ubuntu 22.04/WSL 2.')
    conda = shutil.which('conda')
    if conda is None:
        parser.error('Conda belum tersedia. Pasang Miniconda di Ubuntu terlebih dahulu.')
    environments = json.loads(subprocess.check_output([conda, 'env', 'list', '--json'], text=True))['envs']
    if not any(Path(path).name == 'sugar' for path in environments):
        run(conda, 'env', 'create', '-f', str(ROOT / 'environment.yml'))
    run(conda, 'install', '-n', 'sugar', '-y', 'nvidia/label/cuda-11.8.0::cuda-toolkit')
    for package in ('diff-gaussian-rasterization', 'simple-knn'):
        run(conda, 'run', '--no-capture-output', '-n', 'sugar', 'python', '-m', 'pip', 'install',
            '--no-build-isolation', '-e', str(ROOT / 'gaussian_splatting/submodules' / package))
    run(conda, 'run', '-n', 'sugar', 'python', '-m', 'pip', 'install',
        'opencv-python-headless==4.10.0.84', 'ninja==1.11.1.1')
    if args.with_nvdiffrast:
        run(conda, 'run', '--no-capture-output', '-n', 'sugar', 'python', '-m', 'pip', 'install',
            '--no-build-isolation', 'git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3')
    run(conda, 'run', '-n', 'sugar', 'python', 'pipeline_tools/init_dataset_folders.py')
    print('Instalasi selesai. Jalankan conda activate sugar lalu python pipeline_tools/check_environment.py')


if __name__ == '__main__':
    main()
