"""Check native method environments on the training server without starting training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(name, python, repo, entrypoints, gpu):
    print(f'\n[{name}] Python: {python}\nRepository: {repo}', flush=True)
    if not (repo / 'train.py').is_file():
        print('FAIL: repository belum lengkap.', flush=True)
        return False
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    probes = [('PyTorch + CUDA', [python, '-c',
        'import torch; print("PyTorch:", torch.__version__); '
        'assert torch.cuda.is_available(), "CUDA tidak tersedia"; '
        'print("GPU:", torch.cuda.get_device_name(0))'])]
    probes += [(label, [python, *args, '--help']) for label, args in entrypoints]
    success = True
    for label, command in probes:
        try:
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True,
                                    text=True, encoding='utf-8', errors='replace', timeout=90)
            if result.returncode:
                success = False
                print(f'FAIL {label}: exit {result.returncode}\n{result.stdout[-2000:]}{result.stderr[-4000:]}', flush=True)
                if label == 'PyTorch + CUDA':
                    break
            else:
                print(f'PASS {label}', flush=True)
                if label == 'PyTorch + CUDA':
                    print(result.stdout.strip(), flush=True)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f'FAIL {label}: {error}', flush=True)
            success = False
            if label == 'PyTorch + CUDA':
                break
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python_2dgs', default=sys.executable)
    parser.add_argument('--python_radegs', default=sys.executable)
    parser.add_argument('--scaffold_python', default=sys.executable)
    parser.add_argument('--scaffold_repo', type=Path, default=ROOT / 'external/Scaffold-GS')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('GPU harus >= 0.')
    probes = [
        ('2DGS', args.python_2dgs, ROOT / 'external/2DGS',
         [(script, [script]) for script in ('train.py', 'render.py', 'metrics.py')]),
        ('RaDe-GS', args.python_radegs, ROOT / 'external/RaDe-GS',
         [(script, [script]) for script in ('train.py', 'render.py', 'metric.py', 'mesh_extract.py')]),
        ('Scaffold-GS', args.scaffold_python, args.scaffold_repo.resolve(),
         [('adapter training', [str(ROOT / 'pipeline_tools/scaffold_train.py'), '--repo', str(args.scaffold_repo.resolve())])]),
    ]
    results = [check(name, python, repo, stages, args.gpu) for name, python, repo, stages in probes]
    print('\n' + ('PASS semua environment.' if all(results) else 'FAIL: perbaiki environment yang ditandai di atas.'))
    print('Pemeriksaan ini tidak menjalankan training; kecukupan VRAM dan kualitas data belum diuji.')
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
