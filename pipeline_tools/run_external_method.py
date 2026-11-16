"""Run 2DGS/RaDe-GS in their own Python environment under TogogLab's GPU lock."""
import argparse
import os
import json
import hashlib
import shutil
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_tools.dataset_registry import scene_pairs, has_colmap_model
from pipeline_tools.external_methods import METHODS, trained, visual_metrics, mesh_path, output_base, checkpoint
from pipeline_tools.recovery_state import atomic_json, scene_identity, fingerprint
from pipeline_tools.dataset_registry import file_signature
from pipeline_tools.gpu_lock import gpu_lock
from pipeline_tools.logger_utils import ThesisLogger


def commands(method, python, scene, output, resolution=1):
    common = ['-s', str(scene), '-m', str(output)]
    train = [python, '-u', 'train.py', *common, '--iterations', '30000', '--eval',
             '-r', str(resolution), '--checkpoint_iterations', '7000', '15000', '30000']
    render = [python, '-u', 'render.py', *common, '--iteration', '30000', '--skip_train']
    if method == '2dgs':
        render.remove('--skip_train')  # Native mesh export lives in train/ours_30000.
    metric = [python, '-u', 'metrics.py' if method == '2dgs' else 'metric.py', '-m', str(output)]
    result = [('Training', train), ('Rendering', render), ('Evaluasi_Visual', metric)]
    if method == 'radegs':
        result.append(('Ekstraksi_Mesh', [python, '-u', 'mesh_extract.py', '-m', str(output),
                                        '-r', '2', '--num_cluster', '8', '--depth_ratio', '0.5']))
    return result


def staged_anchor_scene(scene, output, initial_ply):
    """RaDe-GS consumes sparse/0/points3D.ply; never replace the original COLMAP."""
    identity = {'scene': scene_identity(scene), 'anchor_sha256': hashlib.sha256(initial_ply.read_bytes()).hexdigest()}
    target = output / 'initialization' / fingerprint(identity)
    receipt = target / 'ready.json'
    if receipt.is_file() and (target / 'sparse/0/points3D.ply').is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)
    def link_or_copy(source, destination):
        if Path(destination).exists():
            return destination
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination
    for folder in scene.iterdir():
        if folder.is_dir() and (folder.name == 'images' or folder.name.startswith('images_')):
            shutil.copytree(folder, target / folder.name, copy_function=link_or_copy, dirs_exist_ok=True)
    shutil.copytree(scene / 'sparse', target / 'sparse', dirs_exist_ok=True)
    shutil.copy2(initial_ply, target / 'sparse/0/points3D.ply')
    atomic_json(receipt, identity)
    return target


def run(args):
    scene_pairs(args.statue_class, args.model_id)
    scene = ROOT / 'datasets' / args.statue_class / args.model_id
    experiment = getattr(args, 'experiment', 'baseline')
    output = output_base(ROOT, experiment) / args.method / args.statue_class / args.model_id
    initial_ply = Path(args.initial_ply).resolve() if getattr(args, 'initial_ply', None) else None
    if experiment == 'scaffold_anchor' and not args.evaluate_only and (not initial_ply or not initial_ply.is_file()):
        raise ValueError('Ablasi memerlukan initial_ply anchor Scaffold-GS yang valid.')
    if experiment == 'baseline' and initial_ply:
        raise ValueError('Baseline tidak boleh memakai anchor Scaffold.')
    repo = ROOT / 'external' / METHODS[args.method][1]
    if not has_colmap_model(scene):
        raise ValueError('Jalankan COLMAP terlebih dahulu: images/ dan sparse/0 harus lengkap.')
    if not (repo / 'train.py').is_file():
        raise ValueError(f'Repository belum tersedia: {repo}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1')
    # Fail before writing research artifacts if the selected environment is not usable.
    subprocess.run([args.python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA tidak tersedia"'],
                   cwd=repo, env=env, check=True)
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'togoglab_run.json'
    identity = {'scene': scene_identity(scene), 'resolution': args.resolution, 'experiment': experiment,
                'initial_sha256': hashlib.sha256(initial_ply.read_bytes()).hexdigest() if initial_ply else None}
    previous_run = json.loads(receipt.read_text()) if receipt.is_file() else {}
    matches = previous_run.get('identity') == identity
    if previous_run and not matches and not args.no_resume and not args.evaluate_only:
        raise ValueError('Konfigurasi/dataset berubah. Nonaktifkan Lewati tahap yang selesai untuk training ulang.')
    postprocess_only = getattr(args, 'postprocess_only', False)
    checkpoint_matches = trained(output) and previous_run.get('training_checkpoint') == file_signature(checkpoint(output))
    imported_baseline = not previous_run and experiment == 'baseline'
    skip_training = args.evaluate_only or postprocess_only or (not args.no_resume and trained(output) and ((matches and checkpoint_matches) or imported_baseline))
    if (args.evaluate_only or postprocess_only) and not trained(output):
        raise ValueError('Checkpoint 30.000 iterasi dan cfg_args belum lengkap.')
    if not args.evaluate_only:
        if (args.no_resume and not postprocess_only) or not matches:
            previous_run = {'identity': identity}
            if not skip_training:
                previous_run['ignored_checkpoints'] = {p.name: file_signature(p) for p in output.glob('chkpnt*.pth')}
        if skip_training and trained(output):
            previous_run['training_checkpoint'] = file_signature(checkpoint(output))
        atomic_json(receipt, previous_run)
    training_scene = staged_anchor_scene(scene, output, initial_ply) if initial_ply and args.method == 'radegs' else scene
    stages = commands(args.method, args.python, training_scene, output, args.resolution)
    if initial_ply and args.method == '2dgs':
        stages[0][1].extend(['--init_ply', str(initial_ply)])
    if getattr(args, 'train_only', False):
        stages = stages[:1]
    current_checkpoint = file_signature(checkpoint(output)) if trained(output) else None
    if (not args.no_resume and not args.evaluate_only and skip_training and visual_metrics(output)
            and mesh_path(output, args.method).is_file() and previous_run.get('evaluated_checkpoint') == current_checkpoint):
        print('[SKIP] Training, mesh, dan evaluasi metode sudah lengkap dan cocok.', flush=True)
        return
    for stage, command in stages:
        if stage == 'Rendering' and args.evaluate_only and args.method == '2dgs':
            command.extend(['--skip_mesh', '--skip_train'])
        if stage == 'Training' and skip_training:
            print('[SKIP] Checkpoint training dipakai kembali.', flush=True)
            continue
        if stage == 'Ekstraksi_Mesh' and args.evaluate_only:
            continue
        # Keep phase logs separate so visual evaluation does not replace training time.
        logger = ThesisLogger(args.statue_class, args.model_id, METHODS[args.method][0] + ('_scaffold_anchor' if experiment == 'scaffold_anchor' else ''))
        try:
            logger.set_parameters({'phase': stage, 'repository': str(repo), 'python': args.python})
            if stage == 'Training' and not args.no_resume and matches:
                ignored = previous_run.get('ignored_checkpoints', {})
                saved = sorted((p for p in output.glob('chkpnt*.pth') if ignored.get(p.name) != file_signature(p)),
                               key=lambda p: int(p.stem[6:]) if p.stem[6:].isdigit() else -1)
                if saved and saved[-1].stem[6:].isdigit():
                    command.extend(['--start_checkpoint', str(saved[-1])])
            if stage == 'Evaluasi_Visual':
                # Native scripts can swallow exceptions; never accept stale metrics as success.
                result_path = output / 'results.json'
                previous = result_path.stat().st_mtime_ns if result_path.exists() else None
            logger.start_stage(stage)
            print('[RUN] ' + subprocess.list2cmdline(command), flush=True)
            process = subprocess.Popen(command, cwd=repo, env=env)
            try:
                if stage == 'Training':
                    logger.start_vram_monitor(process.pid)
                if process.wait() != 0:
                    raise RuntimeError(f'{stage} gagal. Periksa log di atas.')
            finally:
                logger.stop_vram_monitor()
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            logger.end_stage(stage)
            if stage == 'Training' and not trained(output):
                raise RuntimeError('Training berakhir tanpa checkpoint lengkap.')
            if stage == 'Training':
                previous_run['training_checkpoint'] = file_signature(checkpoint(output))
                atomic_json(receipt, previous_run)
            if stage == 'Evaluasi_Visual':
                values = visual_metrics(output)
                if not values or result_path.stat().st_mtime_ns == previous:
                    raise RuntimeError('Evaluasi tidak menghasilkan PSNR/SSIM/LPIPS baru yang valid.')
                logger.set_metrics(psnr=values['PSNR'], ssim=values['SSIM'], lpips=values['LPIPS'])
            if stage == 'Ekstraksi_Mesh' and not mesh_path(output, args.method).is_file():
                raise RuntimeError('Ekstraksi tidak menghasilkan mesh.')
            if stage == 'Rendering' and args.method == '2dgs' and not args.evaluate_only and not mesh_path(output, args.method).is_file():
                raise RuntimeError('Rendering 2DGS tidak menghasilkan mesh.')
            if stage in ('Training', 'Evaluasi_Visual'):
                logger.finalize(True)
        except BaseException:
            logger.finalize(False)
            raise
    if not getattr(args, 'train_only', False) and visual_metrics(output) and trained(output):
        previous_run['evaluated_checkpoint'] = file_signature(checkpoint(output))
        atomic_json(receipt, previous_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--statue_class', required=True)
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resolution', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--evaluate_only', action='store_true')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--postprocess_only', action='store_true')
    parser.add_argument('--experiment', choices=('baseline', 'scaffold_anchor'), default='baseline')
    parser.add_argument('--initial_ply')
    args = parser.parse_args()
    with gpu_lock(args.gpu):
        run(args)


if __name__ == '__main__':
    main()
