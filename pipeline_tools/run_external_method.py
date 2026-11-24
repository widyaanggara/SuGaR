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
from pipeline_tools.visual_protocol import specification
from pipeline_tools.run_settings import add_resolution_arguments, resolution_settings


def commands(method, python, scene, output, resolution=-1):
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


def portable_identity(identity):
    """Dataset identity that survives moving/copying the project folder.

    The absolute scene path and image modification times change when the project is moved,
    copied or synced, while the photos stay the same. Photos are compared by name and size;
    the COLMAP model and every other setting keep their exact comparison.
    """
    identity = dict(identity)
    scene = identity.get('scene')
    if isinstance(scene, dict):
        scene = {key: value for key, value in scene.items() if key != 'scene'}
        if isinstance(scene.get('images'), dict):
            scene['images'] = {name: value.get('size') if isinstance(value, dict) else value
                               for name, value in scene['images'].items()}
        identity['scene'] = scene
    return identity


def training_identities_match(previous, current):
    """Compare effective settings; shared camera sizing overrides native -r.

    Preserve all other checks, including dataset, protocol, initialization and
    YAML settings. Existing receipts need no migration or checkpoint deletion.
    """
    if not isinstance(previous, dict):
        return False
    previous, current = portable_identity(previous), portable_identity(current)
    if previous.get('visual_protocol') and current.get('visual_protocol'):
        previous.pop('resolution', None)
        current.pop('resolution', None)
    return previous == current


def run(args):
    mesh_only = getattr(args, 'mesh_only', False)
    mesh_backend = getattr(args, 'mesh_backend', 'native')
    if mesh_only and args.method != 'radegs':
        raise ValueError('--mesh_only saat ini khusus RaDe-GS.')
    if mesh_only and (args.evaluate_only or getattr(args, 'train_only', False)):
        raise ValueError('--mesh_only tidak dapat digabung dengan --evaluate_only atau --train_only.')
    scene_pairs(args.statue_class, args.model_id)
    scene = ROOT / 'datasets' / args.statue_class / args.model_id
    experiment = getattr(args, 'experiment', 'baseline')
    output = output_base(ROOT, experiment) / args.method / args.statue_class / args.model_id
    initial_ply = Path(args.initial_ply).resolve() if getattr(args, 'initial_ply', None) else None
    if experiment == 'scaffold_anchor' and not args.evaluate_only and not mesh_only and (not initial_ply or not initial_ply.is_file()):
        raise ValueError('Ablasi memerlukan initial_ply anchor Scaffold-GS yang valid.')
    if experiment == 'baseline' and initial_ply:
        raise ValueError('Baseline tidak boleh memakai anchor Scaffold.')
    repo = ROOT / 'external' / METHODS[args.method][1]
    shared = getattr(args, 'visual_protocol', 'legacy') == 'shared'
    sizing = resolution_settings(vars(args))
    protocol = specification(sizing['image_downscale'], sizing['image_max_size']) if shared else None
    if not has_colmap_model(scene):
        raise ValueError('Jalankan COLMAP terlebih dahulu: images/ dan sparse/0 harus lengkap.')
    if not (repo / 'train.py').is_file():
        raise ValueError(f'Repository belum tersedia: {repo}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
    print(f'[CONFIG] Metode={args.method}; repository={repo}; python={args.python}; '
          f'resolusi={args.resolution} '
          + ('(otomatis, lebar maksimum 1600 px)' if args.resolution == -1 else '(faktor downscale)'),
          flush=True)
    # Fail before writing research artifacts if the selected environment is not usable.
    try:
        subprocess.run([args.python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA tidak tersedia"; '
                        'print("[GPU]", torch.cuda.get_device_name(0), flush=True)'],
                       cwd=repo, env=env, check=True)
    except FileNotFoundError as exc:
        raise ValueError(f'Python metode tidak ditemukan: {args.python}. '
                         'Periksa path environment pada server (bin/python, bukan bin/pytho).') from exc
    plan = None
    if args.method == 'radegs' and getattr(args, 'radegs_profile', 'legacy') == 'yaml':
        config_name = 'tanks_temples.yaml' if args.statue_class == 'TanksAndTemples' else 'togog_bali.yaml'
        config_path = ROOT / 'pipeline_tools/configs/radegs' / config_name
        try:
            result = subprocess.run([args.python, str(ROOT / 'pipeline_tools/radegs_yaml.py'),
                '--repo', str(repo), '--config', str(config_path), '--scene', str(scene), '--output', str(output)],
                cwd=repo, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Gagal membaca/validasi {config_path}:\n{exc.stderr}') from exc
        plan = json.loads(result.stdout)
        print(f'[YAML] {config_path}; resolusi bawaan YAML={plan["resolution"]}; '
              f'parameter training={plan["config"]["training"]["extra_args"]}. '
              + ('Resolusi efektif mengikuti protokol bersama.' if shared else
                 '--resolution dashboard digantikan oleh YAML untuk RaDe-GS.'), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'togoglab_run.json'
    identity = {'scene': scene_identity(scene), 'resolution': args.resolution, 'experiment': experiment,
                'initial_sha256': hashlib.sha256(initial_ply.read_bytes()).hexdigest() if initial_ply else None}
    if plan:
        identity['resolution'] = plan['resolution']
        identity['radegs_yaml'] = {key: plan['config'][key] for key in
            ('training', 'rendering', 'image_metrics', 'mesh', 'resource_guard')}
    if shared:
        identity['visual_protocol'] = protocol
        print(f'[PROTOCOL] {protocol}; overrides native/YAML image sizing for training and rendering.', flush=True)
    previous_run = json.loads(receipt.read_text()) if receipt.is_file() else {}
    matches = training_identities_match(previous_run.get('identity'), identity)
    if previous_run and not matches and not args.no_resume and not args.evaluate_only and not mesh_only:
        raise ValueError('Konfigurasi/dataset berubah. Nonaktifkan Lewati tahap yang selesai untuk training ulang.')
    postprocess_only = getattr(args, 'postprocess_only', False) or mesh_only
    if (plan or shared) and not previous_run and trained(output) and not (args.no_resume or args.evaluate_only or postprocess_only):
        raise ValueError('Checkpoint lama tidak memiliki identitas YAML. Cadangkan hasil lama lalu '
                         'nonaktifkan Lewati tahap yang selesai untuk training dengan YAML.')
    checkpoint_matches = trained(output) and previous_run.get('training_checkpoint') == file_signature(checkpoint(output))
    imported_baseline = not previous_run and experiment == 'baseline'
    skip_training = args.evaluate_only or postprocess_only or (not args.no_resume and trained(output) and ((matches and checkpoint_matches) or imported_baseline))
    if (args.evaluate_only or postprocess_only) and not trained(output):
        raise ValueError('Checkpoint 30.000 iterasi dan cfg_args belum lengkap.')
    if not args.evaluate_only and not mesh_only:
        if (args.no_resume and not postprocess_only) or not matches:
            previous_run = {'identity': identity}
            if not skip_training:
                previous_run['ignored_checkpoints'] = {p.name: file_signature(p) for p in output.glob('chkpnt*.pth')}
        if skip_training and trained(output):
            previous_run['training_checkpoint'] = file_signature(checkpoint(output))
        atomic_json(receipt, previous_run)
    training_scene = staged_anchor_scene(scene, output, initial_ply) if initial_ply and args.method == 'radegs' else scene
    stages = commands(args.method, args.python, training_scene, output, args.resolution)
    if plan:
        atomic_json(output / 'radegs_effective_config.json', plan['config'])
        names = [('Training', 'training'), ('Rendering', 'rendering'),
                 ('Evaluasi_Visual', 'image_metrics'), ('Ekstraksi_Mesh', 'mesh_extraction')]
        stages = [(name, [args.python, '-u', *plan['commands'][key][1:]]) for name, key in names]
        train_command = stages[0][1]
        train_command[train_command.index('-s') + 1] = str(training_scene)
    if initial_ply and args.method == '2dgs':
        stages[0][1].extend(['--init_ply', str(initial_ply)])
    if args.method == 'radegs' and mesh_backend == 'scalable':
        script = repo / 'mesh_extract_tnt.py'
        if not script.is_file():
            raise FileNotFoundError(f'Backend ScalableTSDFVolume tidak tersedia: {script}')
        stages = [(stage, [args.python, '-u', str(ROOT / 'pipeline_tools/radegs_mesh.py'),
                   '--repo', str(repo), '-m', str(output),
                   '--iteration', '30000', '-r', '2', '--num_cluster', '8', '--depth_ratio', '0.5',
                   '--voxel_size', '0.009', '--max_depth', '8.0'] if stage == 'Ekstraksi_Mesh' else command)
                  for stage, command in stages]
    if mesh_only:
        stages = [(stage, command) for stage, command in stages if stage == 'Ekstraksi_Mesh']
    if shared:
        wrapped = []
        for stage, command in stages:
            if stage in ('Training', 'Rendering', 'Evaluasi_Visual'):
                command = [args.python, '-u', str(ROOT / 'pipeline_tools/external_visual_launcher.py'),
                    '--repo', str(repo), '--image_downscale', str(sizing['image_downscale']),
                    '--image_max_size', str(sizing['image_max_size']), *command[2:]]
            wrapped.append((stage, command))
        stages = wrapped
    else:
        stages = [(stage, [args.python, '-u', str(ROOT / 'pipeline_tools/external_visual_launcher.py'),
                   '--repo', str(repo), '--visual_protocol', 'legacy', *command[2:]]
                  if stage == 'Training' else command) for stage, command in stages]
    if getattr(args, 'train_only', False):
        stages = stages[:1]
    current_checkpoint = file_signature(checkpoint(output)) if trained(output) else None
    if (not mesh_only and not args.no_resume and not args.evaluate_only and skip_training and visual_metrics(output)
            and mesh_path(output, args.method).is_file() and previous_run.get('evaluated_checkpoint') == current_checkpoint
            and previous_run.get('evaluated_protocol') == protocol
            and (args.method != 'radegs' or (previous_run.get('mesh_backend', 'native') == mesh_backend
                 and previous_run.get('mesh_checkpoint') == current_checkpoint))):
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
            stage_env = env.copy()
            vram_report = None
            if stage == 'Training':
                import uuid
                vram_report = output / 'resource_reports' / f'training_{uuid.uuid4().hex}.json'
                stage_env['TOGOGLAB_TRAINING_VRAM_REPORT'] = str(vram_report)
            logger.set_parameters({'phase': stage, 'repository': str(repo), 'python': args.python})
            if stage == 'Training' and not args.no_resume and matches:
                ignored = previous_run.get('ignored_checkpoints', {})
                saved = sorted((p for p in output.glob('chkpnt*.pth') if ignored.get(p.name) != file_signature(p)),
                               key=lambda p: int(p.stem[6:]) if p.stem[6:].isdigit() else -1)
                if saved and saved[-1].stem[6:].isdigit():
                    command.extend(['--start_checkpoint', str(saved[-1])])
                    if args.method == 'radegs':
                        # Native RaDe-GS saves full training state, including NumPy
                        # scalars. PyTorch 2.6's weights-only default cannot load it.
                        # Limit compatibility to this resume child and a checkpoint
                        # from this run's matching local output, never global settings.
                        stage_env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
                        print(f'[RESUME] Memuat state training lokal tepercaya: {saved[-1]} '
                              '(kompatibilitas checkpoint PyTorch 2.6+).', flush=True)
            if stage == 'Evaluasi_Visual':
                # Native scripts can swallow exceptions; never accept stale metrics as success.
                result_path = output / 'results.json'
                previous = result_path.stat().st_mtime_ns if result_path.exists() else None
            logger.start_stage(stage)
            if stage == 'Ekstraksi_Mesh':
                mesh_result = mesh_path(output, args.method)
                mesh_before = file_signature(mesh_result) if mesh_result.is_file() else None
            if stage == 'Ekstraksi_Mesh' and mesh_backend == 'scalable':
                import time
                old_meshes = [output / name for name in ('recon.ply', 'recon_post.ply') if (output / name).is_file()]
                if old_meshes:
                    backup = output / 'mesh_backups' / str(time.time_ns())
                    backup.mkdir(parents=True)
                    for old_mesh in old_meshes:
                        shutil.copy2(old_mesh, backup / old_mesh.name)
                    print(f'[MESH] Mesh sebelumnya dicadangkan: {backup}', flush=True)
                print('[MESH] ScalableTSDFVolume; integrasi per kamera; voxel=0.009, depth=8.0. '
                      'Backend berbeda dari VoxelBlockGrid; hasil geometri perlu dievaluasi ulang.', flush=True)
            if plan and stage == 'Ekstraksi_Mesh':
                subprocess.run([args.python, str(ROOT / 'pipeline_tools/radegs_yaml.py'),
                    '--repo', str(repo), '--guard', str(output / 'radegs_effective_config.json')],
                    cwd=repo, env=stage_env, check=True)
            print('[RUN] ' + subprocess.list2cmdline(command), flush=True)
            process = subprocess.Popen(command, cwd=repo, env=stage_env)
            try:
                if stage == 'Training':
                    logger.start_vram_monitor(process.pid)
                returncode = process.wait()
                if returncode != 0:
                    detail = f'signal {-returncode}' if returncode < 0 else f'exit code {returncode}'
                    raise RuntimeError(f'{stage} gagal ({detail}). Periksa log di atas; '
                                       'jika proses dibunuh OS, cek log kernel WSL.')
            finally:
                logger.stop_vram_monitor()
                if vram_report is not None:
                    logger.record_child_vram(vram_report, process.pid)
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
                if shared:
                    atomic_json(output / 'evaluation_protocol.json', {
                        'visual_protocol': protocol,
                        'training_identity': previous_run.get('identity'),
                        'checkpoint_signature': file_signature(checkpoint(output))})
            if stage == 'Ekstraksi_Mesh' and not mesh_path(output, args.method).is_file():
                raise RuntimeError('Ekstraksi tidak menghasilkan mesh.')
            if stage == 'Ekstraksi_Mesh':
                if dict(file_signature(mesh_result)) == dict(mesh_before or {}):  # exact: detects a rewrite
                    raise RuntimeError('Ekstraksi tidak memperbarui mesh; hasil lama tidak dianggap sukses.')
                previous_run['mesh_backend'] = mesh_backend
                previous_run['mesh_checkpoint'] = file_signature(checkpoint(output))
                atomic_json(receipt, previous_run)
            if stage == 'Rendering' and args.method == '2dgs' and not args.evaluate_only and not mesh_path(output, args.method).is_file():
                raise RuntimeError('Rendering 2DGS tidak menghasilkan mesh.')
            if stage in ('Training', 'Evaluasi_Visual'):
                logger.finalize(True)
        except BaseException:
            logger.finalize(False)
            raise
    if not mesh_only and not getattr(args, 'train_only', False) and visual_metrics(output) and trained(output):
        previous_run['evaluated_checkpoint'] = file_signature(checkpoint(output))
        previous_run['evaluated_protocol'] = protocol
        atomic_json(receipt, previous_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--radegs_profile', choices=('yaml', 'legacy'), default='yaml',
                        help='yaml: konfigurasi Togog Bali/Tanks and Temples; legacy: parameter runner lama.')
    parser.add_argument('--statue_class', required=True)
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--visual_protocol', choices=('shared', 'legacy'), default='shared')
    add_resolution_arguments(parser)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resolution', type=int, choices=(-1, 1, 2, 4, 8), default=-1,
                        help='-1: otomatis, lebar maksimum 1600 px; 1: ukuran asli; 2/4/8: downscale.')
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--evaluate_only', action='store_true')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--postprocess_only', action='store_true')
    parser.add_argument('--mesh_only', action='store_true', help='Ulangi mesh RaDe-GS dari checkpoint tanpa training/evaluasi.')
    parser.add_argument('--mesh_backend', choices=('native', 'scalable'), default='native',
                        help='scalable memakai mesh_extract_tnt.py (ScalableTSDFVolume), bukan VoxelBlockGrid.')
    parser.add_argument('--experiment', choices=('baseline', 'scaffold_anchor'), default='baseline')
    parser.add_argument('--initial_ply')
    parser.add_argument('--object_region_json')
    args = parser.parse_args()
    with gpu_lock(args.gpu):
        run(args)


if __name__ == '__main__':
    main()
