import os
import sys
import shutil
import subprocess
import argparse
import struct
import json
import hashlib
from collections import deque
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline_tools.logger_utils import ThesisLogger
from pipeline_tools.dataset_registry import media_files, has_colmap_model
from pipeline_tools.gpu_lock import gpu_lock
from pipeline_tools.run_settings import colmap_signature, colmap_settings, COLMAP_DEFAULTS, add_colmap_arguments
from pipeline_tools.colmap_capabilities import resolve_device


def run_command(cmd, logger: ThesisLogger = None):
    """Menjalankan perintah terminal secara cross-platform dan mencatat log."""
    if logger:
        logger.log(f"[RUN CMD] {cmd}")
    else:
        print(f"[RUN CMD] {cmd}")

    tail = deque(maxlen=200)
    proc = subprocess.Popen(cmd, text=True, encoding='utf-8', errors='replace',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    try:
        for line in proc.stdout:
            clean = line.rstrip()
            tail.append(clean)
            if clean:
                if logger:
                    logger.log('[COLMAP] ' + clean)
                else:
                    print('[COLMAP] ' + clean, flush=True)
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        err_msg = f"Perintah gagal (exit code {proc.returncode}):\n" + '\n'.join(tail)
        if logger:
            logger.log(err_msg, level="error")
        raise RuntimeError(err_msg)
    return '\n'.join(tail)


def run_colmap_pipeline(scene_dir: str, 
                        camera_model: str = "OPENCV", 
                        colmap_exe: str = "colmap",
                        use_gpu: bool = False,
                        logger: ThesisLogger = None,
                        max_image_size: int = 3200, colmap_device=None,
                        colmap_matcher='exhaustive', colmap_max_features=8192,
                        colmap_sequential_overlap=10):
    """
    Menjalankan Structure-from-Motion (COLMAP) untuk menghasilkan pose kamera dan sparse point cloud.
    Sesuai format resmi 3DGS dan SuGaR.
    """
    scene_path = Path(scene_dir).resolve()
    images_dir = scene_path / "images"
    input_dir = scene_path / "input"
    if not 256 <= max_image_size <= 16384:
        raise ValueError('Ukuran COLMAP harus 256–16384 piksel.')
    settings = colmap_settings(dict(colmap_device=colmap_device or ('cuda' if use_gpu else 'cpu'),
        colmap_matcher=colmap_matcher, colmap_max_features=colmap_max_features,
        colmap_sequential_overlap=colmap_sequential_overlap))
    use_gpu = resolve_device(settings['colmap_device'], colmap_exe)
    # Keep databases separate for different features, backends and matching graphs.
    actual_settings = dict(settings, resolved_device='cuda' if use_gpu else 'cpu')
    suffix = '' if settings == COLMAP_DEFAULTS else '_' + hashlib.sha256(
        json.dumps(actual_settings, sort_keys=True).encode()).hexdigest()[:12]
    distorted_dir = scene_path / 'distorted' / f'maxsize_{max_image_size}{suffix}'
    sparse_dir = scene_path / "sparse" / "0"

    # Jika user meletakkan foto di images/, siapkan input/ agar kompatibel dengan standard 3DGS
    if not media_files(input_dir) and media_files(images_dir):
        if logger:
            logger.log("Menyiapkan folder input/ dari images/ untuk proses undistortion COLMAP...")
        input_dir.mkdir(parents=True, exist_ok=True)
        for img_file in media_files(images_dir):
            shutil.copy2(str(img_file), str(input_dir / img_file.name))

    if not media_files(input_dir):
        raise FileNotFoundError(f"Tidak ada file citra di {input_dir} atau {images_dir}")

    img_count = len(media_files(input_dir))
    if logger:
        logger.log(f"Memulai COLMAP untuk {img_count} citra pada scene: {scene_path}")
    device_message = (f"COLMAP SIFT: {'CUDA (GPU logis 0)' if use_gpu else 'CPU'}; "
                      f"matcher: {colmap_matcher}; fitur: {colmap_max_features}; batas: {max_image_size} px.")
    if settings['colmap_device'] == 'auto' and not use_gpu:
        device_message += ' Mode otomatis: build COLMAP dengan CUDA tidak terdeteksi.'
    if logger:
        logger.log(device_message)
    else:
        print(device_message, flush=True)

    # 1. Feature Extraction
    distorted_dir.mkdir(parents=True, exist_ok=True)
    (distorted_dir / "sparse").mkdir(parents=True, exist_ok=True)
    db_path = distorted_dir / "database.db"

    gpu_flag = 1 if use_gpu else 0
    extract_cmd = [colmap_exe, 'feature_extractor', '--database_path', str(db_path),
        '--image_path', str(input_dir), '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', camera_model, '--SiftExtraction.use_gpu', str(gpu_flag),
        '--SiftExtraction.max_image_size', str(max_image_size),
        '--SiftExtraction.max_num_features', str(colmap_max_features)]
    if use_gpu:
        # CUDA_VISIBLE_DEVICES in main() maps the selected physical GPU to logical 0.
        extract_cmd.extend(['--SiftExtraction.gpu_index', '0'])
    if logger:
        logger.set_parameters({'colmap_max_image_size': max_image_size, **actual_settings})
    if logger:
        logger.start_stage("COLMAP_Feature_Extraction")
    run_command(extract_cmd, logger)
    if logger:
        logger.end_stage("COLMAP_Feature_Extraction")

    # 2. Match all pairs, or neighbouring frames in filename order.
    match_cmd = [colmap_exe, colmap_matcher + '_matcher', '--database_path', str(db_path),
                 '--SiftMatching.use_gpu', str(gpu_flag)]
    if use_gpu:
        match_cmd.extend(['--SiftMatching.gpu_index', '0'])
    if colmap_matcher == 'sequential':
        match_cmd.extend(['--SequentialMatching.overlap', str(colmap_sequential_overlap)])
    if logger:
        logger.start_stage("COLMAP_Feature_Matching")
    run_command(match_cmd, logger)
    if logger:
        logger.end_stage("COLMAP_Feature_Matching")

    # 3. Mapper (Bundle Adjustment)
    distorted_sparse = distorted_dir / "sparse"
    mapper_cmd = [colmap_exe, 'mapper', '--database_path', str(db_path),
        '--image_path', str(input_dir), '--output_path', str(distorted_sparse),
        '--Mapper.ba_global_function_tolerance', '0.000001']
    if logger:
        logger.start_stage("COLMAP_Mapper")
    run_command(mapper_cmd, logger)
    if logger:
        logger.end_stage("COLMAP_Mapper")

    # 4. Image Undistortion (Mengubah ke kamera pinhole ideal yang dibutuhkan 3DGS)
    def registered_count(folder):
        path = folder / 'images.bin'
        if path.is_file() and path.stat().st_size >= 8:
            with path.open('rb') as stream:
                return struct.unpack('<Q', stream.read(8))[0]
        return 0

    subfolders = sorted((f for f in distorted_sparse.iterdir() if f.is_dir()), key=lambda p: p.name)
    if not subfolders or max(map(registered_count, subfolders)) < 2:
        raise RuntimeError("Mapper COLMAP tidak menghasilkan model dengan minimal 2 kamera!")
    undist_sparse_in = max(subfolders, key=registered_count)
    if logger:
        logger.set_parameters({'input_images': img_count, 'registered_images': registered_count(undist_sparse_in),
                               'selected_submodel': undist_sparse_in.name})

    undistorted_dir = distorted_dir / 'undistorted'
    undistorted_dir.mkdir(parents=True, exist_ok=True)
    undist_cmd = [colmap_exe, 'image_undistorter', '--image_path', str(input_dir),
        '--input_path', str(undist_sparse_in), '--output_path', str(undistorted_dir), '--output_type', 'COLMAP']
    if logger:
        logger.start_stage("COLMAP_Image_Undistortion")
    run_command(undist_cmd, logger)
    if logger:
        logger.end_stage("COLMAP_Image_Undistortion")

    # Pastikan file ada di sparse/0/
    sparse_root = undistorted_dir / 'sparse'
    sparse_0 = scene_path / 'sparse/0'
    sparse_0.mkdir(parents=True, exist_ok=True)
    generated_sparse = sparse_root / '0' if (sparse_root / '0').is_dir() else sparse_root
    for item in generated_sparse.iterdir():
        if item.is_file():
            shutil.copy2(item, sparse_0 / item.name)
    shutil.copytree(undistorted_dir / 'images', images_dir, dirs_exist_ok=True)
    # 3DGS otherwise keeps the stale point-cloud conversion from the previous SfM.
    (sparse_0 / 'points3D.ply').unlink(missing_ok=True)

    if not has_colmap_model(scene_path):
        raise RuntimeError('Output COLMAP tidak lengkap setelah undistorsi.')
    settings_path = scene_path / 'colmap_settings.json'
    temporary = settings_path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'max_image_size': max_image_size,
        **actual_settings,
        'work_dir': distorted_dir.relative_to(scene_path).as_posix(),
        'model_signature': colmap_signature(scene_path)}, indent=2), encoding='utf-8')
    temporary.replace(settings_path)

    if logger:
        logger.log(f"COLMAP selesai! Model sparse tersimpan di: {sparse_0}")


def main():
    parser = argparse.ArgumentParser(description="Wrapper otomatisasi COLMAP SfM untuk 3DGS & SuGaR.")
    parser.add_argument("--scene_path", "-s", type=str, required=True, help="Folder model patung (misal: datasets/Dewa/model_01).")
    parser.add_argument("--camera_model", default="OPENCV", type=str, help="Model kamera COLMAP (default: OPENCV).")
    parser.add_argument("--colmap_exe", default="colmap", type=str, help="Path executable colmap jika tidak ada di PATH.")
    parser.add_argument("--use_gpu", action="store_true", help="Aktifkan akselerasi GPU jika COLMAP dibangun dengan dukungan CUDA.")
    parser.add_argument("--gpu", type=int, default=0, help="Indeks GPU fisik untuk COLMAP CUDA.")
    parser.add_argument('--colmap_max_image_size', type=int, default=3200)
    add_colmap_arguments(parser)
    parser.add_argument("--no_gpu", action="store_true", help="Alias kompatibilitas; COLMAP sudah memakai CPU secara default.")
    parser.add_argument("--statue_class", type=str, default="Unknown", help="Nama kelas patung.")
    parser.add_argument("--model_id", type=str, default="model_01", help="ID model patung.")

    args = parser.parse_args()

    logger = ThesisLogger(statue_class=args.statue_class, model_id=args.model_id, method="COLMAP")

    try:
        # Serialize even a CUDA-enabled COLMAP with the training wrappers.
        with gpu_lock(args.gpu):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            logger.start_stage("COLMAP_Total")
            run_colmap_pipeline(
                scene_dir=args.scene_path,
                camera_model=args.camera_model,
                colmap_exe=args.colmap_exe,
                use_gpu=args.use_gpu and not args.no_gpu,
                logger=logger,
                max_image_size=args.colmap_max_image_size,
                colmap_device='cpu' if args.no_gpu else args.colmap_device,
                colmap_matcher=args.colmap_matcher, colmap_max_features=args.colmap_max_features,
                colmap_sequential_overlap=args.colmap_sequential_overlap
            )
        logger.end_stage("COLMAP_Total")
        logger.finalize(success=True)
    except Exception as e:
        logger.log(f"Error pada pipeline COLMAP: {str(e)}", level="error")
        logger.finalize(success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
