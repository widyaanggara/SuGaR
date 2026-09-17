import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline_tools.logger_utils import ThesisLogger


def run_command(cmd: str, logger: ThesisLogger = None):
    """Menjalankan perintah terminal secara cross-platform dan mencatat log."""
    if logger:
        logger.log(f"[RUN CMD] {cmd}")
    else:
        print(f"[RUN CMD] {cmd}")

    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.returncode != 0:
        err_msg = f"Perintah gagal (exit code {res.returncode}):\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        if logger:
            logger.log(err_msg, level="error")
        raise RuntimeError(err_msg)
    return res.stdout


def run_colmap_pipeline(scene_dir: str, 
                        camera_model: str = "OPENCV", 
                        colmap_exe: str = "colmap",
                        use_gpu: bool = True,
                        logger: ThesisLogger = None):
    """
    Menjalankan Structure-from-Motion (COLMAP) untuk menghasilkan pose kamera dan sparse point cloud.
    Sesuai format resmi 3DGS dan SuGaR.
    """
    scene_path = Path(scene_dir).resolve()
    images_dir = scene_path / "images"
    input_dir = scene_path / "input"
    distorted_dir = scene_path / "distorted"
    sparse_dir = scene_path / "sparse" / "0"

    # Jika user meletakkan foto di images/, siapkan input/ agar kompatibel dengan standard 3DGS
    if images_dir.exists() and not input_dir.exists():
        if logger:
            logger.log("Menyiapkan folder input/ dari images/ untuk proses undistortion COLMAP...")
        input_dir.mkdir(parents=True, exist_ok=True)
        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.dng']:
                shutil.copy2(str(img_file), str(input_dir / img_file.name))

    if not input_dir.exists() or len(list(input_dir.iterdir())) == 0:
        raise FileNotFoundError(f"Tidak ada file citra di {input_dir} atau {images_dir}")

    img_count = len([f for f in input_dir.iterdir() if f.is_file()])
    if logger:
        logger.log(f"Memulai COLMAP untuk {img_count} citra pada scene: {scene_path}")

    # 1. Feature Extraction
    distorted_dir.mkdir(parents=True, exist_ok=True)
    (distorted_dir / "sparse").mkdir(parents=True, exist_ok=True)
    db_path = distorted_dir / "database.db"

    gpu_flag = 1 if use_gpu else 0
    extract_cmd = (
        f'"{colmap_exe}" feature_extractor '
        f'--database_path "{db_path}" '
        f'--image_path "{input_dir}" '
        f'--ImageReader.single_camera 1 '
        f'--ImageReader.camera_model {camera_model} '
        f'--SiftExtraction.use_gpu {gpu_flag}'
    )
    if logger:
        logger.start_stage("COLMAP_Feature_Extraction")
    run_command(extract_cmd, logger)
    if logger:
        logger.end_stage("COLMAP_Feature_Extraction")

    # 2. Exhaustive Matcher
    match_cmd = (
        f'"{colmap_exe}" exhaustive_matcher '
        f'--database_path "{db_path}" '
        f'--SiftMatching.use_gpu {gpu_flag}'
    )
    if logger:
        logger.start_stage("COLMAP_Feature_Matching")
    run_command(match_cmd, logger)
    if logger:
        logger.end_stage("COLMAP_Feature_Matching")

    # 3. Mapper (Bundle Adjustment)
    distorted_sparse = distorted_dir / "sparse"
    mapper_cmd = (
        f'"{colmap_exe}" mapper '
        f'--database_path "{db_path}" '
        f'--image_path "{input_dir}" '
        f'--output_path "{distorted_sparse}" '
        f'--Mapper.ba_global_function_tolerance=0.000001'
    )
    if logger:
        logger.start_stage("COLMAP_Mapper")
    run_command(mapper_cmd, logger)
    if logger:
        logger.end_stage("COLMAP_Mapper")

    # 4. Image Undistortion (Mengubah ke kamera pinhole ideal yang dibutuhkan 3DGS)
    undist_sparse_in = distorted_sparse / "0"
    if not undist_sparse_in.exists():
        # Jika mapper menghasilkan folder tanpa subfolder '0'
        subfolders = [f for f in distorted_sparse.iterdir() if f.is_dir()]
        if subfolders:
            undist_sparse_in = subfolders[0]
        else:
            raise RuntimeError("Mapper COLMAP tidak menghasilkan model rekonstruksi 3D!")

    undist_cmd = (
        f'"{colmap_exe}" image_undistorter '
        f'--image_path "{input_dir}" '
        f'--input_path "{undist_sparse_in}" '
        f'--output_path "{scene_path}" '
        f'--output_type COLMAP'
    )
    if logger:
        logger.start_stage("COLMAP_Image_Undistortion")
    run_command(undist_cmd, logger)
    if logger:
        logger.end_stage("COLMAP_Image_Undistortion")

    # Pastikan file ada di sparse/0/
    sparse_root = scene_path / "sparse"
    sparse_0 = sparse_root / "0"
    sparse_0.mkdir(parents=True, exist_ok=True)
    for item in sparse_root.iterdir():
        if item.is_file():
            shutil.move(str(item), str(sparse_0 / item.name))

    if logger:
        logger.log(f"COLMAP selesai! Model sparse tersimpan di: {sparse_0}")


def main():
    parser = argparse.ArgumentParser(description="Wrapper otomatisasi COLMAP SfM untuk 3DGS & SuGaR.")
    parser.add_argument("--scene_path", "-s", type=str, required=True, help="Folder model patung (misal: datasets/Dewa/model_01).")
    parser.add_argument("--camera_model", default="OPENCV", type=str, help="Model kamera COLMAP (default: OPENCV).")
    parser.add_argument("--colmap_exe", default="colmap", type=str, help="Path executable colmap jika tidak ada di PATH.")
    parser.add_argument("--no_gpu", action="store_true", help="Nonaktifkan akselerasi GPU pada COLMAP.")
    parser.add_argument("--statue_class", type=str, default="Unknown", help="Nama kelas patung.")
    parser.add_argument("--model_id", type=str, default="model_01", help="ID model patung.")

    args = parser.parse_args()

    logger = ThesisLogger(statue_class=args.statue_class, model_id=args.model_id, method="COLMAP")
    logger.start_stage("COLMAP_Total")

    try:
        run_colmap_pipeline(
            scene_dir=args.scene_path,
            camera_model=args.camera_model,
            colmap_exe=args.colmap_exe,
            use_gpu=not args.no_gpu,
            logger=logger
        )
        logger.end_stage("COLMAP_Total")
        logger.finalize(success=True)
    except Exception as e:
        logger.log(f"Error pada pipeline COLMAP: {str(e)}", level="error")
        logger.finalize(success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
