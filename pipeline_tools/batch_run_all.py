import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline_tools.logger_utils import ThesisLogger
from pipeline_tools.01_extract_frames import extract_frames
from pipeline_tools.02_run_colmap import run_colmap_pipeline
from pipeline_tools.03_train_3dgs import train_vanilla_3dgs
from pipeline_tools.04_train_sugar import train_sugar_pipeline


CLASSES = [
    "Dewa",
    "Dewi",
    "Mitologi",
    "Penabuh",
    "Pengapit",
    "Punakawan",
    "Raksasa",
    "Wanara"
]


def process_single_statue(scene_dir: Path, 
                           statue_class: str, 
                           model_id: str,
                           skip_colmap: bool = False,
                           skip_3dgs: bool = False,
                           skip_sugar: bool = False,
                           refinement_time: str = "short",
                           gpu: int = 0):
    """Memproses satu patung secara lengkap dari awal hingga akhir."""
    print(f"\n=======================================================")
    print(f"MEMULAI PROSES: {statue_class} - {model_id}")
    print(f"Direktori: {scene_dir}")
    print(f"=======================================================")

    images_dir = scene_dir / "images"
    video_dir = scene_dir / "raw_video"
    sparse_dir = scene_dir / "sparse" / "0"

    # 1. Cek Citra atau Video Cadangan
    has_images = images_dir.exists() and len([f for f in images_dir.iterdir() if f.is_file()]) > 0
    if not has_images:
        # Coba cek video cadangan
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4"))
        if videos:
            print(f"Folder images kosong. Mengekstrak frame dari video cadangan: {videos[0].name}...")
            frame_logger = ThesisLogger(statue_class=statue_class, model_id=model_id, method="FrameExtraction")
            extract_frames(str(videos[0]), str(images_dir), fps_extract=2.0, min_sharpness=80.0, logger=frame_logger)
            frame_logger.finalize(success=True)
        else:
            print(f"SKIP {statue_class}/{model_id}: Folder images/ dan raw_video/ tidak memuat data.")
            return

    # 2. COLMAP (SfM)
    if not skip_colmap and not sparse_dir.exists():
        print(f"\n--- Menjalankan COLMAP untuk {statue_class}/{model_id} ---")
        colmap_logger = ThesisLogger(statue_class=statue_class, model_id=model_id, method="COLMAP")
        try:
            run_colmap_pipeline(str(scene_dir), logger=colmap_logger)
            colmap_logger.finalize(success=True)
        except Exception as e:
            print(f"COLMAP Gagal untuk {statue_class}/{model_id}: {str(e)}")
            colmap_logger.finalize(success=False)
            return
    else:
        print(f"COLMAP dilewati (sparse/0/ sudah ada atau flag --skip_colmap aktif).")

    # 3. Vanilla 3DGS
    out_3dgs_dir = PROJECT_ROOT / "outputs" / "3dgs" / statue_class / model_id
    chkpt_7k = out_3dgs_dir / "point_cloud" / "iteration_7000"
    if not skip_3dgs:
        print(f"\n--- Melatih Vanilla 3DGS untuk {statue_class}/{model_id} ---")
        dgs_logger = ThesisLogger(statue_class=statue_class, model_id=model_id, method="3DGS")
        try:
            train_vanilla_3dgs(
                scene_path=str(scene_dir),
                output_dir=str(out_3dgs_dir),
                iterations=30000,
                save_iterations=[7000, 30000],
                checkpoint_iterations=[7000, 30000],
                gpu=gpu,
                logger=dgs_logger
            )
            dgs_logger.finalize(success=True)
        except Exception as e:
            print(f"Pelatihan 3DGS Gagal: {str(e)}")
            dgs_logger.finalize(success=False)
            return
    else:
        print(f"Pelatihan 3DGS dilewati.")

    # 4. SuGaR
    out_sugar_dir = PROJECT_ROOT / "outputs" / "sugar" / statue_class / model_id
    if not skip_sugar:
        print(f"\n--- Melatih SuGaR untuk {statue_class}/{model_id} ---")
        sugar_logger = ThesisLogger(statue_class=statue_class, model_id=model_id, method="SuGaR")
        try:
            train_sugar_pipeline(
                scene_path=str(scene_dir),
                checkpoint_3dgs=str(out_3dgs_dir),
                output_base_dir=str(out_sugar_dir),
                regularization_type="dn_consistency",
                refinement_time=refinement_time,
                n_vertices=250000,
                gpu=gpu,
                logger=sugar_logger
            )
            sugar_logger.finalize(success=True)
        except Exception as e:
            print(f"Pelatihan SuGaR Gagal: {str(e)}")
            sugar_logger.finalize(success=False)
            return
    else:
        print(f"Pelatihan SuGaR dilewati.")

    print(f"\nSELESAI MEMPROSES: {statue_class} - {model_id}!")


def main():
    parser = argparse.ArgumentParser(description="Master Batch Automation Runner untuk seluruh 16 patung Bali.")
    parser.add_argument("--class_name", "-c", type=str, default=None, choices=CLASSES, help="Nama kelas patung tertentu (default: semua kelas).")
    parser.add_argument("--model_id", "-m", type=str, default=None, help="ID model tertentu (misal: model_01).")
    parser.add_argument("--skip_colmap", action="store_true", help="Lewati tahap COLMAP jika sudah pernah dijalankan.")
    parser.add_argument("--skip_3dgs", action="store_true", help="Lewati tahap 3DGS jika sudah selesai.")
    parser.add_argument("--skip_sugar", action="store_true", help="Lewati tahap SuGaR.")
    parser.add_argument("--refinement_time", type=str, default="short", choices=["short", "medium", "long"], help="Waktu refine SuGaR (default: short).")
    parser.add_argument("--gpu", type=int, default=0, help="ID GPU (default: 0).")

    args = parser.parse_args()
    datasets_root = PROJECT_ROOT / "datasets"

    classes_to_run = [args.class_name] if args.class_name else CLASSES

    for cls in classes_to_run:
        class_folder = datasets_root / cls
        if not class_folder.exists():
            continue

        models_in_class = [args.model_id] if args.model_id else [f.name for f in class_folder.iterdir() if f.is_dir()]
        for mdl in sorted(models_in_class):
            model_folder = class_folder / mdl
            if model_folder.is_dir():
                process_single_statue(
                    scene_dir=model_folder,
                    statue_class=cls,
                    model_id=mdl,
                    skip_colmap=args.skip_colmap,
                    skip_3dgs=args.skip_3dgs,
                    skip_sugar=args.skip_sugar,
                    refinement_time=args.refinement_time,
                    gpu=args.gpu
                )


if __name__ == "__main__":
    main()
