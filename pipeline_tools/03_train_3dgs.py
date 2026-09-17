import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline_tools.logger_utils import ThesisLogger


def train_vanilla_3dgs(scene_path: str,
                       output_dir: str,
                       iterations: int = 30000,
                       save_iterations: list = [7000, 30000],
                       checkpoint_iterations: list = [7000, 30000],
                       white_background: bool = False,
                       eval_split: bool = True,
                       gpu: int = 0,
                       logger: ThesisLogger = None):
    """
    Melatih model vanilla 3D Gaussian Splatting baseline.
    Menyimpan checkpoint pada 7.000 iterasi (untuk SuGaR) dan 30.000 iterasi (baseline final).
    """
    scene_path = Path(scene_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_script = PROJECT_ROOT / "gaussian_splatting" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Script pelatihan 3DGS tidak ditemukan di: {train_script}")

    save_iter_str = " ".join(str(i) for i in save_iterations)
    chkpt_iter_str = " ".join(str(i) for i in checkpoint_iterations)
    white_bg_flag = "-w" if white_background else ""
    eval_flag = "--eval" if eval_split else ""

    cmd = [
        sys.executable,
        str(train_script),
        "-s", str(scene_path),
        "-m", str(output_dir),
        "--iterations", str(iterations),
        "--save_iterations", *[str(i) for i in save_iterations],
        "--checkpoint_iterations", *[str(i) for i in checkpoint_iterations],
    ]
    if white_background:
        cmd.append("-w")
    if eval_split:
        cmd.append("--eval")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if logger:
        logger.log(f"Memulai pelatihan Vanilla 3DGS:")
        logger.log(f"Scene: {scene_path}")
        logger.log(f"Output: {output_dir}")
        logger.log(f"Total Iterasi: {iterations}, Checkpoint di: {checkpoint_iterations}")
        logger.start_stage("Pelatihan_3DGS")

    # Jalankan proses pelatihan
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    for line in proc.stdout:
        line_clean = line.strip()
        if line_clean:
            if logger and any(k in line_clean for k in ["Iteration:", "Training progress", "[", "Loss"]):
                logger.log(f"[3DGS] {line_clean}")
            elif not logger:
                print(f"[3DGS] {line_clean}")

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Pelatihan 3DGS gagal dengan kode error: {proc.returncode}")

    if logger:
        logger.end_stage("Pelatihan_3DGS")
        logger.log(f"Pelatihan 3DGS selesai dengan sukses! Model tersimpan di {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pelatihan baseline Vanilla 3D Gaussian Splatting.")
    parser.add_argument("--scene_path", "-s", type=str, required=True, help="Folder dataset (misal: datasets/Dewa/model_01).")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Folder tujuan output (default: outputs/3dgs/<Kelas>/<Model_ID>).")
    parser.add_argument("--statue_class", type=str, default="Dewa", help="Nama kelas patung.")
    parser.add_argument("--model_id", type=str, default="model_01", help="ID model patung.")
    parser.add_argument("--iterations", type=int, default=30000, help="Total iterasi pelatihan (default: 30000).")
    parser.add_argument("--white_background", "-w", action="store_true", help="Gunakan background putih.")
    parser.add_argument("--gpu", type=int, default=0, help="ID GPU (default: 0).")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "3dgs" / args.statue_class / args.model_id
    else:
        output_dir = Path(args.output_dir)

    logger = ThesisLogger(statue_class=args.statue_class, model_id=args.model_id, method="3DGS")
    logger.set_parameters({
        "iterations": args.iterations,
        "gpu": args.gpu,
        "white_background": args.white_background
    })

    try:
        train_vanilla_3dgs(
            scene_path=args.scene_path,
            output_dir=str(output_dir),
            iterations=args.iterations,
            white_background=args.white_background,
            gpu=args.gpu,
            logger=logger
        )
        logger.finalize(success=True)
    except Exception as e:
        logger.log(f"Error pada pelatihan 3DGS: {str(e)}", level="error")
        logger.finalize(success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
