import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from pipeline_tools.logger_utils import ThesisLogger


def extract_frames(video_path: str, 
                   output_dir: str, 
                   fps_extract: float = 2.0, 
                   min_sharpness: float = 80.0,
                   logger: ThesisLogger = None):
    """
    Mengekstrak frame dari video cadangan (MP4) ke dalam folder images/.
    Dilengkapi filter ketajaman (Laplacian variance) sesuai SOP Bab 6 untuk
    mengeliminasi frame yang buram (motion blur).
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) belum terinstall. Jalankan: pip install opencv-python")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Tidak dapat membuka file video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / orig_fps if orig_fps > 0 else 0

    if logger:
        logger.log(f"Membuka video: {video_path.name}")
        logger.log(f"FPS Asli: {orig_fps:.2f}, Total Frame: {total_frames}, Durasi: {video_duration:.1f} detik")
        logger.log(f"Target Ekstraksi: {fps_extract} frame/detik, Threshold Ketajaman: {min_sharpness}")

    step = max(1, int(round(orig_fps / fps_extract))) if fps_extract > 0 else 1

    frame_idx = 0
    saved_count = 0
    skipped_blur = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # Hitung ketajaman gambar via varians Laplacian
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            if sharpness >= min_sharpness:
                out_name = f"frame_{saved_count:05d}.jpg"
                out_file = output_dir / out_name
                cv2.imwrite(str(out_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_count += 1
            else:
                skipped_blur += 1

        frame_idx += 1

    cap.release()

    if logger:
        logger.log(f"Selesai ekstraksi: {saved_count} frame disimpan ke {output_dir}")
        logger.log(f"Frame diabaikan karena buram (blur): {skipped_blur} frame")

    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Ekstraksi frame dari video cadangan ke dataset images.")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path ke file video MP4.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Folder tujuan (contoh: datasets/Dewa/model_01/images).")
    parser.add_argument("--fps", type=float, default=2.0, help="Jumlah frame per detik yang diambil (default: 2.0).")
    parser.add_argument("--sharpness", type=float, default=80.0, help="Ambang batas minimal ketajaman (default: 80.0).")
    parser.add_argument("--statue_class", type=str, default="Unknown", help="Nama kelas patung.")
    parser.add_argument("--model_id", type=str, default="model_01", help="ID model patung.")

    args = parser.parse_args()

    logger = ThesisLogger(statue_class=args.statue_class, model_id=args.model_id, method="FrameExtraction")
    logger.start_stage("Ekstraksi_Frame")

    try:
        count = extract_frames(
            video_path=args.video,
            output_dir=args.output,
            fps_extract=args.fps,
            min_sharpness=args.sharpness,
            logger=logger
        )
        logger.end_stage("Ekstraksi_Frame")
        logger.set_parameters({
            "video_path": args.video,
            "extracted_frames": count,
            "fps": args.fps,
            "sharpness_threshold": args.sharpness
        })
        logger.finalize(success=True)
    except Exception as e:
        logger.log(f"Error ekstraksi frame: {str(e)}", level="error")
        logger.finalize(success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
