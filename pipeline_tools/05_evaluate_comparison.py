import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline_tools.logger_utils import ThesisLogger

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def align_point_clouds_icp(source_pcd, target_pcd, max_correspondence_distance: float = 0.05):
    """
    Menyelaraskan posisi dan rotasi source point cloud ke target LiDAR
    menggunakan algoritma Iterative Closest Point (ICP) di Open3D.
    """
    if not O3D_AVAILABLE:
        return source_pcd, np.identity(4), 0.0

    # 1. Samakan pusat massa (Centroid Alignment)
    source_center = source_pcd.get_center()
    target_center = target_pcd.get_center()
    source_pcd.translate(target_center - source_center)

    # 2. Estimasi normal jika belum ada
    if not source_pcd.has_normals():
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 3. Fine ICP (Point-to-Plane untuk akurasi permukaan terbaik)
    trans_init = np.identity(4)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    # Terapkan transformasi hasil ICP
    source_aligned = source_pcd.transform(reg_p2l.transformation)
    return source_aligned, reg_p2l.transformation, reg_p2l.inlier_rmse


def compute_chamfer_distance(source_pcd_path: str, 
                             target_pcd_path: str, 
                             max_samples: int = 100000,
                             scale_factor: float = 1.0,
                             apply_icp: bool = True,
                             save_aligned_path: str = None):
    """
    Menghitung Chamfer Distance dan akurasi/kelengkapan geometri (cm)
    terhadap Ground Truth LiDAR dengan penyelarasan ICP otomatis.
    """
    if not O3D_AVAILABLE:
        raise ImportError("Pustaka Open3D belum terinstall. Jalankan: pip install open3d")

    # Load data
    src = o3d.io.read_point_cloud(str(source_pcd_path))
    tgt = o3d.io.read_point_cloud(str(target_pcd_path))

    if len(src.points) == 0:
        # Coba load sebagai mesh jika format .obj atau .ply segitiga
        mesh = o3d.io.read_triangle_mesh(str(source_pcd_path))
        if len(mesh.vertices) > 0:
            src = mesh.sample_points_uniformly(number_of_points=max_samples)

    if len(tgt.points) == 0:
        mesh_tgt = o3d.io.read_triangle_mesh(str(target_pcd_path))
        if len(mesh_tgt.vertices) > 0:
            tgt = mesh_tgt.sample_points_uniformly(number_of_points=max_samples)

    if len(src.points) == 0 or len(tgt.points) == 0:
        raise ValueError("Point cloud sumber atau target kosong!")

    # 1. Terapkan faktor skala jika diberikan
    if scale_factor != 1.0:
        src.scale(scale_factor, center=src.get_center())

    # 2. Lakukan registrasi ICP otomatis jika diaktifkan
    icp_rmse_cm = None
    if apply_icp:
        src, transformation, rmse = align_point_clouds_icp(src, tgt, max_correspondence_distance=0.08)
        icp_rmse_cm = round(float(rmse * 100.0), 4)

    # 3. Simpan model yang sudah selaras (untuk visualisasi bab 4 skripsi)
    if save_aligned_path:
        Path(save_aligned_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(save_aligned_path), src)

    # Subsampling untuk efisiensi komputasi jika terlalu padat
    if len(src.points) > max_samples:
        src = src.random_down_sample(max_samples / len(src.points))
    if len(tgt.points) > max_samples:
        tgt = tgt.random_down_sample(max_samples / len(tgt.points))

    # Jarak src -> tgt (Accuracy)
    dists_src_to_tgt = np.asarray(src.compute_point_cloud_distance(tgt))
    # Jarak tgt -> src (Completeness)
    dists_tgt_to_src = np.asarray(tgt.compute_point_cloud_distance(src))

    mean_src_to_tgt = np.mean(dists_src_to_tgt)
    mean_tgt_to_src = np.mean(dists_tgt_to_src)

    # Chamfer distance (meter ke cm)
    chamfer_dist_m = 0.5 * (mean_src_to_tgt + mean_tgt_to_src)
    chamfer_dist_cm = chamfer_dist_m * 100.0

    # Akurasi dan kelengkapan pada ambang 2 cm (0.02 m) dan 5 cm (0.05 m) sesuai SOP
    acc_2cm = np.mean(dists_src_to_tgt < 0.02) * 100.0
    comp_2cm = np.mean(dists_tgt_to_src < 0.02) * 100.0
    acc_5cm = np.mean(dists_src_to_tgt < 0.05) * 100.0
    comp_5cm = np.mean(dists_tgt_to_src < 0.05) * 100.0

    return {
        "chamfer_distance_cm": round(float(chamfer_dist_cm), 4),
        "mean_accuracy_cm": round(float(mean_src_to_tgt * 100.0), 4),
        "mean_completeness_cm": round(float(mean_tgt_to_src * 100.0), 4),
        "accuracy_at_2cm_pct": round(float(acc_2cm), 2),
        "completeness_at_2cm_pct": round(float(comp_2cm), 2),
        "accuracy_at_5cm_pct": round(float(acc_5cm), 2),
        "completeness_at_5cm_pct": round(float(comp_5cm), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluasi kuantitatif komparasi 3DGS vs SuGaR untuk Skripsi.")
    parser.add_argument("--statue_class", "-c", type=str, required=True, help="Nama kelas patung (misal: Dewa).")
    parser.add_argument("--model_id", "-m", type=str, required=True, help="ID model patung (misal: model_01).")
    parser.add_argument("--method", type=str, default="SuGaR", choices=["3DGS", "SuGaR"], help="Metode yang dievaluasi.")
    parser.add_argument("--model_path", "-p", type=str, required=True, help="Path ke file model (.ply atau .obj).")
    parser.add_argument("--gt_lidar", "-g", type=str, default=None, help="Path ke point cloud LiDAR ground truth (.ply).")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Faktor pengali skala metrik (default: 1.0).")
    parser.add_argument("--no_icp", action="store_true", help="Nonaktifkan registrasi ICP otomatis.")
    parser.add_argument("--psnr", type=float, default=None, help="Nilai PSNR hasil evaluasi visual (dB).")
    parser.add_argument("--ssim", type=float, default=None, help="Nilai SSIM (0-1).")
    parser.add_argument("--lpips", type=float, default=None, help="Nilai LPIPS.")
    parser.add_argument("--num_elements", type=int, default=None, help="Jumlah partikel Gaussians atau vertex mesh.")

    args = parser.parse_args()

    logger = ThesisLogger(statue_class=args.statue_class, model_id=args.model_id, method=args.method)
    logger.log(f"Memulai evaluasi metrik untuk {args.method} - {args.statue_class}/{args.model_id}")

    chamfer_cm = None
    geo_metrics = {}

    # 1. Evaluasi Geometri jika ada Ground Truth LiDAR
    if args.gt_lidar and os.path.exists(args.gt_lidar):
        logger.start_stage("Evaluasi_Geometri_LiDAR")
        aligned_out = PROJECT_ROOT / "outputs" / "logs" / "evaluation_reports" / f"{args.statue_class}_{args.model_id}_{args.method}_aligned.ply"
        try:
            logger.log("Menjalankan penyelarasan ICP otomatis terhadap LiDAR...")
            geo_metrics = compute_chamfer_distance(
                source_pcd_path=args.model_path,
                target_pcd_path=args.gt_lidar,
                scale_factor=args.scale_factor,
                apply_icp=not args.no_icp,
                save_aligned_path=str(aligned_out)
            )
            chamfer_cm = geo_metrics["chamfer_distance_cm"]
            logger.log(f"Model terselaraskan disimpan di: {aligned_out}")
            logger.log(f"Chamfer Distance: {chamfer_cm:.4f} cm")
            logger.log(f"Akurasi (< 2cm): {geo_metrics['accuracy_at_2cm_pct']}%, Kelengkapan (< 2cm): {geo_metrics['completeness_at_2cm_pct']}%")
            logger.log(f"Akurasi (< 5cm): {geo_metrics['accuracy_at_5cm_pct']}%, Kelengkapan (< 5cm): {geo_metrics['completeness_at_5cm_pct']}%")
        except Exception as e:
            logger.log(f"Gagal menghitung Chamfer Distance: {str(e)}", level="warning")
        logger.end_stage("Evaluasi_Geometri_LiDAR")

    # 2. Rekam seluruh metrik ke file log skripsi
    logger.set_metrics(
        psnr=args.psnr,
        ssim=args.ssim,
        lpips=args.lpips,
        chamfer_distance_cm=chamfer_cm,
        num_elements=args.num_elements
    )
    logger.set_parameters({
        "model_path": args.model_path,
        "gt_lidar_path": args.gt_lidar,
        "geometric_metrics_detail": geo_metrics
    })

    logger.finalize(success=True)
    logger.log("Evaluasi selesai! Tabel rekapitulasi CSV & Markdown berhasil diperbarui.")


if __name__ == "__main__":
    main()
