import os
import sys
import json
import argparse
import math
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline_tools.logger_utils import ThesisLogger
from pipeline_tools.external_methods import output_base
from pipeline_tools.geometry_metrics import metrics_from_distances
from pipeline_tools.dataset_registry import file_signature
from pipeline_tools.geometry_config import LIDAR_UNITS

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False

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
    centroid_transform = np.identity(4)
    centroid_transform[:3, 3] = target_center - source_center
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

    if reg_p2l.fitness < 0.1 or not math.isfinite(reg_p2l.inlier_rmse):
        raise ValueError(f"Registrasi ICP tidak memadai (fitness={reg_p2l.fitness:.4f}). Periksa skala, orientasi, dan area objek.")
    # Terapkan transformasi hasil ICP
    source_aligned = source_pcd.transform(reg_p2l.transformation)
    return source_aligned, reg_p2l.transformation @ centroid_transform, reg_p2l.inlier_rmse


def compute_chamfer_distance(source_pcd_path: str, 
                             target_pcd_path: str, 
                             max_samples: int = 100000,
                             scale_factor: float = 1.0,
                             apply_icp: bool = True,
                             save_aligned_path: str = None,
                             auto_align: bool = False,
                             lidar_unit: str = 'm', log=None, allow_weak_alignment: bool = False):
    """
    Menghitung Chamfer Distance dan akurasi/kelengkapan geometri (cm)
    terhadap Ground Truth LiDAR dengan penyelarasan ICP otomatis.
    """
    log = log or (lambda message: None)
    if not O3D_AVAILABLE:
        raise ImportError("Pustaka Open3D belum terinstall. Jalankan: pip install open3d")
    if (not auto_align and (scale_factor is None or not math.isfinite(scale_factor) or scale_factor <= 0)) or max_samples < 1:
        raise ValueError('Skala harus finite dan > 0; max_samples harus positif.')
    if lidar_unit not in LIDAR_UNITS:
        raise ValueError('Satuan LiDAR harus m, cm, atau mm.')
    if allow_weak_alignment and not auto_align:
        raise ValueError('Opsi alignment lemah hanya berlaku untuk mode otomatis.')
    if auto_align and not apply_icp:
        raise ValueError('Mode otomatis harus mengaktifkan penyelarasan.')

    # Fixed sampling seed makes repeated runs comparable. Meshes are sampled on
    # triangle surfaces, rather than comparing only their vertices.
    o3d.utility.random.seed(0)
    log(f'[INPUT 1/2] Membaca rekonstruksi: {source_pcd_path}')
    mesh = o3d.io.read_triangle_mesh(str(source_pcd_path))
    log('[INPUT] Representasi: ' + ('mesh; sampling merata pada permukaan.' if mesh.has_triangles()
        else 'point cloud; memakai koordinat titik (untuk 3DGS: pusat Gaussian).'))
    src = (mesh.sample_points_uniformly(number_of_points=max_samples) if mesh.has_triangles()
           else o3d.io.read_point_cloud(str(source_pcd_path)))
    log(f'[INPUT 2/2] Membaca ground truth LiDAR: {target_pcd_path}')
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
        raise ValueError(f'Data kosong atau gagal dibaca: rekonstruksi={len(src.points)} titik; '
                         f'LiDAR={len(tgt.points)} titik. Periksa file dan format input di atas.')

    if not np.isfinite(np.asarray(src.points)).all() or not np.isfinite(np.asarray(tgt.points)).all():
        raise ValueError('Koordinat mengandung NaN atau infinity.')
    # Convert GT around the origin, preserving absolute coordinates consistently.
    tgt.scale(LIDAR_UNITS[lidar_unit], center=(0, 0, 0))
    log(f'[INPUT] Titik sebelum pembatasan sampel: rekonstruksi={len(src.points):,}, LiDAR={len(tgt.points):,}.')
    for label, cloud in [('rekonstruksi (satuan asli)', src), ('LiDAR (meter)', tgt)]:
        extent = np.ptp(np.asarray(cloud.points), axis=0)
        log(f'[UKURAN] Rentang XYZ {label}: ' + ' x '.join(f'{v:.6g}' for v in extent)
            + '; bounding box termasuk noise, bukan ukuran fisik patung.')
    log(f'[SATUAN] LiDAR={lidar_unit}; pengali ke meter={LIDAR_UNITS[lidar_unit]:g}. '
        'Satuan berasal dari pilihan pengguna, tidak dideteksi dari PLY.')
    # Fixed bounded samples for registration AND metrics; keep full object coverage.
    sampling = np.random.default_rng(0)
    if len(src.points) > max_samples:
        src = src.select_by_index(sampling.choice(len(src.points), max_samples, replace=False).tolist())
    if len(tgt.points) > max_samples:
        tgt = tgt.select_by_index(sampling.choice(len(tgt.points), max_samples, replace=False).tolist())
    log(f'[SAMPLING] Evaluasi: rekonstruksi={len(src.points):,}, LiDAR={len(tgt.points):,}; seed=0.')

    # 1. Terapkan faktor skala jika diberikan
    scale_transform = np.eye(4)
    if not auto_align and scale_factor != 1.0:
        center = src.get_center()
        scale_transform[:3, :3] *= scale_factor
        scale_transform[:3, 3] = center * (1 - scale_factor)
        src.scale(scale_factor, center=src.get_center())

    # 2. Lakukan registrasi ICP otomatis jika diaktifkan
    icp_rmse_cm = None
    alignment_details = {}
    if auto_align:
        from pipeline_tools.geometry_alignment import automatic_similarity_alignment, AlignmentRejected
        try:
            transformation, alignment_details = automatic_similarity_alignment(
                np.asarray(src.points), np.asarray(tgt.points), log=log,
                allow_weak=allow_weak_alignment)
        except AlignmentRejected as error:
            log(f'[ALIGN DITOLAK] {error}')
            if save_aligned_path:
                import tempfile
                parent = Path(save_aligned_path).parent / 'alignment_failures'
                parent.mkdir(parents=True, exist_ok=True)
                folder = Path(tempfile.mkdtemp(prefix='attempt_', dir=parent))
                diagnostics = dict(error.diagnostics,
                    source_path=str(source_pcd_path), lidar_path=str(target_pcd_path),
                    lidar_unit=lidar_unit, sampling_seed=0,
                    source_evaluation_points=len(src.points), target_evaluation_points=len(tgt.points))
                src.transform(np.asarray(diagnostics['alignment_transform']))
                for name, cloud in [('rejected_candidate.ply', src), ('ground_truth_meters.ply', tgt)]:
                    if not o3d.io.write_point_cloud(str(folder / name), cloud):
                        raise OSError(f'Gagal menyimpan diagnostik: {folder / name}') from error
                (folder / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
                log(f'[DIAGNOSTIK] Folder percobaan gagal: {folder.resolve()}')
                log('[DIAGNOSTIK] Buka rejected_candidate.ply dan ground_truth_meters.ply bersama '
                    'di CloudCompare. diagnostics.json berisi angka dan transformasi kandidat; bukan skor evaluasi.')
                raise AlignmentRejected(f'{error} Diagnostik kandidat yang ditolak: {folder}', diagnostics) from error
            raise
        src.transform(transformation)
        scale_factor = alignment_details['scale_factor_to_meters']
    elif apply_icp:
        log(f'[ALIGN MANUAL] Faktor skala={scale_factor:g}; memperbaiki posisi/rotasi dengan ICP.')
        src, transformation, rmse = align_point_clouds_icp(src, tgt, max_correspondence_distance=0.08)
        transformation = transformation @ scale_transform
        icp_rmse_cm = round(float(rmse * 100.0), 4)

    # 3. Simpan model yang sudah selaras (untuk visualisasi bab 4 skripsi)
    if save_aligned_path:
        log(f'[SIMPAN] Menulis overlay hasil alignment: {save_aligned_path}')
        Path(save_aligned_path).parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(save_aligned_path), src):
            raise OSError('Gagal menyimpan rekonstruksi yang telah diselaraskan.')
        if not o3d.io.write_point_cloud(str(Path(save_aligned_path).with_name('ground_truth_meters.ply')), tgt):
            raise OSError('Gagal menyimpan LiDAR acuan dalam meter.')

    # Jarak src -> tgt (Accuracy)
    log('[METRIK] Alignment LEMAH (tidak lolos overlap); skor dihitung dengan penanda alignment_passed=false...'
        if alignment_details.get('alignment_passed') is False
        else '[METRIK] Alignment diterima; menghitung jarak tetangga terdekat dua arah...'
        if auto_align else '[METRIK] Menghitung jarak tetangga terdekat dua arah; periksa overlay alignment manual.')
    dists_src_to_tgt = np.asarray(src.compute_point_cloud_distance(tgt))
    # Jarak tgt -> src (Completeness)
    dists_tgt_to_src = np.asarray(tgt.compute_point_cloud_distance(src))

    return {
        **metrics_from_distances(dists_src_to_tgt, dists_tgt_to_src),
        "sampling_seed": 0,
        "max_samples": max_samples,
        "icp_inlier_rmse_cm": icp_rmse_cm,
        "alignment_transform": (transformation if apply_icp else scale_transform).tolist(),
        "scale_factor_to_meters": scale_factor,
        "lidar_unit": lidar_unit,
        "lidar_unit_to_meters": LIDAR_UNITS[lidar_unit],
        **alignment_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluasi kuantitatif komparasi 3DGS vs SuGaR untuk Skripsi.")
    parser.add_argument("--statue_class", "-c", type=str, required=True, help="Nama kelas patung (misal: Dewa).")
    parser.add_argument("--model_id", "-m", type=str, required=True, help="ID model patung (misal: model_01).")
    parser.add_argument("--method", type=str, default="SuGaR", choices=["3DGS", "SuGaR", "2DGS", "RaDe-GS"], help="Metode yang dievaluasi.")
    parser.add_argument("--model_path", "-p", type=str, required=True, help="Path ke file model (.ply atau .obj).")
    parser.add_argument("--gt_lidar", "-g", type=str, default=None, help="Path ke point cloud LiDAR ground truth (.ply).")
    parser.add_argument("--scale_factor", type=float, default=None,
                        help="Faktor skala manual; tidak diperlukan pada --auto_align.")
    parser.add_argument('--auto_align', action='store_true', help='Cari skala, rotasi, dan posisi otomatis (Sim(3)).')
    parser.add_argument('--lidar_unit', choices=tuple(LIDAR_UNITS), default='m',
                        help='Satuan koordinat file LiDAR; CLI lama menganggap meter.')
    parser.add_argument('--allow_weak_alignment', action='store_true',
                        help='Tetap hitung skor walau overlap registrasi <35%; hasil ditandai alignment_passed=false.')
    parser.add_argument("--no_icp", action="store_true", help="Nonaktifkan registrasi ICP otomatis.")
    parser.add_argument("--psnr", type=float, default=None, help="Nilai PSNR hasil evaluasi visual (dB).")
    parser.add_argument("--ssim", type=float, default=None, help="Nilai SSIM (0-1).")
    parser.add_argument("--lpips", type=float, default=None, help="Nilai LPIPS.")
    parser.add_argument("--num_elements", type=int, default=None, help="Jumlah partikel Gaussians atau vertex mesh.")

    parser.add_argument("--experiment", choices=("baseline", "scaffold_anchor"), default="baseline")
    args = parser.parse_args()

    method_label = args.method + ("_scaffold_anchor" if args.experiment == "scaffold_anchor" else "")
    logger = ThesisLogger(statue_class=args.statue_class, model_id=args.model_id, method=method_label)
    logger.log(f"Memulai evaluasi metrik untuk {args.method} - {args.statue_class}/{args.model_id}")
    logger.log(f'[KONFIGURASI] Eksperimen={args.experiment}; metode={args.method}; Python={sys.executable}')
    logger.log(f'[KONFIGURASI] Alignment={"otomatis Sim(3)" if args.auto_align else "manual"}; '
               f'LiDAR={args.lidar_unit}; ambang metrik=2 cm dan 5 cm.')

    model_path = Path(args.model_path)
    if not model_path.is_file():
        logger.log(f"File model tidak ditemukan: {model_path}", level="error")
        logger.finalize(success=False)
        sys.exit(1)

    if args.gt_lidar and not Path(args.gt_lidar).is_file():
        logger.log(f"File LiDAR tidak ditemukan: {args.gt_lidar}", level="error")
        logger.finalize(success=False)
        sys.exit(1)

    has_visual_metrics = any(value is not None for value in (args.psnr, args.ssim, args.lpips))
    if not args.gt_lidar and not has_visual_metrics:
        logger.log("Tidak ada evaluasi yang diminta. Berikan --gt_lidar atau setidaknya satu metrik visual.", level="error")
        logger.finalize(success=False)
        sys.exit(2)

    if args.auto_align and (args.no_icp or args.scale_factor is not None):
        logger.log('--auto_align tidak boleh digabung --no_icp atau --scale_factor.', level='error')
        logger.finalize(success=False)
        sys.exit(2)
    if args.allow_weak_alignment and not args.auto_align:
        logger.log('--allow_weak_alignment hanya berlaku bersama --auto_align.', level='error')
        logger.finalize(success=False)
        sys.exit(2)
    if args.gt_lidar and not args.auto_align and args.scale_factor is None:
        logger.log("Pilih --auto_align atau berikan --scale_factor hasil kalibrasi manual.", level="error")
        logger.finalize(success=False)
        sys.exit(2)
    if args.gt_lidar and not args.auto_align and (not math.isfinite(args.scale_factor) or args.scale_factor <= 0):
        logger.log("--scale_factor harus berupa angka finite yang lebih besar dari 0.", level="error")
        logger.finalize(success=False)
        sys.exit(2)

    chamfer_cm = None
    geo_metrics = {}

    # 1. Evaluasi Geometri jika ada Ground Truth LiDAR
    if args.gt_lidar:
        logger.start_stage("Evaluasi_Geometri_LiDAR")
        aligned_out = (output_base(PROJECT_ROOT, args.experiment) / args.method.replace("-", "").lower() / args.statue_class /
                       args.model_id / 'geometry/aligned_reconstruction.ply')
        try:
            logger.log('Mencari skala, rotasi dan posisi otomatis (Sim(3)); periksa overlay hasilnya.' if args.auto_align
                       else 'Evaluasi terhadap LiDAR tanpa ICP; model harus sudah selaras.' if args.no_icp
                       else 'Menjalankan penyelarasan ICP dengan skala manual...')
            geo_metrics = compute_chamfer_distance(
                source_pcd_path=args.model_path,
                target_pcd_path=args.gt_lidar,
                scale_factor=args.scale_factor,
                apply_icp=not args.no_icp,
                auto_align=args.auto_align,
                lidar_unit=args.lidar_unit,
                save_aligned_path=str(aligned_out), log=logger.log,
                allow_weak_alignment=args.allow_weak_alignment
            )
            chamfer_cm = geo_metrics["chamfer_distance_cm"]
            if geo_metrics.get('alignment_passed') is False:
                logger.log('[PERINGATAN] Alignment tidak lolos pemeriksaan overlap; skor di bawah ini '
                           'ditandai alignment_passed=false dan belum valid sebagai hasil akhir.', level='warning')
            logger.log(f"Satuan LiDAR: {args.lidar_unit}; dikonversi ke meter sebelum evaluasi.")
            if args.auto_align:
                logger.log(f"Skala otomatis rekonstruksi ke meter: {geo_metrics['scale_factor_to_meters']:.10g}; "
                           f"inisialisasi: {geo_metrics['alignment_initialization']}. Periksa overlay hasil.")
            logger.log(f"Model terselaraskan disimpan di: {aligned_out}")
            logger.log(f"Chamfer Distance: {chamfer_cm:.4f} cm")
            logger.log(f"Akurasi (< 2cm): {geo_metrics['accuracy_at_2cm_pct']}%, Kelengkapan (< 2cm): {geo_metrics['completeness_at_2cm_pct']}%")
            logger.log(f"Akurasi (< 5cm): {geo_metrics['accuracy_at_5cm_pct']}%, Kelengkapan (< 5cm): {geo_metrics['completeness_at_5cm_pct']}%")
            for threshold in (2, 5):
                logger.log(f"Geometri < {threshold} cm — Precision: {geo_metrics[f'precision_at_{threshold}cm_pct']:.4f}%, "
                           f"Recall: {geo_metrics[f'recall_at_{threshold}cm_pct']:.4f}%, "
                           f"F1: {geo_metrics[f'f1_at_{threshold}cm_pct']:.4f}%")
            logger.set_geometry_metrics(geo_metrics)
            report = dict(geo_metrics, experiment=args.experiment, method=args.method, statue_class=args.statue_class, model_id=args.model_id,
                model_path=str(model_path.resolve()), model_signature=file_signature(model_path),
                gt_lidar_path=str(Path(args.gt_lidar).resolve()), gt_signature=file_signature(Path(args.gt_lidar)),
                aligned_model_path=str(aligned_out),
                aligned_lidar_path=str(aligned_out.with_name('ground_truth_meters.ply')),
                scale_factor_to_meters=geo_metrics.get('scale_factor_to_meters', args.scale_factor),
                apply_icp=not args.no_icp,
                geometry_options=dict(lidar_unit=args.lidar_unit,
                    alignment_mode='automatic' if args.auto_align else 'manual',
                    **({'allow_weak_alignment': True} if args.allow_weak_alignment else {}),
                    **({} if args.auto_align else {'scale_factor': args.scale_factor})))
            geometry_dir = output_base(PROJECT_ROOT, args.experiment) / args.method.replace("-", "").lower() / args.statue_class / args.model_id / 'geometry'
            geometry_dir.mkdir(parents=True, exist_ok=True)
            temporary = geometry_dir / 'metrics.json.tmp'
            temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
            temporary.replace(geometry_dir / 'metrics.json')
        except Exception as e:
            from pipeline_tools.geometry_alignment import AlignmentRejected
            logger.log(f'[GEOMETRI GAGAL] {type(e).__name__}: {e}', level='error')
            logger.log('Tidak ada skor geometri baru yang diterbitkan. Hasil sukses sebelumnya, jika ada, '
                       'bukan hasil percobaan ini.', level='error')
            if isinstance(e, AlignmentRejected):
                logger.log('[LANGKAH CEK] 1. Pastikan LiDAR dan rekonstruksi merupakan patung/model yang sama. '
                           '2. Periksa apakah alas, latar, dan cakupan crop sepadan. '
                           '3. Periksa kandidat overlay di folder diagnostik. '
                           'Noise atau bentuk yang tidak lengkap juga bisa mengganggu pencarian alignment; '
                           'log ini belum memastikan penyebabnya.', level='error')
                logger.log('[KIRIM LOG] Salin mulai [INPUT 1/2] sampai [GEOMETRI GAGAL], '
                           'beserta diagnostics.json bila tersedia. Tidak perlu mengulang training untuk diagnosis.', level='error')
            else:
                import traceback
                logger.log('[DETAIL TEKNIS]\n' + traceback.format_exc(), level='error')
            logger.end_stage("Evaluasi_Geometri_LiDAR")
            logger.set_parameters({
                "model_path": args.model_path,
                "gt_lidar_path": args.gt_lidar,
                "scale_factor_to_meters": args.scale_factor,
                "lidar_unit": args.lidar_unit,
                "alignment_mode": 'automatic' if args.auto_align else 'manual',
            })
            logger.finalize(success=False)
            sys.exit(1)
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
        "scale_factor_to_meters": geo_metrics.get('scale_factor_to_meters', args.scale_factor),
        "lidar_unit": args.lidar_unit,
        "alignment_mode": 'automatic' if args.auto_align else 'manual',
        "geometric_metrics_detail": geo_metrics
    })

    logger.finalize(success=True)
    logger.log("Evaluasi selesai! Tabel rekapitulasi CSV & Markdown berhasil diperbarui.")


if __name__ == "__main__":
    main()
