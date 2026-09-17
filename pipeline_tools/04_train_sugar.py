import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline_tools.logger_utils import ThesisLogger


def train_sugar_pipeline(scene_path: str,
                         checkpoint_3dgs: str,
                         output_base_dir: str,
                         regularization_type: str = "dn_consistency",
                         refinement_time: str = "short",
                         n_vertices: int = 250000,
                         gaussians_per_triangle: int = 1,
                         export_obj: bool = True,
                         gpu: int = 0,
                         white_background: bool = False,
                         logger: ThesisLogger = None):
    """
    Menjalankan pipeline SuGaR lengkap secara teratur dan aman dari masalah memori/path.
    
    Tahapan:
    1. Coarse Surface-Aligned Optimization (dn_consistency)
    2. Poisson Surface Mesh Extraction
    3. Refinement Training (dengan setting short / 2k iterasi agar tidak berhari-hari)
    4. Export Textured Mesh (.obj, .mtl, .png)
    """
    from sugar_utils.general_utils import str2bool
    from sugar_trainers.coarse_density import coarse_training_with_density_regularization
    from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization
    from sugar_trainers.coarse_density_and_dn_consistency import coarse_training_with_density_regularization_and_dn_consistency
    from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
    from sugar_trainers.refine import refined_training
    from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    scene_path = str(Path(scene_path).resolve())
    checkpoint_3dgs = str(Path(checkpoint_3dgs).resolve())
    if not checkpoint_3dgs.endswith(os.path.sep):
        checkpoint_3dgs += os.path.sep

    output_dir = Path(output_base_dir).resolve()
    coarse_dir = output_dir / "coarse"
    mesh_dir = output_dir / "mesh"
    refined_dir = output_dir / "refined"
    textured_dir = output_dir / "textured_mesh"

    for d in [coarse_dir, mesh_dir, refined_dir, textured_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Tentukan jumlah iterasi refinement
    if refinement_time == "short":
        refinement_iterations = 2000
    elif refinement_time == "medium":
        refinement_iterations = 7000
    elif refinement_time == "long":
        refinement_iterations = 15000
    else:
        refinement_iterations = 2000

    if logger:
        logger.log(f"Memulai Pipeline SuGaR untuk Scene: {scene_path}")
        logger.log(f"Regularisasi: {regularization_type}, Refinement Iterations: {refinement_iterations}, Target Vertex: {n_vertices}")

    # ==================== 1. Coarse SuGaR ====================
    if logger:
        logger.start_stage("SuGaR_01_Coarse_Optimization")

    coarse_args = AttrDict({
        'checkpoint_path': checkpoint_3dgs,
        'scene_path': scene_path,
        'iteration_to_load': 7000,
        'output_dir': str(coarse_dir),
        'eval': True,
        'estimation_factor': 0.2,
        'normal_factor': 0.2,
        'gpu': gpu,
        'white_background': white_background,
    })

    if regularization_type == 'sdf':
        coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args)
    elif regularization_type == 'density':
        coarse_sugar_path = coarse_training_with_density_regularization(coarse_args)
    elif regularization_type == 'dn_consistency':
        coarse_sugar_path = coarse_training_with_density_regularization_and_dn_consistency(coarse_args)
    else:
        raise ValueError(f"Unknown regularization type: {regularization_type}")

    if logger:
        logger.end_stage("SuGaR_01_Coarse_Optimization")
        logger.log(f"Coarse SuGaR tersimpan di: {coarse_sugar_path}")

    # ==================== 2. Mesh Extraction ====================
    if logger:
        logger.start_stage("SuGaR_02_Mesh_Extraction")

    coarse_mesh_args = AttrDict({
        'scene_path': scene_path,
        'checkpoint_path': checkpoint_3dgs,
        'iteration_to_load': 7000,
        'coarse_model_path': coarse_sugar_path,
        'surface_level': 0.3,
        'decimation_target': n_vertices,
        'project_mesh_on_surface_points': True,
        'mesh_output_dir': str(mesh_dir),
        'bboxmin': None,
        'bboxmax': None,
        'center_bbox': True,
        'gpu': gpu,
        'eval': True,
        'use_centers_to_extract_mesh': False,
        'use_marching_cubes': False,
        'use_vanilla_3dgs': False,
    })
    coarse_mesh_path = extract_mesh_from_coarse_sugar(coarse_mesh_args)[0]

    if logger:
        logger.end_stage("SuGaR_02_Mesh_Extraction")
        logger.log(f"Poisson surface mesh tersimpan di: {coarse_mesh_path}")

    # ==================== 3. Refine SuGaR ====================
    if logger:
        logger.start_stage("SuGaR_03_Refinement")

    refined_args = AttrDict({
        'scene_path': scene_path,
        'checkpoint_path': checkpoint_3dgs,
        'mesh_path': coarse_mesh_path,
        'output_dir': str(refined_dir),
        'iteration_to_load': 7000,
        'normal_consistency_factor': 0.1,
        'gaussians_per_triangle': gaussians_per_triangle,
        'n_vertices_in_fg': n_vertices,
        'refinement_iterations': refinement_iterations,
        'bboxmin': None,
        'bboxmax': None,
        'export_ply': True,
        'eval': True,
        'gpu': gpu,
        'white_background': white_background,
    })
    refined_sugar_path = refined_training(refined_args)

    if logger:
        logger.end_stage("SuGaR_03_Refinement")
        logger.log(f"Refined model tersimpan di: {refined_sugar_path}")

    # ==================== 4. Textured Mesh Extraction ====================
    if export_obj:
        if logger:
            logger.start_stage("SuGaR_04_Textured_Mesh_Export")

        refined_mesh_args = AttrDict({
            'scene_path': scene_path,
            'iteration_to_load': 7000,
            'checkpoint_path': checkpoint_3dgs,
            'refined_model_path': refined_sugar_path,
            'mesh_output_dir': str(textured_dir),
            'n_gaussians_per_surface_triangle': gaussians_per_triangle,
            'square_size': 8,
            'eval': True,
            'gpu': gpu,
            'postprocess_mesh': False,
            'postprocess_density_threshold': 0.1,
            'postprocess_iterations': 5,
        })
        refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)

        if logger:
            logger.end_stage("SuGaR_04_Textured_Mesh_Export")
            logger.log(f"Textured OBJ Mesh tersimpan di: {textured_dir}")

    if logger:
        logger.log("Seluruh tahapan SuGaR berhasil diselesaikan!")


def main():
    parser = argparse.ArgumentParser(description="Pelatihan SuGaR terstruktur bebas bug path dan hemat VRAM.")
    parser.add_argument("--scene_path", "-s", type=str, required=True, help="Folder dataset (misal: datasets/Dewa/model_01).")
    parser.add_argument("--checkpoint_3dgs", "-c", type=str, required=True, help="Folder output 3DGS yang memuat checkpoint 7000.")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Folder tujuan (default: outputs/sugar/<Kelas>/<Model_ID>).")
    parser.add_argument("--statue_class", type=str, default="Dewa", help="Nama kelas patung.")
    parser.add_argument("--model_id", type=str, default="model_01", help="ID model patung.")
    parser.add_argument("--regularization", "-r", type=str, default="dn_consistency", choices=["dn_consistency", "density", "sdf"], help="Tipe regularisasi (default: dn_consistency).")
    parser.add_argument("--refinement_time", type=str, default="short", choices=["short", "medium", "long"], help="Waktu refinement (default: short / 2000 iterasi agar cepat dan aman VRAM).")
    parser.add_argument("--vertices", type=int, default=250000, help="Jumlah target vertex mesh (default: 250000).")
    parser.add_argument("--gpu", type=int, default=0, help="ID GPU device (default: 0).")
    parser.add_argument("--white_background", "-w", action="store_true", help="Gunakan background putih.")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "sugar" / args.statue_class / args.model_id
    else:
        output_dir = Path(args.output_dir)

    logger = ThesisLogger(statue_class=args.statue_class, model_id=args.model_id, method="SuGaR")
    logger.set_parameters({
        "regularization": args.regularization,
        "refinement_time": args.refinement_time,
        "target_vertices": args.vertices,
        "gpu": args.gpu
    })

    try:
        train_sugar_pipeline(
            scene_path=args.scene_path,
            checkpoint_3dgs=args.checkpoint_3dgs,
            output_base_dir=str(output_dir),
            regularization_type=args.regularization,
            refinement_time=args.refinement_time,
            n_vertices=args.vertices,
            gpu=args.gpu,
            white_background=args.white_background,
            logger=logger
        )
        logger.finalize(success=True)
    except Exception as e:
        logger.log(f"Error pada pipeline SuGaR: {str(e)}", level="error")
        logger.finalize(success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
