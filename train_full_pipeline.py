import os
import argparse
from sugar_utils.general_utils import str2bool


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')
    
    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  
    
    # Vanilla 3DGS optimization at beginning
    parser.add_argument('--gs_output_dir', type=str, default=None,
                        help='(Optional) If None, will automatically train a vanilla Gaussian Splatting model at the beginning of the training. '
                        'Else, skips the vanilla Gaussian Splatting optimization and use the checkpoint in the provided directory.')
    
    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str,
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf", "density" or "dn_consistency". ' 
                        'We recommend using "dn_consistency" for the best mesh quality.')
    
    # Extract mesh
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000, 
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('--project_mesh_on_surface_points', type=str2bool, default=True, 
                        help='If True, project the mesh on the surface points for better details.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    
    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1, 
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations.')
    
    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_obj', type=str2bool, default=True, 
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                        'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=8, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, 
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')
    
    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                        'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")
      
    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')

    # Parse arguments
    args = parser.parse_args()
    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Using low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Using high poly config.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    if args.export_obj:
        print('Will export a UV-textured mesh as an .obj file.')
    if args.export_ply:
        print('Will export a ply file with the refined 3D Gaussians at the end of the training.')
        
    import subprocess
    import sys

    # Output directory for the vanilla 3DGS checkpoint
    norm_scene = os.path.normpath(args.scene_path).replace('\\', '/').rstrip('/')
    scene_parts = [p for p in norm_scene.split('/') if p]
    scene_name = scene_parts[-1] if scene_parts else 'scene'

    if args.gs_output_dir is None:
        gs_checkpoint_dir = os.path.join("output", "vanilla_gs", scene_name)
        if not gs_checkpoint_dir.endswith(os.path.sep):
            gs_checkpoint_dir += os.path.sep

        # Trains a 3DGS scene for 7k iterations
        train_gs_cmd = [
            sys.executable, "./gaussian_splatting/train.py",
            "-s", str(args.scene_path),
            "-m", str(gs_checkpoint_dir),
            "--iterations", "7000"
        ]
        if args.white_background:
            train_gs_cmd.append("-w")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        print(f"Running 3DGS 7k training: {' '.join(train_gs_cmd)}")
        subprocess.run(train_gs_cmd, env=env, check=True)
    else:
        print("A vanilla 3DGS checkpoint was provided. Skipping the vanilla 3DGS optimization.")
        gs_checkpoint_dir = args.gs_output_dir
        if not gs_checkpoint_dir.endswith(os.path.sep):
            gs_checkpoint_dir += os.path.sep
    
    # Runs the train.py python script with the given arguments
    train_sugar_cmd = [
        sys.executable, "train.py",
        "-s", str(args.scene_path),
        "-c", str(gs_checkpoint_dir),
        "-i", "7000",
        "-r", str(args.regularization_type),
        "-l", str(args.surface_level),
        "-v", str(args.n_vertices_in_mesh),
        "--project_mesh_on_surface_points", str(args.project_mesh_on_surface_points),
        "-g", str(args.gaussians_per_triangle),
        "-f", str(args.refinement_iterations),
        "-t", str(args.export_obj),
        "--square_size", str(args.square_size),
        "--postprocess_mesh", str(args.postprocess_mesh),
        "--postprocess_density_threshold", str(args.postprocess_density_threshold),
        "--postprocess_iterations", str(args.postprocess_iterations),
        "--export_ply", str(args.export_ply),
        "--low_poly", str(args.low_poly),
        "--high_poly", str(args.high_poly),
        "--eval", str(args.eval),
        "--gpu", str(args.gpu),
        "--white_background", str(args.white_background),
        "--center_bbox", str(args.center_bbox),
    ]
    if args.bboxmin is not None and args.bboxmin != 'None':
        train_sugar_cmd.extend(["--bboxmin", str(args.bboxmin)])
    if args.bboxmax is not None and args.bboxmax != 'None':
        train_sugar_cmd.extend(["--bboxmax", str(args.bboxmax)])
    if args.refinement_time is not None:
        train_sugar_cmd.extend(["--refinement_time", str(args.refinement_time)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Running SuGaR training: {' '.join(train_sugar_cmd)}")
    subprocess.run(train_sugar_cmd, env=env, check=True)