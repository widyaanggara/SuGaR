"""Evaluate actual checkpoints on the same held-out camera split."""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_tools.dataset_registry import has_checkpoint, has_sugar_result, scene_pairs, file_signature
from pipeline_tools.gpu_lock import gpu_lock
from pipeline_tools.logger_utils import ThesisLogger


def training_background(output):
    """Read one literal setting from 3DGS cfg_args without executing its contents."""
    tree = ast.parse((Path(output) / 'cfg_args').read_text(encoding='utf-8'), mode='eval')
    if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name) or tree.body.func.id != 'Namespace':
        raise ValueError('Format cfg_args 3DGS tidak valid.')
    options = {kw.arg: kw.value for kw in tree.body.keywords}
    if 'eval' not in options or ast.literal_eval(options['eval']) is not True:
        raise ValueError('Checkpoint harus dilatih dengan --eval agar citra uji tidak masuk pelatihan.')
    return bool(ast.literal_eval(options.get('white_background', ast.Constant(False))))


def evaluate_rendering(cls, mdl, method, gpu, logger):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ.setdefault('SUGAR_IMAGE_DEVICE', 'cpu')
    import numpy as np
    import torch
    from PIL import Image
    from sugar_scene.gs_model import GaussianSplattingWrapper
    from gaussian_splatting.utils.image_utils import psnr
    from gaussian_splatting.utils.loss_utils import ssim
    from gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA tidak tersedia. Jalankan preflight di environment sugar pada server.')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    scene = PROJECT_ROOT / 'datasets' / cls / mdl
    baseline = PROJECT_ROOT / 'outputs/3dgs' / cls / mdl
    output = PROJECT_ROOT / 'outputs' / ('3dgs' if method == '3DGS' else 'sugar') / cls / mdl
    iteration = 30000 if method == '3DGS' else 7000
    if not has_checkpoint(baseline, iteration):
        raise FileNotFoundError(f'Checkpoint 3DGS {iteration} belum lengkap.')
    white = training_background(baseline)
    manifest = None
    if method == 'SuGaR':
        if not has_sugar_result(output):
            raise FileNotFoundError('Hasil pipeline SuGaR belum lengkap; perlu pipeline_result.json.')
        manifest = json.loads((output / 'pipeline_result.json').read_text(encoding='utf-8'))
        if manifest['white_background'] != white:
            raise ValueError('Background SuGaR dan baseline berbeda; latih dengan konfigurasi yang sama.')
    wrapper = GaussianSplattingWrapper(source_path=str(scene), output_path=str(baseline),
        iteration_to_load=iteration, load_gt_images=True, eval_split=True, eval_split_interval=8,
        white_background=white)
    if not wrapper.test_cam_list:
        raise ValueError('Tidak ada kamera uji.')
    refined = None
    if method == 'SuGaR':
        import open3d as o3d
        from sugar_scene.sugar_model import SuGaR
        from sugar_utils.spherical_harmonics import SH2RGB
        checkpoint = torch.load(str(output / manifest['refined_checkpoint']), map_location=wrapper.device)
        mesh = o3d.io.read_triangle_mesh(str(output / manifest['coarse_mesh']))
        if not mesh.has_triangles():
            raise ValueError('Coarse mesh kosong.')
        refined = SuGaR(nerfmodel=wrapper, points=checkpoint['state_dict']['_points'],
            colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
            initialize=False, sh_levels=wrapper.gaussians.active_sh_degree + 1,
            keep_track_of_knn=False, knn_to_track=0, beta_mode='average',
            surface_mesh_to_bind=mesh, n_gaussians_per_surface_triangle=manifest['gaussians_per_triangle'])
        refined.load_state_dict(checkpoint['state_dict'])
        refined.eval()
        del checkpoint
    criterion = LPIPS(net_type='vgg', version='0.1').to(wrapper.device).eval()
    folder = output / 'evaluation'
    for sub in ('renders', 'ground_truth'):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    result_path = folder / 'metrics.json'
    result_path.unlink(missing_ok=True)
    values = []
    with torch.no_grad():
        for index, camera in enumerate(wrapper.test_cam_list):
            gt = wrapper.get_test_gt_image(index, to_cuda=True).permute(2, 0, 1).unsqueeze(0)
            if refined is None:
                prediction = wrapper.render_image(nerf_cameras=wrapper.test_cameras, camera_indices=index)
            else:
                prediction = refined.render_image_gaussian_rasterizer(nerf_cameras=wrapper.test_cameras,
                    camera_indices=index, bg_color=wrapper.background,
                    sh_deg=wrapper.gaussians.active_sh_degree, compute_color_in_rasterizer=True)
            prediction = prediction.clamp(0, 1).permute(2, 0, 1).unsqueeze(0)
            if prediction.shape != gt.shape:
                raise ValueError('Ukuran rendering dan ground truth berbeda.')
            row = {'image': camera.image_name, 'psnr': float(psnr(prediction, gt).mean()),
                   'ssim': float(ssim(prediction, gt)),
                   'lpips': float(criterion(prediction * 2 - 1, gt * 2 - 1).mean())}
            values.append(row)
            for sub, tensor in (('renders', prediction), ('ground_truth', gt)):
                pixels = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
                Image.fromarray(pixels).save(folder / sub / f'{index:05d}.png')
            logger.log(f"[EVAL] {index + 1}/{len(wrapper.test_cam_list)} {camera.image_name}: PSNR={row['psnr']:.4f}")
    metrics = {key: float(np.mean([row[key] for row in values])) for key in ('psnr', 'ssim', 'lpips')}
    report = dict(metrics, method=method, representation='gaussian_splatting' if refined is None else 'sugar_hybrid',
                  eval_split_interval=8, white_background=white, per_view=values,
                  lpips_network='vgg', lpips_input_range=[-1, 1],
                  baseline_iteration=iteration, checkpoint=manifest['refined_checkpoint'] if manifest else
                  'point_cloud/iteration_30000/point_cloud.ply')
    checkpoint_path = output / manifest['refined_checkpoint'] if manifest else baseline / report['checkpoint']
    report['checkpoint_signature'] = file_signature(checkpoint_path)
    temp = result_path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(result_path)
    logger.set_parameters({'phase': 'visual_evaluation', 'eval_split_interval': 8,
                           'test_images': [row['image'] for row in values], 'output': str(folder),
                           'representation': report['representation']})
    logger.set_metrics(**metrics)
    logger.record_vram()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statue_class', required=True)
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--method', choices=['3DGS', 'SuGaR'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('GPU harus >= 0.')
    scene_pairs(args.statue_class, args.model_id)
    logger = ThesisLogger(args.statue_class, args.model_id, args.method)
    try:
        with gpu_lock(args.gpu):
            logger.start_stage('Evaluasi_Visual')
            evaluate_rendering(args.statue_class, args.model_id, args.method, args.gpu, logger)
            logger.end_stage('Evaluasi_Visual')
        logger.finalize(True)
        return 0
    except Exception as exc:
        logger.log(f'[ERROR] Evaluasi visual gagal: {exc}', level='error')
        logger.finalize(False)
        return 1


if __name__ == '__main__':
    sys.exit(main())
