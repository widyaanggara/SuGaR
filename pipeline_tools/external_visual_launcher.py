"""Apply shared conventions without modifying separately cloned repositories."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--image_downscale', type=int, default=1)
    parser.add_argument('--image_max_size', type=int, default=1600)
    parser.add_argument('--visual_protocol', choices=('shared', 'legacy'), default='shared')
    parser.add_argument('script', choices=('train.py', 'render.py', 'metrics.py', 'metric.py'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # RaDe-GS normally sets this before its first torch import. Do the same here.
    if args.script == 'train.py' and '--enable_allocator_gc' in args.arguments:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native,garbage_collection_threshold:0.8'
    sys.path.insert(0, str(args.repo.resolve()))
    from pipeline_tools.visual_protocol import install_camera_sizing, install_metrics
    if args.visual_protocol == 'shared' and args.script in ('train.py', 'render.py'):
        # Initialize scene first, as native entry points do. camera_utils imports
        # scene.cameras while scene.__init__ imports camera_utils functions.
        import scene
        import utils.camera_utils as camera_utils
        install_camera_sizing(camera_utils, args.image_downscale, args.image_max_size)
        if args.script == 'render.py' and (args.repo / 'metrics.py').is_file():
            # 2DGS originally truncates PNG values; align with torchvision/SuGaR rounding.
            import utils.render_utils as render_utils
            from pipeline_tools.visual_protocol import save_rgb_png
            render_utils.save_img_u8 = save_rgb_png
    elif args.visual_protocol == 'shared':
        install_metrics()
    sys.argv = [str(args.repo / args.script), *args.arguments]
    try:
        runpy.run_path(str(args.repo / args.script), run_name='__main__')
    finally:
        if args.script == 'train.py':
            from pipeline_tools.training_vram import write_training_peak
            write_training_peak()


if __name__ == '__main__':
    main()
