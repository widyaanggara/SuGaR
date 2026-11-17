"""Versioned, shared image and metric conventions (imports are GPU-lazy)."""
VERSION = 'longest-side-v1-png8-vgg'


def specification(downscale=1, max_size=1600):
    return dict(version=VERSION, image_downscale=downscale, image_max_size=max_size,
                resize='longest_side_no_upscale', test_split='sorted_name_every_8',
                lpips_network='vgg', lpips_input_range=[-1, 1], metric_pixels='rgb_uint8')


def quantize(image):
    return (image.clamp(0, 1) * 255).round() / 255


def save_rgb_png(image, path):
    import numpy as np
    from PIL import Image
    Image.fromarray((np.clip(np.nan_to_num(image), 0, 1) * 255).round().astype(np.uint8)).save(path, 'PNG')


def install_camera_sizing(camera_utils, downscale, max_size):
    from gaussian_splatting.utils.resolution_utils import scaled_resolution
    original_load = camera_utils.loadCam
    original_convert = camera_utils.PILtoTorch
    def load(args, camera_id, camera_info, resolution_scale):
        size = scaled_resolution(*camera_info.image.size, downscale, max_size, resolution_scale)
        # Native cameras derive dimensions/intrinsics from the resized tensor.
        camera_utils.PILtoTorch = lambda image, ignored: original_convert(image, size)
        try:
            return original_load(args, camera_id, camera_info, resolution_scale)
        finally:
            camera_utils.PILtoTorch = original_convert
    camera_utils.loadCam = load


def install_metrics():
    import lpipsPyTorch
    import utils.loss_utils as native_loss
    import utils.image_utils as native_image
    from gaussian_splatting.utils.loss_utils import ssim
    from gaussian_splatting.utils.image_utils import psnr
    from gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS
    criterion = None
    def perceptual(x, y, net_type='vgg', version='0.1'):
        nonlocal criterion
        if criterion is None:
            criterion = LPIPS(net_type='vgg', version='0.1').to(x.device).eval()
        return criterion(x * 2 - 1, y * 2 - 1)
    lpipsPyTorch.lpips = perceptual
    native_loss.ssim = ssim
    native_image.psnr = psnr
