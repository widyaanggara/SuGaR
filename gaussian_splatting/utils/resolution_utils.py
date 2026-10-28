"""Shared image sizing for 3DGS and SuGaR (no tensor dependencies)."""
def scaled_resolution(width, height, downscale=1, max_size=1600, resolution_scale=1):
    if downscale not in (1, 2, 4, 8) or max_size < 0 or resolution_scale <= 0:
        raise ValueError('Invalid image resolution settings.')
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive.')
    factor = max(float(downscale), max(width, height) / max_size if max_size else 1.)
    factor *= resolution_scale
    return max(1, int(round(width / factor))), max(1, int(round(height / factor)))
