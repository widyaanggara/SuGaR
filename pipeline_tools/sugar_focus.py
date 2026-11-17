"""Conservative camera-focus crop; does not perform semantic segmentation."""
import numpy as np


def camera_focus_region(cameras):
    if len(cameras) < 3:
        raise ValueError('Minimal tiga kamera diperlukan untuk fokus otomatis.')
    positions = np.asarray([c['position'] for c in cameras], dtype=float)
    rotations = np.asarray([c['rotation'] for c in cameras], dtype=float)
    if positions.shape != (len(cameras), 3) or rotations.shape != (len(cameras), 3, 3):
        raise ValueError('Bentuk data kamera tidak valid.')
    directions = rotations[:, :, 2]
    norms = np.linalg.norm(directions, axis=1)
    if not np.isfinite(positions).all() or not np.isfinite(directions).all() or (norms < 1e-8).any():
        raise ValueError('Koordinat kamera tidak valid.')
    directions = directions / norms[:, None]
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    weights = np.ones(len(cameras))
    for _ in range(5):
        matrix = np.einsum('n,nij->ij', weights, projectors)
        if np.linalg.cond(matrix) > 1e6:
            raise ValueError('Arah kamera terlalu sejajar untuk menentukan pusat patung.')
        center = np.linalg.solve(matrix, np.einsum('n,nij,nj->i', weights, projectors, positions))
        residuals = np.linalg.norm(np.einsum('nij,nj->ni', projectors, center - positions), axis=1)
        scale = max(float(np.median(residuals)), 1e-8)
        weights = np.minimum(1., 2.5 * scale / np.maximum(residuals, 1e-8))
    depth = np.sum((center - positions) * directions, axis=1)
    distance = np.linalg.norm(center - positions, axis=1)
    valid = (depth > 0) & (residuals < 0.35 * distance)
    if valid.sum() < max(3, 0.6 * len(cameras)):
        raise ValueError('Kamera tidak cukup terpusat pada satu objek; mesh scene dipertahankan.')
    half_fov = np.asarray([max(float(c['width']) / (2 * float(c['fx'])),
                               float(c['height']) / (2 * float(c['fy']))) for c in cameras])
    if not np.isfinite(half_fov).all() or (half_fov <= 0).any():
        raise ValueError('Intrinsik kamera tidak valid.')
    radius = 1.15 * float(np.median(depth[valid] * half_fov[valid]))
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError('Ukuran fokus kamera tidak valid.')
    return {'min': (center - radius).tolist(), 'max': (center + radius).tolist(), 'min_triangles': 50}
