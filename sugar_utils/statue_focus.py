"""Statue-only region for SuGaR mesh extraction (ground removal + connected component).

Pure numpy/scipy so it can be tested without a GPU. One rule decides whether a 3D point
belongs to the statue; the same rule selects the points given to Poisson, trims the
Poisson mesh, and (hybrid mode) selects which 3DGS Gaussians the statue mesh replaces.
This is geometric, not semantic segmentation: objects touching the statue stay attached.
"""
import json

import numpy as np

REGION_VERSION = 1


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm < 1e-12:
        raise ValueError('Vektor arah tidak valid.')
    return vector / norm


def camera_up(rotations):
    """World up axis from camera-to-world rotations (COLMAP axes: camera y points down).

    Landscape photos: up = -mean(camera y). Portrait photos (phone turned 90 degrees): camera y
    is horizontal and averages out around an orbit, while camera x is the consistent vertical
    axis. The axis with the more consistent mean is used. For portrait photos the sign is not
    known here; extract_statue tries both signs and keeps the one with real ground below.
    """
    rotations = np.asarray(rotations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError('Rotasi kamera tidak valid.')
    landscape, portrait = -rotations[:, :, 1].mean(axis=0), rotations[:, :, 0].mean(axis=0)
    return _unit(portrait if np.linalg.norm(portrait) > np.linalg.norm(landscape) else landscape)


def fit_ground_plane(points, up, threshold, iterations=1000, min_alignment=0.85, seed=0, sample=200_000):
    """RANSAC restricted to planes facing `up`; returns (normal, offset, inlier_fraction) or None."""
    up = _unit(up)
    rng = np.random.default_rng(seed)
    if len(points) < 3:
        return None
    pts = points if len(points) <= sample else points[rng.choice(len(points), sample, replace=False)]
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        if normal @ up < 0:
            normal = -normal
        if normal @ up < min_alignment:
            continue
        offset = -normal @ a
        count = int((np.abs(pts @ normal + offset) < threshold).sum())
        if count > best_count:
            best, best_count = (normal, offset), count
    if best is None:
        return None
    normal, offset = best
    inliers = pts[np.abs(pts @ normal + offset) < threshold]
    if len(inliers) >= 3:
        centroid = inliers.mean(axis=0)
        refined = np.linalg.svd(inliers - centroid, full_matrices=False)[2][-1]
        if refined @ up < 0:
            refined = -refined
        if refined @ up >= min_alignment:
            normal, offset = refined, -refined @ centroid
    return normal, float(offset), best_count / len(pts)


def _grid_index(region, points):
    index = np.floor((points - region['origin']) / region['voxel']).astype(np.int64)
    inside = ((index >= 0) & (index < np.asarray(region['grid'].shape))).all(axis=1)
    return index, inside


def region_contains(region, points):
    """True for points inside the statue voxels and above the ground cut."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    index, inside = _grid_index(region, points)
    result = np.zeros(len(points), dtype=bool)
    hit = index[inside]
    result[inside] = region['grid'][hit[:, 0], hit[:, 1], hit[:, 2]]
    if region['has_ground']:
        result &= points @ region['normal'] + region['offset'] > region['keep_min']
    return result


def extract_statue(points, up, center, box_min, box_max, voxel_fraction=0.01,
                   threshold_fraction=0.004, min_points_per_voxel=2, margin_voxels=2):
    """Build the statue region from surface points inside the camera focus box."""
    from scipy import ndimage
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    box_min, box_max, center = (np.asarray(v, dtype=float) for v in (box_min, box_max, center))
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Titik permukaan foreground kosong atau tidak finite.')
    up = _unit(up)
    extent = float(np.linalg.norm(box_max - box_min))
    threshold = threshold_fraction * extent
    has_ground, normal, offset, ground_fraction = False, up, 0., 0.
    # The sign of `up` is unknown for portrait photos: real ground has the scene above it,
    # so only the correct sign passes the "below" check.
    for direction in (up, -up):
        plane = fit_ground_plane(points, direction, threshold)
        if plane is None or plane[2] < 0.05:
            continue
        signed = points @ plane[0] + plane[1]
        inliers = points[np.abs(signed) < threshold]
        axis = np.cross(plane[0], [1., 0., 0.] if abs(plane[0][0]) < 0.9 else [0., 1., 0.])
        axis /= np.linalg.norm(axis)
        spreads = [np.subtract(*np.percentile(inliers @ a, [95, 5])) for a in (axis, np.cross(plane[0], axis))]
        outside = np.abs(signed) >= 2 * threshold
        below = (signed < -2 * threshold).sum() / max(1, outside.sum())
        # Real ground is wide and has the scene above it; a horizontal slice through
        # the statue (or a table-sized object) fails one of these checks.
        if min(spreads) >= 0.5 * float(np.min(box_max - box_min)) and below <= 0.1:
            has_ground, (normal, offset, ground_fraction), up = True, plane, direction
            break
    keep_min = 2 * threshold
    candidates = points[points @ normal + offset > keep_min] if has_ground else points
    if not len(candidates):
        raise ValueError('Tidak ada titik di atas bidang tanah.')
    voxel = voxel_fraction * extent
    shape = np.maximum(np.ceil((box_max - box_min) / voxel).astype(np.int64), 1)
    index = np.clip(np.floor((candidates - box_min) / voxel).astype(np.int64), 0, shape - 1)
    counts = np.zeros(tuple(shape), dtype=np.int32)
    np.add.at(counts, (index[:, 0], index[:, 1], index[:, 2]), 1)
    occupied = counts >= min_points_per_voxel
    if not occupied.any():
        occupied = counts > 0
    bridged = ndimage.binary_dilation(occupied, iterations=1)
    labels, n_components = ndimage.label(bridged, structure=np.ones((3, 3, 3)))
    labels = np.where(occupied, labels, 0)
    sizes = np.bincount(labels.ravel(), minlength=n_components + 1)
    sizes[0] = 0
    voxels = np.argwhere(labels > 0)
    voxel_centers = box_min + (voxels + 0.5) * voxel
    offsets = voxel_centers - center
    horizontal = np.linalg.norm(offsets - np.outer(offsets @ up, up), axis=1)
    owner = labels[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    nearest = np.full(n_components + 1, np.inf)
    np.minimum.at(nearest, owner, horizontal)
    # Ignore tiny floaters near the focus axis; prefer the closest substantial component.
    eligible = sizes >= max(1, 0.1 * sizes.max())
    nearest[~eligible] = np.inf
    chosen = int(np.argmin(nearest))
    statue = ndimage.binary_dilation(labels == chosen, iterations=margin_voxels)
    members = voxel_centers[owner == chosen]
    region = dict(version=REGION_VERSION, origin=box_min, voxel=float(voxel), grid=statue,
                  has_ground=has_ground, normal=np.asarray(normal, dtype=float), offset=float(offset),
                  keep_min=float(keep_min), up=up, center=center,
                  statue_min=members.min(axis=0) - voxel / 2, statue_max=members.max(axis=0) + voxel / 2)
    region['summary'] = dict(ground_found=has_ground, ground_inlier_fraction=float(ground_fraction),
                             components=int(n_components), statue_voxels=int(sizes[chosen]),
                             statue_points=int(region_contains(region, points).sum()), input_points=len(points),
                             statue_size=(region['statue_max'] - region['statue_min']).tolist())
    return region


def save_region(path, region):
    np.savez_compressed(path, version=region['version'], origin=region['origin'], voxel=region['voxel'],
                        grid=region['grid'], has_ground=region['has_ground'], normal=region['normal'],
                        offset=region['offset'], keep_min=region['keep_min'], up=region['up'],
                        center=region['center'], statue_min=region['statue_min'],
                        statue_max=region['statue_max'], summary=json.dumps(region['summary']))


def load_region(path):
    with np.load(path, allow_pickle=False) as data:
        if int(data['version']) != REGION_VERSION:
            raise ValueError('Versi statue_region tidak didukung.')
        region = {key: data[key] for key in data.files}
    for key in ('voxel', 'offset', 'keep_min'):
        region[key] = float(region[key])
    region['has_ground'] = bool(region['has_ground'])
    region['grid'] = region['grid'].astype(bool)
    region['summary'] = json.loads(str(region['summary']))
    return region
