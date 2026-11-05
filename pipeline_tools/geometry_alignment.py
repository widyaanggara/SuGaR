"""Seeded, CPU-only similarity registration for the same object/ROI.

LiDAR units are supplied explicitly. Robust normalization initializes scale;
proper PCA rotations and optional FPFH/RANSAC initialize orientation. Trimmed,
bidirectional nearest-neighbour fitting then estimates one uniform scale,
rotation and translation (Sim(3)), never a nonrigid deformation.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from pipeline_tools.geometry_config import ALIGNMENT_VERSION


def checked_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30:
        raise ValueError('Penyelarasan memerlukan minimal 30 titik 3D per model.')
    if not np.isfinite(points).all():
        raise ValueError('Koordinat model mengandung NaN atau infinity.')
    centered = points - np.median(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 0 or singular[1] < singular[0] * 1e-6:
        raise ValueError('Titik berimpit atau membentuk garis; penyelarasan tidak dapat ditentukan.')
    return points


def transform_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def fit_similarity(source, target):
    """Least-squares Umeyama fit, reflection forbidden, scale strictly positive."""
    source, target = np.asarray(source), np.asarray(target)
    x, y = source.mean(axis=0), target.mean(axis=0)
    a, b = source - x, target - y
    variance = np.sum(a * a) / len(a)
    if variance <= np.finfo(float).eps:
        raise ValueError('Korespondensi tidak cukup untuk menghitung skala.')
    u, values, vt = np.linalg.svd(b.T @ a / len(a))
    signs = np.ones(3)
    signs[-1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    rotation = (u * signs) @ vt
    scale = float(values @ signs / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Estimasi skala tidak valid.')
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = y - scale * rotation @ x
    return matrix


def _normalized(points, limit=2500):
    center = np.median(points, axis=0)
    radius = float(np.quantile(np.linalg.norm(points - center, axis=1), .9))
    if radius <= 1e-12:
        raise ValueError('Ukuran objek terlalu kecil atau titik berimpit.')
    matrix = np.eye(4)
    matrix[:3, :3] /= radius
    matrix[:3, 3] = -center / radius
    normalized = transform_points(points, matrix)
    # Only registration excludes the furthest 1%; evaluation keeps all samples.
    distance = np.linalg.norm(normalized, axis=1)
    normalized = normalized[distance <= np.quantile(distance, .99)]
    if len(normalized) > limit:
        normalized = normalized[np.random.default_rng(0).choice(len(normalized), limit, replace=False)]
    return normalized, matrix


def _pca_candidates(source, target):
    _, source_basis = np.linalg.eigh(np.cov(source.T))
    _, target_basis = np.linalg.eigh(np.cov(target.T))
    yield np.eye(4), 'identity'
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = target_basis[:, permutation] @ np.diag(signs) @ source_basis.T
            if np.linalg.det(rotation) > 0:
                matrix = np.eye(4)
                matrix[:3, :3] = rotation
                matrix[:3, 3] = target.mean(axis=0) - rotation @ source.mean(axis=0)
                yield matrix, 'pca'


def _feature_candidate(source, target):
    # Optional only in lightweight tests; Open3D is already required by the evaluator.
    try:
        import open3d as o3d
    except ImportError:
        return None
    o3d.utility.random.seed(0)
    clouds, features = [], []
    for points in (source, target):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.05)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.15, max_nn=40))
        features.append(o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=.3, max_nn=100)))
        clouds.append(cloud)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        clouds[0], clouds[1], features[0], features[1], True, .15,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.15)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(50000, .999))
    return result.transformation if result.fitness > 0 else None


def _distances(source, target, matrix, target_tree):
    transformed = transform_points(source, matrix)
    forward, target_ids = target_tree.query(transformed)
    backward, source_ids = cKDTree(transformed).query(target)
    return forward, backward, target_ids, source_ids


def _score(source, target, matrix, target_tree):
    forward, backward, _, _ = _distances(source, target, matrix, target_tree)
    # Both directions prevent a tiny collapsed reconstruction scoring well.
    return sum(np.mean(np.sort(d)[:max(1, int(len(d) * .8))]) for d in (forward, backward))


def _refine(source, target, matrix, target_tree, iterations):
    for _ in range(iterations):
        forward, backward, target_ids, source_ids = _distances(source, target, matrix, target_tree)
        pairs_source = np.concatenate((source, source[source_ids]))
        pairs_target = np.concatenate((target[target_ids], target))
        distances = np.concatenate((forward, backward))
        keep = distances <= np.quantile(distances, .8)
        fitted = fit_similarity(pairs_source[keep], pairs_target[keep])
        scale = np.cbrt(np.linalg.det(fitted[:3, :3]))
        if not .2 <= scale <= 5:
            break
        difference = np.max(np.abs(fitted - matrix))
        matrix = fitted
        if difference < 1e-7:
            break
    return matrix


def automatic_similarity_alignment(source_points, target_points, use_features=True):
    """Return original-source -> metre-LiDAR transform and alignment diagnostics.

    Requires comparable object coverage. A failed overlap check raises instead
    of publishing geometry scores. Passing is a numerical check, not a claim
    that semantic alignment was independently verified.
    """
    source_points, target_points = checked_points(source_points), checked_points(target_points)
    source, source_norm = _normalized(source_points)
    target, target_norm = _normalized(target_points)
    target_tree = cKDTree(target)
    candidates = list(_pca_candidates(source, target))
    if use_features:
        feature = _feature_candidate(source, target)
        if feature is not None and np.isfinite(feature).all():
            candidates.append((feature, 'fpfh_ransac'))
    results = []
    for matrix, name in candidates:
        matrix = _refine(source, target, matrix, target_tree, 18)
        results.append((_score(source, target, matrix, target_tree), matrix, name))
    refined = []
    for _, matrix, name in sorted(results, key=lambda result: result[0])[:3]:
        matrix = _refine(source, target, matrix, target_tree, 100)
        refined.append((_score(source, target, matrix, target_tree), matrix, name))
    _, matrix, name = min(refined, key=lambda result: result[0])
    normalized_scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    if not np.isfinite(matrix).all() or not .2 <= normalized_scale <= 5:
        raise ValueError('Skala hasil registrasi di luar rentang pencarian; periksa cakupan objek.')
    forward, backward, _, _ = _distances(source, target, matrix, target_tree)
    overlaps = [float(np.mean(d < .08)) for d in (forward, backward)]
    if min(overlaps) < .35:
        raise ValueError('Penyelarasan otomatis belum memadai. Periksa cakupan patung/ROI '
                         'dan lihat model LiDAR serta rekonstruksi; gunakan kalibrasi manual bila perlu.')
    final = np.linalg.inv(target_norm) @ matrix @ source_norm
    scale = float(np.cbrt(np.linalg.det(final[:3, :3])))
    if not np.isfinite(final).all() or scale <= 0:
        raise ValueError('Transformasi otomatis tidak valid.')
    return final, {
        'alignment_algorithm': ALIGNMENT_VERSION, 'alignment_initialization': name,
        'scale_factor_to_meters': scale, 'registration_source_overlap': overlaps[0],
        'registration_target_overlap': overlaps[1], 'registration_overlap_threshold': .35,
        'registration_distance_normalized': .08,
        'registration_inlier_rmse_cm': float(np.sqrt(np.mean(forward[forward < .08] ** 2))
                                              / target_norm[0, 0] * 100),
        'alignment_review': 'automatic_requires_visual_review',
        'registration_points_source': len(source), 'registration_points_target': len(target),
        'alignment_transform': final.tolist(),
    }
