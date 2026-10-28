"""Geometry scores from bidirectional nearest-neighbour distances in metres.

Precision: fraction of reconstructed points within a strict distance threshold
of ground truth. Recall: fraction of ground-truth points covered by reconstruction.
Chamfer here is the mean of the two unsquared mean distances, reported in cm.
"""
import math

GEOMETRY_COLUMNS = {
    f'{metric}_at_{threshold}cm_pct': f'{label} @{threshold}cm (%)'
    for threshold in (2, 5)
    for metric, label in (('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1'))
}


def metrics_from_distances(source_to_target, target_to_source):
    source = [float(value) for value in source_to_target]
    target = [float(value) for value in target_to_source]
    if not source or not target:
        raise ValueError('Jarak geometri sumber dan ground truth tidak boleh kosong.')
    if any(not math.isfinite(value) or value < 0 for values in (source, target) for value in values):
        raise ValueError('Jarak geometri harus finite dan tidak negatif.')
    source_mean = math.fsum(source) / len(source)
    target_mean = math.fsum(target) / len(target)
    result = {
        'chamfer_distance_cm': round(50.0 * (source_mean + target_mean), 4),
        'chamfer_definition': 'half_sum_of_bidirectional_mean_unsquared_distances',
        'mean_accuracy_cm': round(100.0 * source_mean, 4),
        'mean_completeness_cm': round(100.0 * target_mean, 4),
        'source_evaluation_points': len(source), 'target_evaluation_points': len(target),
        'threshold_comparison': 'strict_less_than',
    }
    for threshold in (2, 5):
        precision = 100.0 * sum(distance < threshold / 100.0 for distance in source) / len(source)
        recall = 100.0 * sum(distance < threshold / 100.0 for distance in target) / len(target)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.update({f'precision_at_{threshold}cm_pct': round(precision, 4),
                       f'recall_at_{threshold}cm_pct': round(recall, 4),
                       f'f1_at_{threshold}cm_pct': round(f1, 4),
                       # Preserve the names used by older JSON consumers.
                       f'accuracy_at_{threshold}cm_pct': round(precision, 2),
                       f'completeness_at_{threshold}cm_pct': round(recall, 2)})
    return result
