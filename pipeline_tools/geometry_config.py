"""Per-scene geometry calibration and readiness checks without GPU imports."""
import json
import math
from contextlib import suppress
from pathlib import Path

from pipeline_tools.dataset_registry import PROJECT_ROOT, file_signature, scene_pairs


METHODS = ('3DGS', 'SuGaR')
LIDAR_UNITS = {'m': 1.0, 'cm': .01, 'mm': .001}
ALIGNMENT_VERSION = 'sim3_pca_fpfh_trimmed_v1'


def geometry_options(entry):
    if not isinstance(entry, dict):
        raise ValueError('Pengaturan geometri harus berupa objek JSON.')
    unit = entry.get('lidar_unit', '')
    mode = entry.get('alignment_mode', 'automatic')
    if not isinstance(unit, str) or unit not in LIDAR_UNITS:
        raise ValueError('Pilih satuan LiDAR: m, cm, atau mm. Jangan menebak satuan.')
    if not isinstance(mode, str) or mode not in ('automatic', 'manual'):
        raise ValueError('Penyelarasan harus automatic atau manual.')
    result = dict(lidar_unit=unit, alignment_mode=mode)
    if mode == 'manual':
        result['scale_factor'] = positive_scale(entry.get('scale_factor'))
    return result


def read_geometry_settings(root=PROJECT_ROOT):
    path = Path(root) / 'datasets/geometry_settings.json'
    if not path.is_file():
        return {}
    settings = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(settings, dict):
        raise ValueError('Pengaturan geometri harus berupa objek JSON.')
    for cls, models in settings.items():
        if not isinstance(models, dict):
            raise ValueError('Pengaturan geometri per kelas tidak valid.')
        for mdl, options in models.items():
            validate_target(cls, mdl)
            if not isinstance(options, dict):
                raise ValueError('Pengaturan geometri model tidak valid.')
            models[mdl] = geometry_options(options)
    return settings


def save_geometry_settings(entries, root=PROJECT_ROOT):
    if not isinstance(entries, list) or len(entries) > len(scene_pairs()):
        raise ValueError('Pengaturan geometri harus berupa daftar maksimal 18 model.')
    changes, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Pengaturan setiap model harus berupa objek JSON.')
        cls, mdl = entry.get('class_name'), entry.get('model_id')
        validate_target(cls, mdl)
        if (cls, mdl) in seen:
            raise ValueError('Model yang sama tidak boleh dikirim dua kali.')
        seen.add((cls, mdl))
        if entry.get('clear') is True:
            changes.append((cls, mdl, None))
        else:
            changes.append((cls, mdl, geometry_options(entry)))
    settings = read_geometry_settings(root)
    for cls, mdl, options in changes:
        if options is None:
            settings.get(cls, {}).pop(mdl, None)
            if cls in settings and not settings[cls]:
                del settings[cls]
        else:
            settings.setdefault(cls, {})[mdl] = options
    path = Path(root) / 'datasets/geometry_settings.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return settings


def positive_scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError('Skala kalibrasi harus berupa angka positif.')
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError('Skala kalibrasi harus berupa angka positif.') from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Skala kalibrasi harus finite dan lebih besar dari 0.')
    return result


def validate_target(cls, mdl):
    if not isinstance(cls, str) or not isinstance(mdl, str) or not cls or not mdl:
        raise ValueError('Kelas dan ID model wajib diisi untuk kalibrasi.')
    scene_pairs(cls, mdl)


def read_geometry_scales(root=PROJECT_ROOT):
    path = Path(root) / 'datasets/geometry_scales.json'
    if not path.is_file():
        return {}
    try:
        scales = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(scales, dict):
            raise ValueError('Kalibrasi harus berupa objek JSON.')
        for cls, models in scales.items():
            if not isinstance(models, dict):
                raise ValueError('Kalibrasi per kelas harus berupa objek JSON.')
            for mdl, value in models.items():
                validate_target(cls, mdl)
                models[mdl] = positive_scale(value)
        return scales
    except ValueError as exc:
        raise ValueError('Periksa datasets/geometry_scales.json: ' + str(exc)) from exc


def save_geometry_scales(entries, root=PROJECT_ROOT):
    if not isinstance(entries, list) or len(entries) > len(scene_pairs()):
        raise ValueError('Kalibrasi harus berupa daftar maksimal 18 model.')
    changes, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Kalibrasi setiap model harus berupa objek JSON.')
        cls, mdl = entry.get('class_name'), entry.get('model_id')
        validate_target(cls, mdl)
        if (cls, mdl) in seen:
            raise ValueError('Kalibrasi model yang sama tidak boleh dikirim dua kali.')
        seen.add((cls, mdl))
        value = entry.get('scale_factor')
        changes.append((cls, mdl, None if value is None else positive_scale(value)))
    scales = read_geometry_scales(root)
    for cls, mdl, value in changes:
        if value is not None:
            scales.setdefault(cls, {})[mdl] = value
        elif cls in scales:
            scales[cls].pop(mdl, None)
            if not scales[cls]:
                del scales[cls]
    path = Path(root) / 'datasets/geometry_scales.json'
    temporary = path.with_suffix('.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(json.dumps(scales, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return scales


def geometry_model_path(root, cls, mdl, method):
    if method not in METHODS:
        raise ValueError('Metode geometri harus 3DGS atau SuGaR.')
    output = Path(root) / 'outputs' / method.lower() / cls / mdl
    if method == '3DGS':
        return output / 'point_cloud/iteration_30000/point_cloud.ply'
    manifest = json.loads((output / 'pipeline_result.json').read_text(encoding='utf-8'))
    mesh = manifest['textured_mesh']
    if not isinstance(mesh, str):
        raise ValueError('Path mesh SuGaR tidak valid.')
    model = (output / mesh).resolve()
    if output.resolve() not in model.parents:
        raise ValueError('Mesh geometri harus berada di folder output model.')
    return model


def nonempty_file(path):
    return path.is_file() and path.stat().st_size > 0


def geometry_source_ready(root, cls, mdl, method):
    try:
        return nonempty_file(geometry_model_path(root, cls, mdl, method))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def geometry_result_current(root, cls, mdl, method, scale, options=None):
    if scale is None and options is None:
        return False
    try:
        output = Path(root) / 'outputs' / method.lower() / cls / mdl
        report = json.loads((output / 'geometry/metrics.json').read_text(encoding='utf-8'))
        gt = Path(root) / 'datasets' / cls / mdl / 'ground_truth/lidar.ply'
        metrics = ('chamfer_distance_cm',) + tuple(f'{name}_at_{threshold}cm_pct'
            for threshold in (2, 5) for name in ('precision', 'recall', 'f1'))
        return (report['model_signature'] == file_signature(geometry_model_path(root, cls, mdl, method))
                and report['gt_signature'] == file_signature(gt)
                and (report.get('geometry_options') == options if options is not None else
                     report['scale_factor_to_meters'] == scale)
                and (not options or options['alignment_mode'] != 'automatic' or
                     report.get('alignment_algorithm') == ALIGNMENT_VERSION)
                and all(math.isfinite(float(report[key])) for key in metrics))
    except (OSError, ValueError, KeyError, TypeError):
        return False
