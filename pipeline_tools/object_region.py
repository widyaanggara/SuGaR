"""Object bounds in the original COLMAP coordinates; independent of training."""
import json
import math
from pathlib import Path


def validate_region(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Batas objek harus berupa objek JSON.')
    if value.get('mode') == 'camera_focus_v1':
        return {'mode': 'camera_focus_v1'}
    result = {}
    for key in ('min', 'max'):
        vector = value.get(key)
        if not isinstance(vector, (list, tuple)) or len(vector) != 3:
            raise ValueError('Batas objek memerlukan min/max dengan tiga koordinat X, Y, Z.')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
            raise ValueError('Koordinat batas objek harus angka finite.')
        result[key] = list(vector)
    if any(a >= b for a, b in zip(result['min'], result['max'])):
        raise ValueError('Setiap batas minimum harus lebih kecil dari maksimum.')
    threshold = value.get('min_triangles', 50)
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ValueError('Ambang fragmen harus bilangan bulat >= 0.')
    result['min_triangles'] = threshold
    if 'colmap_signature' in value:
        result['colmap_signature'] = value['colmap_signature']
    return result


def read_region(scene):
    path = Path(scene) / 'object_region.json'
    return validate_region(json.loads(path.read_text(encoding='utf-8'))) if path.is_file() else None


def resolve_region(scene, serialized):
    return read_region(scene) if serialized is None else validate_region(json.loads(serialized))


def save_region(scene, value):
    value = validate_region(value)
    if value is not None:
        from pipeline_tools.run_settings import colmap_signature
        signature = colmap_signature(scene)
        if signature is None:
            raise ValueError('Siapkan COLMAP terlebih dahulu sebelum menyimpan batas patung.')
        value['colmap_signature'] = signature
    path = Path(scene) / 'object_region.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        path.unlink(missing_ok=True)
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    return value


def object_arguments(region):
    return ['--object_region_json', json.dumps(region, separators=(',', ':'))]
