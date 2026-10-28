"""Resolution validation, CLI options, and compatibility of cached stages."""
import ast
import json
import hashlib
from pathlib import Path
from pipeline_tools.dataset_registry import file_signature, has_colmap_model

DEFAULTS = {'colmap_max_image_size': 3200, 'image_downscale': 1, 'image_max_size': 1600}
COLMAP_DEFAULTS = {'colmap_device': 'cpu', 'colmap_matcher': 'exhaustive',
                   'colmap_max_features': 8192, 'colmap_sequential_overlap': 10}


def research_presets():
    root = Path(__file__).resolve().parents[1]
    references = json.loads((root / 'web_dashboard/research_presets.json').read_text(encoding='utf-8'))
    references['source']['local_code_sha256'] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('train.py', 'sugar_extractors/coarse_mesh.py', 'sugar_trainers/refine.py',
                     'pipeline_tools/04_train_sugar.py')}
    return references


def sugar_settings(data):
    # Historical defaults keep already queued jobs reproducible. New UI presets
    # always send all four parameters explicitly.
    result = {key: data.get(key, default) for key, default in {
        'vertices': 250000, 'gaussians_per_triangle': 1,
        'refinement_time': 'short', 'regularization': 'dn_consistency'}.items()}
    for key in ('vertices', 'gaussians_per_triangle'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f'{key} harus berupa bilangan bulat.')
        try:
            result[key] = int(value)
        except ValueError:
            raise ValueError(f'{key} harus berupa bilangan bulat.')
    if not 1000 <= result['vertices'] <= 2000000:
        raise ValueError('Parameter mesh harus 1000–2000000.')
    if result['gaussians_per_triangle'] not in (1, 3, 4, 6):
        raise ValueError('Gaussian per segitiga harus 1, 3, 4, atau 6 sesuai implementasi SuGaR.')
    if result['refinement_time'] not in ('short', 'medium', 'long'):
        raise ValueError('Refinement harus short, medium, atau long.')
    if result['regularization'] not in ('dn_consistency', 'density', 'sdf'):
        raise ValueError('Regularisasi harus dn_consistency, density, atau sdf.')
    return result


def sugar_arguments(data):
    return [arg for key, value in sugar_settings(data).items() for arg in ('--' + key, str(value))]


def colmap_settings(data):
    result = {key: data.get(key, default) for key, default in COLMAP_DEFAULTS.items()}
    if result['colmap_device'] not in ('cpu', 'cuda', 'auto'):
        raise ValueError('Perangkat COLMAP harus cpu, cuda, atau auto.')
    if result['colmap_matcher'] not in ('exhaustive', 'sequential'):
        raise ValueError('Matcher COLMAP harus exhaustive atau sequential.')
    for key, low, high in (('colmap_max_features', 1024, 32768), ('colmap_sequential_overlap', 2, 100)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f'{key} harus berupa bilangan bulat.')
        try:
            result[key] = int(value)
        except (ValueError, TypeError):
            raise ValueError(f'{key} harus berupa bilangan bulat.')
        if not low <= result[key] <= high:
            raise ValueError(f'{key} harus antara {low} dan {high}.')
    return result


def add_colmap_arguments(parser):
    parser.add_argument('--colmap_device', choices=('cpu', 'cuda', 'auto'), default=None)
    parser.add_argument('--colmap_matcher', choices=('exhaustive', 'sequential'), default='exhaustive')
    parser.add_argument('--colmap_max_features', type=int, default=8192)
    parser.add_argument('--colmap_sequential_overlap', type=int, default=10)


def colmap_arguments(settings):
    return [arg for key, value in colmap_settings(settings).items() for arg in ('--' + key, str(value))]


def resolution_settings(data):
    result = {}
    for key, default in DEFAULTS.items():
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f'{key} harus berupa bilangan bulat.')
        try:
            result[key] = int(value)
        except (ValueError, TypeError):
            raise ValueError(f'{key} harus berupa bilangan bulat.')
    if not 256 <= result['colmap_max_image_size'] <= 16384:
        raise ValueError('Ukuran COLMAP harus 256–16384 piksel.')
    if result['image_downscale'] not in (1, 2, 4, 8):
        raise ValueError('Downscale harus 1, 2, 4, atau 8.')
    if result['image_max_size'] != 0 and not 256 <= result['image_max_size'] <= 16384:
        raise ValueError('Batas training harus 0 (asli) atau 256–16384 piksel.')
    return result


def add_resolution_arguments(parser, colmap=False):
    if colmap:
        parser.add_argument('--colmap_max_image_size', type=int, default=3200)
    parser.add_argument('--image_downscale', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--image_max_size', type=int, default=1600,
                        help='Maximum longest image side; 0 keeps size after downscale.')


def colmap_signature(scene):
    sparse = Path(scene) / 'sparse/0'
    for ext in ('.bin', '.txt'):
        files = [sparse / (name + ext) for name in ('cameras', 'images', 'points3D')]
        if all(path.is_file() for path in files):
            return {path.name: file_signature(path) for path in files}
    return None


def colmap_matches(scene, max_size, options=None):
    if not has_colmap_model(scene):
        return False
    path = Path(scene) / 'colmap_settings.json'
    if not path.is_file():
        # Imported/legacy SfM can be reused at the historical default, with a notice.
        return max_size == 3200 and (options is None or colmap_settings(options) == COLMAP_DEFAULTS)
    try:
        settings = json.loads(path.read_text(encoding='utf-8'))
        return (settings['max_image_size'] == max_size and settings['model_signature'] == colmap_signature(scene)
                and (options is None or colmap_settings(settings) == colmap_settings(options)))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def read_training_config(output):
    tree = ast.parse((Path(output) / 'cfg_args').read_text(encoding='utf-8'), mode='eval')
    if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name) or tree.body.func.id != 'Namespace':
        raise ValueError('Format cfg_args 3DGS tidak valid.')
    return {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}


def training_matches(output, scene, downscale, max_size, white_background=False):
    try:
        settings = json.loads((Path(output) / 'training_settings.json').read_text(encoding='utf-8'))
        matches = (settings['image_downscale'] == downscale and settings['image_max_size'] == max_size
                   and settings['colmap_signature'] == colmap_signature(scene)
                   and settings.get('white_background', False) == white_background
                   and settings.get('eval_split', True) is True)
        if (Path(output) / 'cfg_args').is_file():
            config = read_training_config(output)
            matches = matches and config.get('eval') is True and config.get('white_background', False) == white_background
        return matches
    except (OSError, ValueError, KeyError, TypeError, SyntaxError):
        return False


def evaluation_matches(output, downscale, max_size):
    try:
        settings = json.loads((Path(output) / 'evaluation/metrics.json').read_text(encoding='utf-8'))
        return settings['image_downscale'] == downscale and settings['image_max_size'] == max_size
    except (OSError, ValueError, KeyError, TypeError):
        return False
