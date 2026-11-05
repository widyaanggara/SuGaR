"""Editable display names; dataset IDs and artifact paths remain stable."""
import json
from contextlib import suppress
from pathlib import Path
import unicodedata

from pipeline_tools.dataset_registry import PROJECT_ROOT, scene_pairs


def default_model_name(model_id):
    if not model_id:
        return ''
    if model_id.startswith('model_'):
        return 'Model ' + model_id[6:]
    return 'Benchmark ' + model_id.rsplit('_', 1)[-1]


def validate_model_name(value):
    if not isinstance(value, str):
        raise ValueError('Nama togog harus berupa teks.')
    if any(unicodedata.category(char) == 'Cc' for char in value):
        raise ValueError('Nama togog harus satu baris tanpa karakter kontrol.')
    value = value.strip()
    if len(value) > 80:
        raise ValueError('Nama togog maksimal 80 karakter.')
    return value


def _validate_target(class_name, model_id):
    if not isinstance(class_name, str) or not isinstance(model_id, str) or not class_name or not model_id:
        raise ValueError('Pilih satu kelas dan satu model untuk mengubah nama.')
    scene_pairs(class_name, model_id)


def read_model_names(root=PROJECT_ROOT):
    path = Path(root) / 'datasets/model_names.json'
    if not path.is_file():
        return {}
    try:
        names = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(names, dict):
            raise ValueError('Daftar nama harus berupa objek JSON.')
        for cls, models in names.items():
            if not isinstance(models, dict):
                raise ValueError('Daftar model harus berupa objek JSON.')
            for mdl, name in models.items():
                _validate_target(cls, mdl)
                models[mdl] = validate_model_name(name)
        return names
    except ValueError as exc:
        raise ValueError('Periksa datasets/model_names.json: ' + str(exc)) from exc


def model_display_name(class_name, model_id, names):
    return names.get(class_name, {}).get(model_id) or default_model_name(model_id)


def save_model_name(class_name, model_id, display_name, root=PROJECT_ROOT):
    _validate_target(class_name, model_id)
    value = validate_model_name(display_name)
    names = read_model_names(root)
    if value:
        names.setdefault(class_name, {})[model_id] = value
    elif class_name in names:
        names[class_name].pop(model_id, None)
        if not names[class_name]:
            del names[class_name]
    path = Path(root) / 'datasets/model_names.json'
    temporary = path.with_suffix('.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(json.dumps(names, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return dict(class_name=class_name, model_id=model_id, custom_name=value,
                display_name=model_display_name(class_name, model_id, names))
