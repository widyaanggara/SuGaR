"""Repair one migrated SuGaR manifest after verifying its referenced files.

Default is read-only. Stop all dashboard/training writers before using --apply.
Only baseline_signature in pipeline_result.json is changed; checkpoints are not.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def signature(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Path harus relatif dan tidak mengandung '..': " + str(relative))
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def repair(source, target, model, apply=False):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Sumber dan tujuan harus dua folder terpisah.")
    model = Path(model)
    parts = model.parts
    if not (len(parts) == 3 and parts[0] == "sugar" or
            len(parts) == 5 and parts[:3] == ("ablations", "scaffold_anchor", "sugar")):
        raise ValueError("Model harus sugar/Kelas/Model atau ablations/scaffold_anchor/sugar/Kelas/Model")
    old_manifest = inside(source, model / "pipeline_result.json")
    new_manifest = inside(target, model / "pipeline_result.json")
    original = old_manifest.read_bytes()
    current = new_manifest.read_bytes()
    if original != current:
        raise ValueError("Catatan sumber dan F sudah berbeda. Tidak ada perubahan dilakukan.")
    data = json.loads(original)
    baseline = model.parent.parent.parent / "3dgs" / parts[-2] / parts[-1] / "point_cloud/iteration_7000/point_cloud.ply"
    old_baseline, new_baseline = inside(source, baseline), inside(target, baseline)
    if data.get("baseline_signature") != signature(old_baseline):
        raise ValueError("Catatan asli tidak cocok dengan checkpoint cadangan; tidak boleh diperbaiki otomatis.")
    keys = ("coarse_mesh", "refined_checkpoint",
            "textured_mesh" if data.get("textured_mesh") else "surface_mesh")
    relatives = [baseline] + [model / data[key] for key in keys]
    verified = []
    for relative in dict.fromkeys(relatives):
        old, new = inside(source, relative), inside(target, relative)
        old_stat, new_stat = signature(old), signature(new)
        if old_stat["size"] <= 0 or old_stat["size"] != new_stat["size"]:
            raise ValueError("Ukuran tidak cocok/kosong: " + str(relative))
        print("Memeriksa SHA-256: " + str(relative), flush=True)
        if digest(old) != digest(new):
            raise ValueError("ISI FILE BERBEDA: " + str(relative))
        if signature(old) != old_stat or signature(new) != new_stat:
            raise ValueError("File berubah selama pemeriksaan. Hentikan semua penulis output.")
        verified.extend(((old, old_stat), (new, new_stat)))
    if data["baseline_signature"] == signature(new_baseline):
        print("Metadata baseline sudah cocok; tidak perlu perubahan.")
        return
    print("Semua file rujukan cocok. Hanya baseline_signature perlu diperbarui.")
    if not apply:
        print("DRY RUN: tidak ada file yang diubah. Gunakan --apply setelah meninjau hasil.")
        return
    if old_manifest.read_bytes() != original or new_manifest.read_bytes() != current:
        raise ValueError("Catatan berubah selama pemeriksaan.")
    if any(signature(path) != saved for path, saved in verified):
        raise ValueError("File berubah setelah pemeriksaan.")
    backup_dir = Path(tempfile.mkdtemp(prefix="backup-repair-sugar-", dir=target.parent))
    (backup_dir / "pipeline_result.before.json").write_bytes(current)
    if (backup_dir / "pipeline_result.before.json").read_bytes() != current:
        raise OSError("Verifikasi backup gagal.")
    data["baseline_signature"] = signature(new_baseline)
    fd, staging = tempfile.mkstemp(prefix=".manifest-repair-", dir=new_manifest.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, new_manifest)
    print("BERHASIL. Cadangan catatan: " + str(backup_dir))
    print("Tidak ada checkpoint/mesh yang diubah. Evaluasi dan status resume belum diperbaiki oleh alat ini.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        repair(args.source, args.target, args.model, args.apply)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "STOP: " + str(error) + "\n")
