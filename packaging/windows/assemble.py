"""Assemble the current working sources with an application-local CPython runtime."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import zipfile


IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".pytest_cache", "*.egg-info", "segmodel")
SUBPROJECTS = ("4DFlow_Dicom2H5", "PUDIP-Flow", "GUST-Flow", "SpatioTemporalLabeler")


def copy_if_changed(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.is_file():
        before, after = source.stat(), destination.stat()
        if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
            return str(destination)
    return shutil.copy2(source, destination)


def runtime_ignore(directory, names):
    ignored = {name for name in names if name == "__pycache__" or name.endswith(".pyc")}
    if Path(directory).name.endswith(".dist-info") and "licenses" in names:
        ignored.add("licenses")
    return ignored


def copy_licenses(source_site, target_site):
    # New Torch wheels contain deeply nested third-party license paths that can
    # exceed MAX_PATH even under a short portable folder. Keep every notice,
    # flatten nested names, and update License-File entries with a path index.
    for source_info in source_site.glob("*.dist-info"):
        licenses = source_info / "licenses"
        if not licenses.is_dir():
            continue
        target_info = target_site / source_info.name
        target_licenses = target_info / "licenses"
        if target_licenses.exists():
            if not target_licenses.resolve().is_relative_to(target_site.resolve()):
                raise ValueError("License destination is outside the portable runtime")
            shutil.rmtree(target_licenses)
        target_licenses.mkdir(parents=True)
        mapping = {}
        for source in sorted(licenses.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(licenses).as_posix()
            if "/" in relative:
                token = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
                short = f"third-party/{token}.notice"
            else:
                short = relative
            target = target_info / "licenses" / short
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_if_changed(source, target)
            mapping[relative] = short
        metadata = target_info / "METADATA"
        if metadata.is_file():
            text = (source_info / "METADATA").read_text(encoding="utf-8")
            for original, short in mapping.items():
                text = text.replace(f"License-File: {original}\n", f"License-File: {short}\n")
            metadata.write_text(text, encoding="utf-8")
        (target_info / "licenses" / "portable-path-index.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        record = target_info / "RECORD"
        if record.is_file():
            rows = list(csv.reader((source_info / "RECORD").read_text(encoding="utf-8").splitlines()))
            prefix = f"{source_info.name}/licenses/"
            for row in rows:
                if row[0].startswith(prefix):
                    old_name = row[0][len(prefix):]
                    if old_name in mapping:
                        row[0] = prefix + mapping[old_name]
                if row[0].startswith(prefix) or row[0] == f"{source_info.name}/METADATA":
                    changed = target_site / row[0]
                    row[1] = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(digest(changed))).decode("ascii").rstrip("=")
                    row[2] = str(changed.stat().st_size)
            index = target_info / "licenses" / "portable-path-index.json"
            rows.append([prefix + index.name, "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(digest(index))).decode("ascii").rstrip("="), str(index.stat().st_size)])
            with record.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(rows)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(repo, *arguments):
    if not (repo / ".git").exists():
        return "unversioned" if arguments[0] == "rev-parse" else ""
    result = subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def validate_models(models):
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("inference_only") is not True:
        raise ValueError("Use prepare_models.py to create the inference-only model bundle")
    required = {"nnunet3d", "nnunet4d", "nnunet3d-bundled"}
    profiles = {profile["profile"]: profile for profile in manifest["profiles"]}
    if required - profiles.keys():
        raise ValueError(f"Missing model profiles: {sorted(required - profiles.keys())}")
    for name in required:
        folder = models / name
        profile = profiles[name]
        if digest(folder / "dataset.json") != profile["dataset_sha256"] or digest(folder / "plans.json") != profile["plans_sha256"]:
            raise ValueError(f"Model metadata checksum mismatch: {name}")
        for item in profile["checkpoints"]:
            path = folder / item["file"]
            if not path.resolve().is_relative_to(folder.resolve()) or digest(path) != item["sha256"]:
                raise ValueError(f"Model checkpoint checksum mismatch: {name}")
        if not (folder / "fold_all" / "checkpoint_final.pth").is_file():
            raise ValueError(f"Missing final checkpoint: {name}")
    return manifest


def validate_documentation(documentation):
    manifest = json.loads((documentation / "offline-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("offline") is not True or not manifest.get("files"):
        raise ValueError("Build the offline documentation with packaging/docs/build_offline.py")
    for relative in ("index.html", "search/search_index.js"):
        if relative not in manifest["files"]:
            raise ValueError(f"Missing offline documentation entry: {relative}")
    for relative, expected in manifest["files"].items():
        path = documentation / relative
        if not path.resolve().is_relative_to(documentation.resolve()) or digest(path) != expected:
            raise ValueError(f"Offline documentation checksum mismatch: {relative}")
    return manifest


def stage(repo, destination, models, version, documentation, resume=False):
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("Build with Windows x64 CPython 3.11")
    validate_documentation(documentation)
    validate_models(models)
    destination.mkdir(parents=True, exist_ok=True)
    # Extended paths allow assembly without changing Windows' global long-path
    # policy. Deep dependency notices are additionally shortened for extraction.
    runtime = Path("\\\\?\\" + str(destination / "_runtime"))
    runtime.mkdir(exist_ok=resume)
    base = Path(sys.base_prefix)
    for path in base.iterdir():
        if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".pyd"}:
            copy_if_changed(path, runtime / path.name)
    for name in ("DLLs", "Lib"):
        if (base / name).is_dir():
            shutil.copytree(base / name, runtime / name,
                            dirs_exist_ok=resume, copy_function=copy_if_changed,
                            ignore=shutil.ignore_patterns("site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter", "ensurepip"))
    for name in ("LICENSE.txt", "LICENSE"):
        if (base / name).is_file():
            shutil.copy2(base / name, runtime / name)
    site = runtime / "Lib" / "site-packages"
    source_site = Path(sysconfig.get_path("purelib"))
    shutil.copytree(source_site, site, ignore=runtime_ignore,
                    dirs_exist_ok=resume, copy_function=copy_if_changed)
    copy_licenses(source_site, site)
    packaging = repo / "packaging" / "windows"
    shutil.copy2(packaging / "sitecustomize.py", site / "sitecustomize.py")
    shutil.copy2(packaging / "gpu_resampling.py", site / "nnunetv2" / "preprocessing" / "resampling" / "gpu_resampling.py")
    # An explicit path file prevents accidental use of a system Python install,
    # user packages or PYTHONPATH, while enabling the bundled sitecustomize.
    (runtime / "python311._pth").write_text(".\nDLLs\nLib\nLib/site-packages\n../app\nimport site\n", encoding="ascii")
    app = destination / "app"
    # Resume is for an interrupted runtime copy. Application snapshots must
    # still be clean so deleted source/config files cannot survive a rebuild.
    if resume:
        for name in ("app", "configs", "models", "docs", "source"):
            old = destination / name
            if old.exists():
                if not old.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Snapshot destination is outside the portable folder")
                shutil.rmtree(old)
    app.mkdir()
    shutil.copytree(repo / "autoflow", app / "autoflow", ignore=IGNORED)
    for name in ("portable_entry.py", "portable_check.py"):
        shutil.copy2(packaging / name, app / name)
    shutil.copytree(repo / "configs", destination / "configs", ignore=IGNORED)
    shutil.copytree(models, destination / "models", ignore=IGNORED)
    shutil.copytree(repo / "docs" / "en", destination / "docs" / "en", ignore=IGNORED)
    shutil.copy2(repo / "docs" / "index.md", destination / "docs" / "index.md")
    shutil.copytree(documentation, destination / "docs" / "site", ignore=IGNORED)
    shutil.copy2(repo / "LICENSE", destination / "LICENSE")
    shutil.copy2(repo / "README.md", destination / "README.md")
    # Preserve complete application/build sources, including dirty submodule code.
    source = destination / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "mkdocs.yml", "mkdocs.offline.yml"):
        shutil.copy2(repo / name, source / name)
    for name in ("autoflow", "configs", "packaging", ".github", "docs"):
        shutil.copytree(repo / name, source / name, ignore=IGNORED)
    for project in SUBPROJECTS:
        src, target = repo / "third_party" / project, source / "third_party" / project
        shutil.copytree(src / "src", target / "src", ignore=IGNORED)
        for name in ("pyproject.toml", "README.md", "LICENSE", "LICENSE.txt"):
            if (src / name).is_file():
                shutil.copy2(src / name, target / name)
    metadata = {
        "schema_version": 1, "version": version, "platform": "windows-x64", "entry_point": "gui",
        "python": sys.version, "git_head": git(repo, "rev-parse", "HEAD"),
        "working_tree_dirty": bool(git(repo, "status", "--porcelain")),
        "submodules": {name: {"head": git(repo / "third_party" / name, "rev-parse", "HEAD"),
                              "dirty": bool(git(repo / "third_party" / name, "status", "--porcelain"))}
                       for name in SUBPROJECTS},
        "models_manifest_sha256": digest(models / "manifest.json"),
        "documentation_manifest_sha256": digest(documentation / "offline-manifest.json"),
        "packages": sorted([{"name": item.metadata["Name"], "version": item.version}
                            for item in importlib.metadata.distributions()], key=lambda item: item["name"].lower()),
    }
    (destination / "portable.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    hashes = {path.relative_to(source).as_posix(): digest(path) for path in sorted(source.rglob("*")) if path.is_file()}
    (destination / "source-manifest.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(f"Assembled current working sources at {destination}", flush=True)


def archive(destination, output, max_part_bytes=1900 * 1024 * 1024):
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f"{destination.name}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zipped:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                zipped.write(path, path.relative_to(destination.parent).as_posix())
    finalize_archive(archive_path, output, max_part_bytes)


def finalize_archive(archive_path, output, max_part_bytes=1900 * 1024 * 1024):
    with zipfile.ZipFile(archive_path) as verification:
        bad = verification.testzip()
        if bad:
            raise ValueError(f"ZIP CRC verification failed: {bad}")
    print("ZIP CRC verification passed", flush=True)
    checksum = digest(archive_path)
    (output / f"{archive_path.name}.sha256").write_text(f"{checksum}  {archive_path.name}\n", encoding="ascii")
    # GitHub limits each release asset to 2 GiB. Preserve a standard ZIP locally;
    # release very large builds in byte-identical parts with a PowerShell joiner.
    limit = max_part_bytes
    parts = []
    if archive_path.stat().st_size > limit:
        with archive_path.open("rb") as stream:
            index = 1
            while stream.tell() < archive_path.stat().st_size:
                part = output / f"{archive_path.name}.part{index:03d}"
                remaining = limit
                with part.open("wb") as target:
                    while remaining:
                        block = stream.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            break
                        target.write(block)
                        remaining -= len(block)
                part_hash = digest(part)
                parts.append({"file": part.name, "sha256": part_hash, "bytes": part.stat().st_size})
                index += 1
    manifest = {"archive": archive_path.name, "sha256": checksum, "bytes": archive_path.stat().st_size, "parts": parts}
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive_path), "sha256": checksum, "bytes": archive_path.stat().st_size}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "archive"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--docs-site", type=Path)
    parser.add_argument("--version", default="dev")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume a runtime copy with unchanged dependencies")
    args = parser.parse_args()
    if args.action == "stage":
        if args.docs_site is None:
            parser.error("stage requires --docs-site from packaging/docs/build_offline.py")
        stage(args.repo.resolve(), args.destination.resolve(), args.models.resolve(), args.version,
              args.docs_site.resolve(), args.resume)
    else:
        archive(args.destination.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
