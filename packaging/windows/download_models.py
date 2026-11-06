"""Fetch and verify the locked inference ZIP, including split release assets."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote
from urllib.request import Request, urlopen


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_models(lock_path, cache, repo=None):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    def asset_path(name):
        if not name or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Invalid model asset name: {name}")
        return cache / name

    archive = asset_path(lock["asset"])
    if archive.is_file() and sha256(archive) == lock["sha256"]:
        return archive
    parts = lock.get("parts") or [{"file": lock["asset"], "sha256": lock["sha256"]}]
    names = [item["file"] for item in parts]
    if len(names) != len(set(names)) or (lock.get("parts") and lock["asset"] in names):
        raise ValueError("Invalid or duplicate model archive parts")

    def fetch(item):
        path = asset_path(item["file"])
        if path.is_file() and sha256(path) == item["sha256"]:
            return path
        temporary = path.with_name(path.name + ".download")
        print(f"Downloading locked model asset: {path.name}", flush=True)
        if repo:
            # CI has gh and a read-only token; ordinary local builds need neither.
            subprocess.run(["gh", "release", "download", lock["tag"], "--repo", repo,
                            "--pattern", path.name, "--output", str(temporary), "--clobber"], check=True)
        else:
            url = lock["url"] if not lock.get("parts") else (
                lock["url"].rsplit("/", 1)[0] + "/" + quote(path.name)
            )
            with urlopen(Request(url, headers={"User-Agent": "AutoFlow-model-builder"}), timeout=60) as source:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if sha256(temporary) != item["sha256"]:
            temporary.unlink()
            raise ValueError(f"Model asset checksum mismatch: {path.name}")
        temporary.replace(path)
        return path

    with ThreadPoolExecutor(max_workers=4) as workers:
        downloaded = list(workers.map(fetch, parts))
    if lock.get("parts"):
        temporary = archive.with_name(archive.name + ".partial")
        with temporary.open("wb") as target:
            for part in downloaded:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if sha256(temporary) != lock["sha256"]:
            temporary.unlink()
            raise ValueError("Joined model archive checksum mismatch")
        temporary.replace(archive)
        for part in downloaded:
            part.unlink()
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("models.lock.json"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--repo", help="Use GitHub CLI authentication in CI")
    args = parser.parse_args()
    print(download_models(args.lock, args.cache, args.repo), flush=True)


if __name__ == "__main__":
    main()
