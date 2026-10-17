"""Compress a staged portable folder received as a binary TAR stream.

Optional server-side counterpart for native Windows tar. No installed AutoFlow
dependencies are needed; compression and validation use the standard library.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
import zipfile

from assemble import finalize_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Final archive already exists: {output}")
    staged = output.with_suffix(output.suffix + ".partial")
    progress = output.parent / "build-progress.json"
    count = total = 0
    last = 0.0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as stream, zipfile.ZipFile(
        staged, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as target:
        for member in stream:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or name.parts[0] != output.stem:
                raise ValueError(f"Unexpected portable archive member: {member.name}")
            if "__pycache__" in name.parts or name.suffix == ".pyc":
                continue
            info = zipfile.ZipInfo(name.as_posix(), date_time=time.localtime(max(member.mtime, 315532800))[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (member.mode & 0xFFFF) << 16
            with stream.extractfile(member) as source, target.open(info, "w", force_zip64=True) as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            count += 1
            total += member.size
            if time.monotonic() - last > 10:
                progress.write_text(json.dumps({"phase": "compressing", "files": count,
                    "uncompressed_bytes": total, "compressed_bytes": staged.stat().st_size}), encoding="utf-8")
                last = time.monotonic()
    staged.replace(output)
    progress.write_text(json.dumps({"phase": "checking_zip", "files": count,
        "uncompressed_bytes": total, "compressed_bytes": output.stat().st_size}), encoding="utf-8")
    finalize_archive(output, output.parent)
    shutil.copy2(Path(__file__).with_name("Join-Portable.ps1"), output.parent / "Join-Portable.ps1")
    progress.write_text(json.dumps({"phase": "complete", "files": count,
        "uncompressed_bytes": total, "compressed_bytes": output.stat().st_size}), encoding="utf-8")
    print(json.dumps({"zip_verified": True, "files": count, "uncompressed_bytes": total}), flush=True)


if __name__ == "__main__":
    main()
