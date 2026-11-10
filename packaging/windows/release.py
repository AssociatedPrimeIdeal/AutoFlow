"""Select plain vMAJOR.MINOR.PATCH releases and publish verified CI artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib


VERSION = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def run(*args):
    return subprocess.check_output(args, text=True, encoding="utf-8").strip()


def releases(repo):
    pages = json.loads(run("gh", "api", "--paginate", "--slurp",
                           f"repos/{repo}/releases?per_page=100"))
    return [release for page in pages for release in page]


def version_tuple(tag):
    match = VERSION.fullmatch(tag)
    return tuple(map(int, match.groups())) if match else None


def marker(commit):
    return f"<!-- autoflow-commit: {commit} -->"


def next_version(base, tags, previous, commit):
    # A retry resumes its own draft or returns the already published release.
    matching = [item["tag_name"] for item in previous
                if version_tuple(item["tag_name"]) is not None
                and marker(commit) in (item.get("body") or "")]
    if matching:
        return max(matching, key=version_tuple)
    floor = version_tuple("v" + base)
    if floor is None:
        raise ValueError("project.version must be MAJOR.MINOR.PATCH")
    known = [value for tag in [*tags, *(item["tag_name"] for item in previous)]
             if (value := version_tuple(tag)) is not None]
    latest = max(known, default=None)
    chosen = floor if latest is None or floor > latest else (*latest[:2], latest[2] + 1)
    return "v" + ".".join(map(str, chosen))


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def artifact_files(directory, version):
    """Check the actual distributable bytes before making a release visible."""
    manifest = json.loads((directory / "release-manifest.json").read_text(encoding="utf-8"))
    archive = f"AutoFlow-{version}-windows-x64.zip"
    if manifest["archive"] != archive:
        raise ValueError("Portable archive does not match the selected release version")
    parts = manifest.get("parts") or []
    entries = parts or [{"file": archive, "sha256": manifest["sha256"], "bytes": manifest["bytes"]}]
    if parts:
        expected = [f"{archive}.part{index:03d}" for index in range(1, len(parts) + 1)]
        if [item["file"] for item in parts] != expected:
            raise ValueError("Invalid portable archive part order")
    joined = hashlib.sha256()
    total = 0
    for item in entries:
        if Path(item["file"]).name != item["file"]:
            raise ValueError("Invalid portable asset name")
        path = directory / item["file"]
        part_hash = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                part_hash.update(chunk)
                joined.update(chunk)
                total += len(chunk)
        if path.stat().st_size != item["bytes"] or part_hash.hexdigest() != item["sha256"]:
            raise ValueError(f"Portable asset checksum/size mismatch: {path.name}")
    if total != manifest["bytes"] or joined.hexdigest() != manifest["sha256"]:
        raise ValueError("Joined portable archive checksum/size mismatch")
    checksum = directory / f"{archive}.sha256"
    if checksum.read_text(encoding="utf-8").split() != [manifest["sha256"], archive]:
        raise ValueError("Portable checksum file does not match the manifest")
    names = ["release-manifest.json", "Join-Portable.ps1", checksum.name, "build-check.json",
             *(item["file"] for item in entries)]
    files = [directory / name for name in names]
    if not all(path.is_file() for path in files):
        raise ValueError("Portable release is missing required files")
    return files


def check_uploads(release, files):
    assets = {item["name"]: item for item in release["assets"]}
    for path in files:
        asset = assets.get(path.name)
        if not asset or asset["state"] != "uploaded" or asset["size"] != path.stat().st_size:
            raise ValueError(f"Release upload missing or incomplete: {path.name}")
        if asset.get("digest") != "sha256:" + digest(path):
            raise ValueError(f"Release upload checksum mismatch: {path.name}")


def release_notes(repo, commit, version, previous):
    published = [item["tag_name"] for item in previous if not item["draft"]
                 and version_tuple(item["tag_name"]) is not None
                 and version_tuple(item["tag_name"]) < version_tuple(version)]
    start = max(published, key=version_tuple) if published else None
    revision = f"{start}..{commit}" if start else commit
    history = run("git", "log", "--no-merges", "--format=%H%x09%s", revision)
    changes = []
    for line in history.splitlines():
        sha, subject = line.split("\t", 1)
        changes.append(f"- {subject} ([{sha[:8]}](https://github.com/{repo}/commit/{sha}))")
    # Leave room for download instructions within GitHub's release-body limit.
    listing = "\n".join(changes) or "- No additional non-merge commits."
    if len(listing) > 50000:
        listing = listing[:50000].rsplit("\n", 1)[0] + "\n- See the full changelog for remaining commits."
    comparison = (f"https://github.com/{repo}/compare/{start}...{version}" if start else
                  f"https://github.com/{repo}/commits/{version}")
    return (f"## Changes\n\n{listing}\n\n[Full changelog]({comparison})\n\n"
            "## Windows download\n\n"
            "Download the ZIP, or every `.zip.partNNN` file together with "
            "`release-manifest.json` and `Join-Portable.ps1`. For split downloads, run "
            "`Join-Portable.ps1`. Extract the ZIP and launch `AutoFlow.exe`.\n\n"
            "Linux/Windows regression tests and portable verification passed.\n\n"
            f"Source: https://github.com/{repo}/commit/{commit}\n\n{marker(commit)}\n")


def publish(repo, commit, version, directory):
    if version_tuple(version) is None:
        raise ValueError("Release version must be vMAJOR.MINOR.PATCH")
    files = artifact_files(directory, version)
    previous = releases(repo)
    existing = next((item for item in previous if item["tag_name"] == version), None)
    if existing:
        if marker(commit) not in (existing.get("body") or ""):
            raise ValueError("Refusing to replace a release from another commit")
        if not existing["draft"]:
            # ZIP timestamps can differ on rebuilds; keep a published release immutable.
            print(existing["html_url"])
            return
    else:
        notes = release_notes(repo, commit, version, previous)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "notes.md"
            path.write_text(notes, encoding="utf-8")
            run("gh", "release", "create", version, "--repo", repo, "--target", commit,
                "--title", version, "--notes-file", str(path), "--draft")
    run("gh", "release", "upload", version, "--repo", repo, "--clobber", *map(str, files))
    release = next(item for item in releases(repo) if item["tag_name"] == version)
    check_uploads(release, files)
    # Publish only after all assets are present and verified. Failed uploads stay drafts.
    run("gh", "release", "edit", version, "--repo", repo,
        "--draft=false", "--prerelease=false", "--latest")
    print(f"https://github.com/{repo}/releases/tag/{version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "publish"])
    parser.add_argument("--output", type=Path, help="GitHub Actions output file for plan")
    parser.add_argument("--version", help="Planned version for publish")
    parser.add_argument("--artifacts", type=Path, help="Verified portable artifact directory")
    args = parser.parse_args()
    repo, commit = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_SHA"]
    if args.action == "plan":
        project = Path(__file__).resolve().parents[2] / "pyproject.toml"
        base = tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"]
        version = next_version(base, run("git", "tag", "--list").splitlines(), releases(repo), commit)
        if args.output:
            with args.output.open("a", encoding="utf-8") as output:
                output.write(f"version={version}\n")
        print(version)
    else:
        if not args.version or not args.artifacts:
            parser.error("publish requires --version and --artifacts")
        publish(repo, commit, args.version, args.artifacts)


if __name__ == "__main__":
    main()
