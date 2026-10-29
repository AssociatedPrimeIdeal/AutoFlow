"""Build version-matched documentation that works directly from file:// URLs."""

import argparse
import hashlib
import io
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import tarfile
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PageResources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []
        self.links = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "pre" and "mermaid" in attributes.get("class", "").split():
            raise ValueError("Offline diagrams must use the local renderer, not the theme's CDN loader")
        if tag in {"script", "img", "source", "video", "audio", "iframe"}:
            self.resources.extend(attributes[key] for key in ("src", "poster") if attributes.get(key))
        if tag == "link" and set(attributes.get("rel", "").split()) & {
            "stylesheet", "icon", "preload", "modulepreload"
        } and attributes.get("href"):
            self.resources.append(attributes["href"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])


def validate_site(site):
    """Reject missing local files and remote resources needed for reading."""
    pages = sorted(site.rglob("*.html"))
    if not (site / "index.html").is_file() or not (site / "search/search_index.js").is_file():
        raise ValueError("Offline documentation is missing its home page or search index")

    def validate_reference(source, value, *, resource):
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc:
            if resource and parsed.scheme != "data":
                raise ValueError(f"Offline page depends on an external resource: {source}: {value}")
            return
        if not parsed.path:
            return
        target = (source.parent / unquote(parsed.path)).resolve()
        if target.is_dir():
            target = target / "index.html"
        if not target.is_relative_to(site) or not target.is_file():
            raise ValueError(f"Missing offline documentation file: {source}: {value}")

    for page in pages:
        parser = PageResources()
        parser.feed(page.read_text(encoding="utf-8"))
        for value in parser.resources:
            validate_reference(page, value, resource=True)
        for value in parser.links:
            validate_reference(page, value, resource=False)
    for stylesheet in site.rglob("*.css"):
        for value in re.findall(r"url\(\s*['\"]?([^)'\"]+)", stylesheet.read_text(encoding="utf-8")):
            validate_reference(stylesheet, value, resource=True)
    return len(pages)


def build(repo, output, cache):
    from mkdocs.commands.build import build as build_site
    from mkdocs.config import load_config

    repo, output, cache = repo.resolve(), output.resolve(), cache.resolve()
    if repo.is_relative_to(output) or any(output.is_relative_to(repo / name) for name in (
        "autoflow", "docs", "configs", "tests", "packaging", "third_party", ".git", ".github"
    )):
        raise ValueError("Documentation output must not overlap repository sources")
    if cache.is_relative_to(output):
        raise ValueError("Documentation asset cache must be outside the generated site")
    cache.mkdir(parents=True, exist_ok=True)
    lock_path = repo / "packaging/docs/assets.lock.json"
    assets = json.loads(lock_path.read_text(encoding="utf-8"))["assets"]
    cached_assets = []
    downloads = {}
    for item in assets:
        target = output / item["file"]
        if not target.resolve().is_relative_to(output):
            raise ValueError("Documentation asset path is outside the output folder")
        cached = cache / item["sha256"]
        if not cached.is_file() or digest(cached) != item["sha256"]:
            print(f"Downloading documentation asset: {item['file']}", flush=True)
            if item["url"] not in downloads:
                request = Request(item["url"], headers={"User-Agent": "AutoFlow-docs-builder"})
                with urlopen(request, timeout=30) as response:
                    downloads[item["url"]] = response.read()
            with tarfile.open(fileobj=io.BytesIO(downloads[item["url"]]), mode="r:gz") as archive:
                payload = archive.extractfile(item["member"]).read()
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Documentation asset checksum mismatch: {item['file']}")
            cached.write_bytes(payload)
        cached_assets.append((cached, target))

    config = load_config(config_file=str(repo / "mkdocs.offline.yml"), site_dir=str(output), strict=True)
    # The theme auto-loads Mermaid from a CDN for its own .mermaid blocks.
    # Give offline blocks to our bundled classic-script renderer instead.
    for fence in config.mdx_configs.get("pymdownx.superfences", {}).get("custom_fences", []):
        if fence.get("name") == "mermaid":
            fence["class"] = "autoflow-mermaid"
    config.extra_css.append("assets/stylesheets/offline.css")
    build_site(config)
    # HTTP 404 pages use root URLs and are never used by file:// navigation.
    (output / "404.html").unlink(missing_ok=True)
    for cached, target in cached_assets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
    shutil.copy2(repo / "packaging/docs/mermaid-offline.js", output / "assets/javascripts/mermaid-offline.js")
    shutil.copy2(repo / "packaging/docs/offline.css", output / "assets/stylesheets/offline.css")
    page_count = validate_site(output)
    source_files = sorted(path for path in (repo / "docs").rglob("*") if path.is_file())
    source_files += [repo / name for name in ("mkdocs.yml", "mkdocs.offline.yml")]
    source_files += sorted(path for path in (repo / "packaging/docs").rglob("*")
                           if path.is_file() and "__pycache__" not in path.parts)
    source_hashes = {path.relative_to(repo).as_posix(): digest(path) for path in source_files}
    manifest = {
        "schema_version": 1,
        "offline": True,
        "pages": page_count,
        "source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "assets": assets,
        "files": {path.relative_to(output).as_posix(): digest(path)
                  for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "offline-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "site": str(output), "pages": page_count,
                      "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file())}), flush=True)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assets-cache", type=Path)
    args = parser.parse_args()
    build(args.repo, args.output or args.repo / "build/docs-offline",
          args.assets_cache or args.repo / "build/docs-assets")


if __name__ == "__main__":
    main()
