# Changelog

[GitHub Releases](https://github.com/AssociatedPrimeIdeal/AutoFlow/releases) contains the changelog for each published version and its downloadable files. Each automatic application release lists non-merge commits since the preceding version, links each change to its commit, and includes a full code comparison.

Pushes to `main` or `dev` publish the next patch version after Linux/Windows regression tests and Windows portable verification pass. Pull requests and manual CI runs only build and verify. Publication failures stay in drafts until a successful retry.

## v2.0.0

Source and inference-model baseline. The current locked model resources were moved from `windows-models-20261106` to this normal version tag without changing model bytes or checksums. The obsolete `windows-models-20261017` release and both date-based tags were removed.

This baseline provides source and model resources. Windows portable application downloads start with the next successful push release, `v2.0.1`; see [Windows portable application](windows-portable.md).

Maintainers: see [Windows builds and releases](../developer/windows-packaging.md) for version selection, model assets, release notes, and publication checks.
