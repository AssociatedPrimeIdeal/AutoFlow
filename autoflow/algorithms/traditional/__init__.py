"""Bundled traditional phase-unwrapping backends.

The implementation is derived from the MIT-licensed PUDIP-Flow
``TradMethod`` sources.  Keeping it as a package avoids fragile runtime
``sys.path`` mutations while preserving the original three algorithms.
"""

from .flowunwrap import unwrap_data

__all__ = ["unwrap_data"]
