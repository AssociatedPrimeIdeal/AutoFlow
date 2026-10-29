"""Small byte-budget LRU for display datasets; numerical arrays stay intact."""

from collections import OrderedDict
import numpy as np


def cache_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, str)):
        return len(value)
    if all(hasattr(value, key) for key in ("data", "indices", "indptr")):
        return sum(cache_bytes(getattr(value, key), seen) for key in ("data", "indices", "indptr"))
    if hasattr(value, "GetActualMemorySize"):
        return int(value.GetActualMemorySize()) * 1024
    if isinstance(value, dict):
        return sum(cache_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(cache_bytes(v, seen) for v in value)
    return 64


class ByteLRUCache(OrderedDict):
    def __init__(self, max_bytes=256 * 1024 ** 2):
        super().__init__()
        self.max_bytes = max(1, int(max_bytes))
        self.bytes_used = 0
        self._sizes = {}

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __setitem__(self, key, value):
        self.bytes_used -= self._sizes.pop(key, 0)
        super().__setitem__(key, value)
        self.move_to_end(key)
        self._sizes[key] = cache_bytes(value) + cache_bytes(key)
        self.bytes_used += self._sizes[key]
        # Retain at most one oversized visible dataset, so a just-built value
        # remains fetchable even when it exceeds the normal cache budget.
        while self.bytes_used > self.max_bytes and len(self) > 1:
            old, _ = super().popitem(last=False)
            self.bytes_used -= self._sizes.pop(old, 0)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.bytes_used -= self._sizes.pop(key, 0)

    def pop(self, key, *default):
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = super().__getitem__(key)
        del self[key]
        return value

    def clear(self):
        super().clear()
        self._sizes.clear()
        self.bytes_used = 0
