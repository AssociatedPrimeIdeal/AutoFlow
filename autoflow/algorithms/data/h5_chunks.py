"""Bounded parallel decoding of large gzip HDF5 datasets."""

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import os
import zlib

import numpy as np

from ...task_control import check_cancelled


def read_compressed_dataset(dataset, max_workers=8):
    """Use HDF5 for I/O and decode independent gzip/shuffle chunks in threads.

    Unsupported layouts/filters retain HDF5's normal reader. Only the caller
    touches the h5py handle; at most two chunks per worker are queued, keeping
    compressed payload memory bounded. File contents are never changed.
    """
    if (dataset.chunks is None or dataset.compression != "gzip"
            or dataset.fletcher32 or dataset.scaleoffset is not None
            or dataset.dtype.hasobject or dataset.nbytes < 32 * 1024**2
            or not hasattr(dataset.id, "read_direct_chunk")):
        return np.asarray(dataset[...])
    properties = dataset.id.get_create_plist()
    filters = [properties.get_filter(i)[0] for i in range(properties.get_nfilters())]
    if any(value not in (1, 2) for value in filters):
        return np.asarray(dataset[...])
    workers = max(1, min(int(max_workers), os.cpu_count() or 1))
    chunk_shape, itemsize = dataset.chunks, dataset.dtype.itemsize
    output = np.empty(dataset.shape, dtype=dataset.dtype)

    def decode(region, filter_mask, payload):
        for index in reversed(range(len(filters))):
            if filter_mask & (1 << index):
                continue
            if filters[index] == 1:
                payload = zlib.decompress(payload)
            else:
                payload = np.frombuffer(payload, dtype=np.uint8).reshape(itemsize, -1).T.copy().tobytes()
        chunk = np.frombuffer(payload, dtype=output.dtype).reshape(chunk_shape)
        source = tuple(slice(0, part.stop - part.start) for part in region)
        output[region] = chunk[source]

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="autoflow-h5") as executor:
        pending = deque()
        for region in dataset.iter_chunks():
            check_cancelled()
            offset = tuple(part.start for part in region)
            info = dataset.id.get_chunk_info_by_coord(offset)
            # Unallocated chunks use the dataset fill value, not raw storage.
            if info.byte_offset is None:
                output[region] = dataset.fillvalue
                continue
            filter_mask, payload = dataset.id.read_direct_chunk(offset)
            pending.append(executor.submit(decode, region, filter_mask, payload))
            if len(pending) >= 2 * workers:
                pending.popleft().result()
        for future in pending:
            check_cancelled()
            future.result()
    check_cancelled()
    return output
