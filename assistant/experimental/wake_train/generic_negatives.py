"""Loader for the official openWakeWord generic negative-feature dataset
(davidscripka/openwakeword_features, validation_set_features.npy).

The file is a CONTIGUOUS stream of per-80ms embedding vectors, shape
(M, 96) -- ~11 hours at 0.08s/row -- not pre-chunked into (16, 96)
training windows. This reshapes it into non-overlapping 16-frame windows.

Also tolerates a PARTIAL/still-downloading file: reads only as many
complete rows as are actually present on disk (network here is slow and
shared with other local downloads, so training may need to proceed
before the full ~185MB file has finished landing).
"""
from __future__ import annotations

import os
import numpy as np
import numpy.lib.format as fmt


def load_generic_windows(path: str, frames: int = 16) -> np.ndarray:
    with open(path, "rb") as f:
        version = fmt.read_magic(f)
        shape, fortran, dtype = fmt._read_array_header(f, version)
        data_offset = f.tell()
    filesize = os.path.getsize(path)
    row_bytes = int(np.prod(shape[1:]) * dtype.itemsize)
    avail_rows = (filesize - data_offset) // row_bytes
    n_complete = int(min(shape[0], avail_rows))
    flat = np.memmap(path, dtype=dtype, mode="r", offset=data_offset,
                      shape=(n_complete,) + shape[1:])
    n_windows = n_complete // frames
    windows = np.asarray(flat[: n_windows * frames]).reshape(n_windows, frames, shape[1])
    return windows.astype(np.float32)
