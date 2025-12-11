from __future__ import annotations

import gc

import torch

OOM_EXCEPTIONS = (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError)
MIN_BATCH_SIZE = 1


def _cuda_gc():
    """Garbage collector for CUDA."""
    gc.collect()

    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.empty_cache()


def _reduce_batch_size(batch_size, factor=0.5):
    """Reduces the batch size by a factor"""
    return int(batch_size * 0.5)


def adaptive_batch_size(func, initial_batch_size, factor=0.5, *args, **kwargs):
    _cuda_gc()
    batch_size = initial_batch_size

    while True:
        try:
            result = func(batch_size, *args, **kwargs)
            _cuda_gc()
            return result
        except OOM_EXCEPTIONS:
            if batch_size >= MIN_BATCH_SIZE:
                batch_size = _reduce_batch_size(batch_size, factor=factor)
                _cuda_gc()
            else:
                raise
