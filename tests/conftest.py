import torch

_original_compile = torch.compile


def _mock_compile(func, *args, **kwargs):
    return func


torch.compile = _mock_compile  # type: ignore
