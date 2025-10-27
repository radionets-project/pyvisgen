from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from h5py import File

__all__ = ["H5Writer"]


class DataWriter(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def write(self, *args, **kwargs) -> None: ...

    def test_shapes(self, array, name) -> None:
        if array.shape[1] != 2:
            raise ValueError(
                f"Expected array {name} axis 1 to be 2 but got "
                f"{array.shape} with axis 1: {array.shape[1]}!"
            )

        if len(array.shape) != 4:
            raise ValueError(
                f"Expected array {name} shape to be of len 4 but got "
                f"{array.shape} with len {len(array.shape)}!"
            )

    def get_half_image(self, x, y) -> tuple[np.ndarray]:
        half_image = x.shape[2] // 2
        x = x[:, :, : half_image + 5, :]
        y = y[:, :, : half_image + 5, :]

        return x, y

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class H5Writer:
    def __init__(self, path) -> None:
        self.path = path

    def write(self, x, y, name_x="x", name_y="y") -> None:
        """
        write fft_pairs created in second analysis step to h5 file
        """
        x, y = self.get_half_image(x, y)

        self.test_shapes(x, "x")
        self.test_shapes(y, "y")

        with File(self.path, "w") as f:
            f.create_dataset(name_x, data=x)
            f.create_dataset(name_y, data=y)
            f.close()
