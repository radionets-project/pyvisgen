from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
from h5py import File

from pyvisgen.fits.writer import create_hdu_list

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
    def __init__(self, output_path, dataset_type) -> None:
        self.output_path = output_path
        self.dataset_type = dataset_type

    def write(self, x, y, index, name_x="x", name_y="y") -> None:
        """
        write fft_pairs created in second analysis step to h5 file
        """
        output_file = self.output_path / Path(
            f"samp_{self.dataset_type}_" + str(index) + ".h5"
        )

        x, y = self.get_half_image(x, y)

        self.test_shapes(x, "x")
        self.test_shapes(y, "y")

        with File(output_file, "w") as f:
            f.create_dataset(name_x, data=x)
            f.create_dataset(name_y, data=y)


class FITSWriter:
    def __init__(self, output_path, dataset_type) -> None:
        self.output_path = output_path
        self.dataset_type = dataset_type

    def write(
        self,
        vis_data,
        obs,
        index,
        overwrite=True,
    ):
        output_file = self.output_path / Path(
            f"vis_{self.conf.bundle.dataset_type}_" + str(index) + ".fits"
        )
        hdu_list = create_hdu_list(vis_data, obs)
        hdu_list.writeto(output_file, overwrite=overwrite)
