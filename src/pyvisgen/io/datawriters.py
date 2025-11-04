import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import torch
from h5py import File

from pyvisgen.fits.writer import create_hdu_list

__all__ = [
    "DataWriter",
    "FITSWriter",
    "H5Writer",
    "PTWriter",
]


class DataWriter(ABC):
    """Abstract base class for data writers in pyvisgen.

    This class contains methods to get half images and
    test the shapes of the data prior to writing. It also
    supports a context manager protocol.

    Subclasses must implement the `__init__` and `write`
    methods to define writing behavior.

    Parameters
    ----------
    output_path : str or Path
        Path where the dataset will be written.
    dataset_type : str
        Type of dataset being written (e.g., 'train', 'test', 'validation').
    *args
        Variable length argument list passed to subclass implementations.
    **kwargs
        Arbitrary keyword arguments passed to subclass implementations.

    Examples
    --------
    >>> class MyWriter(DataWriter):
    ...     def __init__(self, output_path, dataset_type):
    ...         self.output_path = output_path
    ...         self.dataset_type = dataset_type
    ...
    ...     def write(self, data):
    ...         # Implementation here
    ...         pass
    >>>
    >>> with MyWriter("output_file", dataset_type="train") as writer:
    ...     writer.write(data)
    """

    @abstractmethod
    def __init__(self, output_path: Path, dataset_type: str, *args, **kwargs) -> None:
        """Initialize the data writer.

        This method must be implemented by subclasses to handle the setup
        of the context manager.

        Parameters
        ----------
        output_path : str or Path
            Path where the dataset will be written.
        dataset_type : str
            Type of dataset being written.
        *args
            Additional positional arguments for subclass-specific initialization.
        **kwargs
            Additional keyword arguments for subclass-specific initialization.
        """
        ...

    @abstractmethod
    def write(self, *args, **kwargs) -> None:
        """Write data to the output destination.

        This method must be implemented by subclasses to handle the actual
        writing of data to the specified output format.

        Parameters
        ----------
        *args
            Data and parameters required for writing, defined by subclass.
        **kwargs
            Additional options for writing, defined by subclass.
        """
        ...

    def test_shapes(self, array: np.ndarray, name: str) -> None:
        """Validate the shape of input arrays.

        Arrays should have the shape (B, C, H, W),
        where B is the batch size, C the number of channels
        (2), and W and H the width and height of the images.

        Parameters
        ----------
        array : np.ndarray
            Array to validate.
        name : str
            Name of the array for error reporting.

        Raises
        ------
        ValueError
            If array axis 1 is not size 2.
        ValueError
            If array does not have exactly 4 dimensions.
        """
        if array.shape[1] != 2:
            raise ValueError(
                f"Expected array {name} axis 1 to be 2 but got "
                f"{array.shape} with axis 1: {array.shape[1]}!"
            )

        if array.ndim != 4:
            raise ValueError(
                f"Expected array {name} ndim to be 4 but got "
                f"{array.shape} with ndim {array.ndim}!"
            )

    def get_half_image(
        self, x: np.ndarray, y: np.ndarray, overlap: int = 5
    ) -> tuple[np.ndarray]:
        """Extract half height of every image with a small overlap.

        Parameters
        ----------
        x : np.ndarray
            Simulated data array with shape (B, C, H, W).
        y : np.ndarray
            Ground truth array with shape (B, C, H, W).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple containing the cropped x and y arrays.
        """
        half_image = x.shape[2] // 2
        x = x[:, :, : half_image + overlap, :]
        y = y[:, :, : half_image + overlap, :]

        return x, y

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns
        -------
        Self
            The DataWriter instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager.

        Performs cleanup when exiting the context. Default implementation
        does nothing; subclasses can override to add cleanup logic.

        Parameters
        ----------
        exc_type : type or None
            The type of exception that occurred, if any.
        exc_value : Exception or None
            The exception instance that occurred, if any.
        traceback : traceback or None
            The traceback object for the exception, if any.

        Returns
        -------
        None
            Returns ``None`` per default.
        """
        return None


class H5Writer(DataWriter):
    """HDF5 file writer for pyvisgen datasets.

    This writer saves data arrays to HDF5 files using the h5py
    library. Each sample is written to a separate ``.h5`` file.
    The writer automatically crops images to half their height
    with a small overlap and validates array shapes before writing.

    Parameters
    ----------
    output_path : str or Path
        Directory path where HDF5 files will be written.
    dataset_type : str
        Type of dataset being written (e.g., 'train', 'test',
        'validation'). This is used in the output filename pattern.

    Examples
    --------
    >>> writer = H5Writer(output_path="./data", dataset_type="train")
    >>> writer.write(x_data, y_data, index=0)

    Or as a context manager:

    >>> rng = np.random.default_rng()
    >>>
    >>> with H5Writer(output_path="./data", dataset_type="train") as writer:
    ...     x_data = rng.uniform(size=(5, 10, 2, 256, 256))
    ...     y_data = rng.uniform(size=(5, 10, 2, 256, 256))
    ...
    ...     for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
    ...         writer.write(x, y, index=bundle_id)
    """

    def __init__(self, output_path: Path, dataset_type: str, **kwargs) -> None:
        """Initialize the HDF5 writer.

        Parameters
        ----------
        output_path : str or Path
            Directory path where HDF5 files will be written.
        dataset_type : str
            Type of dataset being written (e.g., 'train', 'test',
            'validation').
        """
        self.output_path = output_path
        self.dataset_type = dataset_type

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    def write(
        self, x, y, *, index, overlap=5, name_x="x", name_y="y", **kwargs
    ) -> None:
        """Write FFT pair data to an HDF5 file.

        Creates a new HDF5 file for each sample with pattern
        ``samp_{dataset_type}_{index}.h5``. The input arrays are cropped
        to half their height (with 5 pixel overlap) and validated
        before writing.

        Parameters
        ----------
        x : np.ndarray
            First array of the FFT pair with shape (batch, 2, height, width).
            Expected to have 4 dimensions with axis 1 of size 2.
        y : np.ndarray
            Second array of the FFT pair with shape (batch, 2, height, width).
            Expected to have 4 dimensions with axis 1 of size 2.
        index : int
            Bundle index used in the output filename.
        overlap : int, optional
            Overlap parameter for extracting half-images. Default: 5.
        name_x : str, optional
            Key of the dataset for x array in the HDF5 file. Default: ``"x"``.
        name_y : str, optional
            Key of the dataset for y array in the HDF5 file. Default: ``"y"``.

        Raises
        ------
        ValueError
            If x or y arrays don't have the expected shape (4 dimensions with
            axis 1 of size 2).

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>>
        >>> with H5Writer(output_path="./data", dataset_type="train") as writer:
        ...     x_data = rng.uniform(size=(5, 10, 2, 256, 256))
        ...     y_data = rng.uniform(size=(5, 10, 2, 256, 256))
        ...
        ...     for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
        ...         writer.write(x, y, index=bundle_id)
        """
        output_file = self.output_path / Path(
            f"samp_{self.dataset_type}_" + str(index) + ".h5"
        )

        x, y = self.get_half_image(x, y, overlap=overlap)

        self.test_shapes(x, "x")
        self.test_shapes(y, "y")

        with File(output_file, "w") as f:
            f.create_dataset(name_x, data=x)
            f.create_dataset(name_y, data=y)


class FITSWriter(DataWriter):
    """FITS file writer for pyvisgen visibility datasets.

    This writer saves visibility data and observation information
    to FITS (Flexible Image Transport System) files. Each sample
    is written to a separate ``.fits`` file.

    Parameters
    ----------
    output_path : str or Path
        Directory path where FITS files will be written.

    Examples
    --------
    >>> writer = FITSWriter(output_path="./data")
    >>> writer.write(vis_data, obs, index=0)

    Or as a context manager:

    >>> with FITSWriter(output_path="./data") as writer:
    ...     writer.write(vis_data, obs, index=0)
    """

    def __init__(self, output_path: Path, **kwargs) -> None:
        """Initialize the FITS writer.

        Parameters
        ----------
        output_path : str or Path
            Directory path where FITS files will be written.
        """
        self.output_path = output_path

    def write(
        self,
        vis_data,
        obs,
        index,
        overwrite=True,
        **kwargs,
    ) -> None:
        """Write visibility data and observation metadata to a FITS file.

        Creates a new FITS file for each sample with pattern
        ``vis_{dataset_type}_{index}.fits``.

        Parameters
        ----------
        vis_data : array-like
            Visibility data to be written to the FITS file.
        obs : object
            Observation metadata object from :class:`~pyvisgen.simulation.Observation`.
        index : int
            Sample index used in the output filename.
        overwrite : bool, optional
            If ``True``, overwrite the output file if it already exists,
            otherwise an error is raised.
            Default: ``True``.

        See Also
        --------
        pyvisgen.fits.writer.create_hdu_list : For more information on
            the parameters.

        Examples
        --------
        >>> writer = FITSWriter(output_path="./data")
        >>> writer.write(vis, obs, index=0)
        >>> # Creates file: ./data/vis_train_0.fits

        >>> writer.write(vis, obs, index=1, overwrite=False)
        >>> # Creates file: ./data/vis_train_1.fits (raises error if exists)
        """
        output_file = self.output_path / Path(
            f"vis_{self.conf.bundle.dataset_type}_" + str(index) + ".fits"
        )
        hdu_list = create_hdu_list(vis_data, obs)
        hdu_list.writeto(output_file, overwrite=overwrite)


class PTWriter(DataWriter):
    """DataWriter class for saving data in PyTorch (.pt) format.

    Creates a new .pt file for each sample with pattern
    ``samp_{dataset_type}_{index}.pt``. The input arrays are cropped
    to half their height (with ``overlap`` pixel overlap) and validated
    before writing.

    Parameters
    ----------
    output_path : Path
        Directory path where .pt files will be written.
    dataset_type : str
        Type of dataset being written (e.g., 'train', 'test', 'validation').
    amp_phase : bool
        If True, metadata ``TYPE`` key will contain 'amp_phase",
        otherwise 'real_imag'.

    Examples
    --------
    >>> writer = PTWriter(output_path="./data", dataset_type="train", amp_phase=True)
    >>> writer.write(x_data, y_data, index=0)

    Or as a context manager:

    >>> rng = np.random.default_rng()
    >>>
    >>> with PTWriter(
    ...     output_path="./data", dataset_type="train", amp_phase=True
    ... ) as writer:
    ...     x_data = rng.uniform(size=(5, 10, 2, 256, 256))
    ...     y_data = rng.uniform(size=(5, 10, 2, 256, 256))
    ...
    ...     for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
    ...         writer.write(x, y, index=bundle_id, bundle_length=len(x_data))
    """

    def __init__(
        self, output_path: Path, dataset_type: str, amp_phase: bool, **kwargs
    ) -> None:
        """Initialize the PT writer.

        Parameters
        ----------
        output_path : str or Path
            Directory path where .pt files will be written.
        dataset_type : str
            Type of dataset being written (e.g., 'train', 'test',
            'validation').
        amp_phase : bool
            If True, metadata key ``TYPE`` will contain 'amp_phase',
            otherwise 'real_imag'.
        """
        self.output_path = output_path
        self.dataset_type = dataset_type

        if amp_phase:
            self.data_type = "amp_phase"
        else:
            self.data_type = "real_imag"

    def write(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        index,
        bundle_length: int,
        overlap: int = 5,
        name_x: str = "X",
        name_y: str = "y",
        **kwargs,
    ) -> None:
        """Write data bundles to individual PyTorch (.pt) files.

        The input arrays are cropped to half their height (with
        ``overlap`` pixel overlap) and validated before writing
        as sparse tensors to .pt files.

        Parameters
        ----------
        x : np.ndarray
            First array of the FFT pair with shape (batch, 2, height, width).
            Expected to have 4 dimensions with axis 1 of size 2.
        y : np.ndarray
            Second array of the FFT pair with shape (batch, 2, height, width).
            Expected to have 4 dimensions with axis 1 of size 2.
        index : int
            Bundle index used in the output filename.
        bundle_length : int
            Number of samples to write in this bundle.
        overlap : int, optional
            Overlap parameter for extracting half-images. Default: 5.
        name_x : str, optional
            Key of the dataset for x array in the HDF5 file. Default: ``"X"``.
        name_y : str, optional
            Key of the dataset for y array in the HDF5 file. Default: ``"y"``.

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>>
        >>> with H5Writer(
        ...     output_path="./data", dataset_type="train", amp_phase=True
        ... ) as writer:
        ...     x_data = rng.uniform(size=(5, 10, 2, 256, 256))
        ...     y_data = rng.uniform(size=(5, 10, 2, 256, 256))
        ...
        ...     for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
        ...         writer.write(x, y, index=bundle_id, bundle_length=len(x))
        """
        x, y = self.get_half_image(x, y, overlap=overlap)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        self.test_shapes(x, "X")
        self.test_shapes(y, "y")

        x = x[:, 0] + 1j * x[:, 1]
        y = y[:, 0] + 1j * y[:, 1]

        for i in range(bundle_length):
            output_file = self.output_path / Path(
                f"samp_{self.dataset_type}_{index + i}.pt"
            )

            torch.save(
                obj={"SIM": x.to_sparse(), "TRUTH": y, "TYPE": self.data_type},
                f=output_file,
            )
