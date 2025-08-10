from pathlib import Path

import natsort
import numpy as np
from astropy.io import fits
from numpy.typing import ArrayLike

__all__ = [
    "fits_data",
]


class fits_data:
    """Class that handles h5 files simulated
    with the radiosim package.

    Parameters
    ----------
    data_path: str or :class:`~pathlib.Path`
        path to fits data directory
    """

    def __init__(self, data_path: str | Path) -> None:
        """Handle h5 files simulated with the
        radiosim package.

        Parameters
        ----------
        data_path: str Path
            path to fits data directory
        """
        self.path = data_path
        self.files = self.get_files(Path(data_path))
        self.num_files = len(self.files)

    def __call__(self) -> None:
        return print("This is the pyvisgen fits files data set class.")

    def __len__(self) -> int:
        """
        Returns the total number of fits files in this dataset.

        Returns
        -------
        num_files : int
            Number of files in data set.
        """
        return self.num_files

    def __getitem__(self, i):
        """Returns the file at index ``i`` of :py:attr:`~self.files`."""
        return self.open_file(i)

    def get_files(self, path: Path) -> list[str]:
        """Returns a list of files in natural ordering.

        Parameters
        ----------
        path : Path
            Path to the data directory.

        Returns
        -------
        fits_files_sorted : list
            List of files in natural ordering.
        """
        fits_files = [str(x) for x in path.iterdir()]
        fits_files_sorted = natsort.natsorted(fits_files)

        return fits_files_sorted

    def get_uv_data(self, i: int) -> ArrayLike:
        """Loads uv data from the file at index ``i`` of
        :py:attr:`~self.files`.

        Parameters
        ----------
        i : int
            Index of the file in :py:attr:`self.files`.

        Returns
        -------
        uv_data : array_like
            Array of uv data.
        """
        with fits.open(self.files[i]) as hdul:
            uv_data = hdul[0].data

        return uv_data

    def get_freq_data(self, i: int) -> ArrayLike:
        """Loads frequency data from the file at index ``i`` of
        :py:attr:`~self.files`.

        Parameters
        ----------
        i : int
            Index of the file in :py:attr:`~self.files`.

        Returns
        -------
        freq_data : array_like
            Array of frequency data.
        base_freq : float
            Base frequency of the observing antenna array.
        """
        with fits.open(self.files[i]) as hdul:
            base_freq = hdul[0].header["CRVAL4"]
            freq_data = hdul[2].data

        return freq_data, base_freq

    def open_file(self, i: int) -> ArrayLike:
        """Opens the file and returns the file information.

        Parameters
        ----------
        i : int
            Index of the file in :py:attr:`~self.files`.

        Returns
        -------
        :py:func:`~astropy.io.fits.info`
            Summary information on the FITS file.
        """
        if isinstance(i, np.ndarray):
            print("Only one file can be open at the same time!")
            return

        with fits.open(self.files[i]) as hdul:
            fits_file = hdul

        return fits_file.info()
