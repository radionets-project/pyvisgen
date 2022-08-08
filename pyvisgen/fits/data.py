import numpy as np
from astropy.io import fits
from pathlib import Path


class fits_data:
    def __init__(self, data_path):
        """Handle h5 files simulated with radiosim package.

        Parameters
        ----------
        data_path: str
            path to fits data directory

        """
        self.path = data_path
        self.files = self.get_files(Path(data_path))
        self.num_files = len(self.files)

    def __call__(self):
        return print("This is the pyvisgen fits files data set class.")

    def __len__(self):
        """
        Returns the total number of fits files in this dataset.
        """
        return self.num_files

    def __getitem__(self, i):
        return self.open_file(i)

    def get_files(self, path):
        return np.sort(np.array([x for x in path.iterdir()]))

    def get_uv_data(self, i):
        with fits.open(self.files[i]) as hdul:
            uv_data = hdul[0].data
            hdul.close()
            return uv_data

    def get_freq_data(self, i):
        with fits.open(self.files[i]) as hdul:
            base_freq = hdul[0].header["CRVAL4"]
            freq_data = hdul[2].data
            hdul.close()
            return freq_data, base_freq

    def open_file(self, i):
        if isinstance(i, np.ndarray):
            print("Only one file can be open at the same time!")
            return

        with fits.open(self.files[i]) as hdul:
            fits_file = hdul
            hdul.close()
            return fits_file.info()
