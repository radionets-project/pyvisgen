import sysconfig
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import torch
from astropy.coordinates import EarthLocation


@dataclass
class Stations:
    """Stores station ID, x, y, and z positions,
    antenna diameter, minimum and maximum elevation,
    system equivalent flux density (SEFD) and altitude.

    Attributes
    ----------
    st_num : list[int]
        Station IDs.
    x : list[float]
        Geocentric x positions of the antennas.
    y : list[float]
        Geocentric y positions of the antennas.
    z : list[float]
        Geocentric z positions of the antennas.
    diam : list[float]
        Antenna dish diameters.
    el_low : list[float]
        Minimum possible elevation of the antennas.
    el_high : list[float]
        Maximum possible elevation of the antennas.
    sefd : list[int]
        System equivalent flux density.
    altitude : list[float]
        Altitude of the antennas.
    """

    st_num: list[int]
    x: list[float]
    y: list[float]
    z: list[float]
    diam: list[float]
    el_low: list[float]
    el_high: list[float]
    sefd: list[int]
    altitude: list[float]

    def __getitem__(self, i: int):
        """Returns fields at index ``i``.

        Parameters
        ----------
        i : int
            Index of the items in the dataclass.

        Returns
        -------
        Stations
            :class:`~pyvisgen.layouts.Stations` dataclass object
            containing elements at index ``i``.
        """
        return Stations(*[getattr(self, f.name)[i] for f in fields(self)])


def get_array_layout(
    array_layout: str | Path | pd.DataFrame,
    writer: bool = False,
) -> Stations:
    r"""Reads a telescope layout txt file and converts it
    into a dataclass. Also allows a DataFrame to be passed
    that is then converted into a dataclass object. This
    allows custom layouts to be used in the simulation.

    **Available built-in arrays:**

    +-----------------+------------------+
    | Experiment      | Layout Name      |
    +=================+==================+
    | ALMA            | ``alma``         |
    +-----------------+------------------+
    | ALMA (DSHARP)   | ``alma_dsharp``  |
    +-----------------+------------------+
    | DSA 2000W       | ``dsa2000W``     |
    +-----------------+------------------+
    | DSA 2000 31B    | ``dsa2000_31b``  |
    +-----------------+------------------+
    | EHT             | ``eht``          |
    +-----------------+------------------+
    | MeerKAT         | ``meerkat``      |
    +-----------------+------------------+
    | MeerKAT (test)* | ``meerkat_test`` |
    +-----------------+------------------+
    | VLA             | ``vla``          |
    +-----------------+------------------+
    | VLBA            | ``vlba``         |
    +-----------------+------------------+
    | VLBA (light)*   | ``vlba_light``   |
    +-----------------+------------------+

    \* reduced layouts for testing purposes


    Parameters
    ----------
    array_layout : str or :class:`~pathlib.Path` or :class:`~pandas.DataFrame`
        Name of telescope array or pd.DataFrame containing
        the array layout.
    writer : bool, optional
        If ``True``, return ``array`` :class:`~pandas.DataFrame`
        instead of :class:`~pyvisgen.layouts.Stations` dataclass
        object.

    Returns
    -------
    Stations
        :class:`~pyvisgen.layouts.Stations` dataclass comprising
        information on all stations in the given array layout.
    """
    if isinstance(array_layout, str):
        root = sysconfig.get_path("data", sysconfig.get_default_scheme())
        path = root + f"/share/resources/layouts/{array_layout}.txt"

        with open(path, "r") as f:
            array = pd.read_csv(f, sep=r"\s+")

        if array_layout == "vla":
            # Change relative positions to absolute positions
            # for the VLA layout
            loc = EarthLocation.of_site("VLA")
            array["X"] += loc.value[0]
            array["Y"] += loc.value[1]
            array["Z"] += loc.value[2]

    elif isinstance(array_layout, pd.DataFrame):
        array = array_layout
    elif isinstance(array_layout, Path):
        array = pd.read_csv(array_layout, sep=r"\s+")
    else:
        raise TypeError(
            "Expected array_layout to be of type str, "
            "pathlib.Path, or pandas.DataFrame!"
        )

    # drop name col and convert to tensor
    tensor = torch.from_numpy(array.iloc[:, 1:].values)
    # add st_num manually (station index)
    tensor = torch.cat([torch.arange(len(array))[..., None], tensor], dim=1)
    # swap axes for easy conversion into stations object
    tensor = tensor.swapaxes(0, 1)

    stations = Stations(*tensor)

    if writer:
        return array
    else:
        return stations


def get_array_names() -> list[str]:
    """Get list of names of built-in arrays for use with
    :class:`~pyvisgen.simulation.Observation` and
    various other methods.

    Returns
    -------
    names : list[str]
        Names of arrays available for use in pyvisgen.

    See Also
    --------
    :func:`~pyvisgen.layouts.get_array_layout` : Returns
        a :class:`~pyvisgen.layouts.Stations` dataclass object
        containing station informations for any array names
        returned by this function.
    """
    root = sysconfig.get_path("data", sysconfig.get_default_scheme())
    path = Path(root + "/share/resources/layouts/")
    return list(file.stem for file in path.glob("*.txt"))
