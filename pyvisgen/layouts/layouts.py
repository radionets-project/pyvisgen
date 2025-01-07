from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import torch
from astropy.coordinates import EarthLocation

file_dir = Path(__file__).parent.resolve()


@dataclass
class Stations:
    st_num: [int]
    x: [float]
    y: [float]
    z: [float]
    diam: [float]
    el_low: [float]
    el_high: [float]
    sefd: [int]
    altitude: [float]

    def __getitem__(self, i):
        return Stations(*[getattr(self, f.name)[i] for f in fields(self)])


def get_array_layout(array_layout: str | Path | pd.DataFrame, writer: bool = False):
    """Reads telescope layout txt file and converts it into a dataclass.
    Also allows a DataFrame to be passed that is then converted into a dataclass
    object.
    Available arrays:
    - EHT

    Parameters
    ----------
    array_layout : str or pathlib.Path or pd.DataFrame
        Name of telescope array or pd.DataFrame containing
        the array layout.
    writer : bool, optional
        If ``True``, return ``array`` DataFrame instead of
        ``Stations`` dataclass object.

    Returns
    -------
    dataclass objects
        Station infos combined in dataclass
    """
    if isinstance(array_layout, str):
        f = array_layout + ".txt"
        array = pd.read_csv(file_dir / f, sep=r"\s+")

        if array_layout == "vla":
            # Change relative positions to absolute positions
            # for the VLA layout
            loc = EarthLocation.of_site("VLA")
            array["X"] += loc.value[0]
            array["Y"] += loc.value[1]
            array["Z"] += loc.value[2]

    elif isinstance(array_layout, pd.DataFrame):
        array = array_layout
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
    """Get list of names of arrays for use with
    `~pyvisgen.simulation.Observation` and various
    other methods.

    Returns
    -------
    names : list[str]
        Names of arrays available for use in pyvisgen.

    See Also
    --------
    get_array_layout : Gets the locations of the telescopes
        for one of the array names this returns.
    """
    return list(file.stem for file in file_dir.glob("*.txt"))
