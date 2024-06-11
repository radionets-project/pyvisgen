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


def get_array_layout(array_name, writer=False):
    """Reads telescope layout txt file and converts it into a dataclass.
    Available arrays:
    - EHT

    Parameters
    ----------
    array_name : str
        Name of telescope array

    Returns
    -------
    dataclass objects
        Station infos combinde in dataclass
    """
    f = array_name + ".txt"
    array = pd.read_csv(file_dir / f, sep=r"\s+")
    if array_name == "vla":
        loc = EarthLocation.of_site("VLA")
        array["X"] += loc.value[0]
        array["Y"] += loc.value[1]
        array["Z"] += loc.value[2]

    if array_name == "test_layout":
        loc = EarthLocation.of_address("dortmund")
        array["X"] += loc.value[0]
        array["Y"] += loc.value[1]
        array["Z"] += loc.value[2]

    stations = Stations(
        torch.arange(len(array)),
        torch.tensor(array["X"].values),
        torch.tensor(array["Y"].values),
        torch.tensor(array["Z"].values),
        torch.tensor(array["dish_dia"].values),
        torch.tensor(array["el_low"].values),
        torch.tensor(array["el_high"].values),
        torch.tensor(array["SEFD"].values),
        torch.tensor(array["altitude"].values),
    )
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
