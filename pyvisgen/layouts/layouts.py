from dataclasses import dataclass
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
        if torch.is_tensor(i):
            return torch.stack([self.__getitem__(int(_i)) for _i in i])
        else:
            station = Station(
                self.st_num[i],
                self.x[i],
                self.y[i],
                self.z[i],
                self.diam[i],
                self.el_low[i],
                self.el_high[i],
                self.sefd[i],
                self.altitude[i],
            )
        return station

    def get_station(self, name):
        return self[torch.where(self.name == name)[0][0]]


@dataclass
class Station:
    st_num: int
    x: float
    y: float
    z: float
    diam: float
    el_low: float
    el_high: float
    sefd: int
    altitude: float


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
    array = pd.read_csv(file_dir / f, sep=" ")
    if array_name == "vla":
        loc = EarthLocation.of_site("VLA")
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
