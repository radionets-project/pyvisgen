from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import numpy as np


file_dir = Path(__file__).parent.resolve()


@dataclass
class Stations:
    st_num: [int]
    name: [str]
    x: [float]
    y: [float]
    z: [float]
    diam: [float]
    el_low: [float]
    el_high: [float]
    sefd: [int]
    altitude: [float]

    def __getitem__(self, i):
        if isinstance(i, np.ndarray):
            return [self.__getitem__(int(_i)) for _i in i]
        else:
            station = Station(
                self.st_num[i],
                self.name[i],
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
        return self[np.where(self.name == name)[0][0]]


@dataclass
class Station:
    st_num: int
    name: str
    x: float
    y: float
    z: float
    diam: float
    el_low: float
    el_high: float
    sefd: int
    altitude: float


def get_array_layout(array_name):
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
    stations = Stations(
        np.arange(len(array)),
        array["station_name"].values,
        array["X"].values,
        array["Y"].values,
        array["Z"].values,
        array["dish_dia"].values,
        array["el_low"].values,
        array["el_high"].values,
        array["SEFD"].values,
        array["altitude"].values,
    )
    return stations
