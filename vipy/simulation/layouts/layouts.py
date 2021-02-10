import numpy as np
from pathlib import Path
import pandas as pd
from dataclasses import dataclass


file_dir = Path(__file__).parent.resolve()

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
    

def array(array_name):
    f = array_name + '.txt'
    array = pd.read_csv(file_dir / f, sep=' ')
    stations = [Station(index, row['station_name'], row['X'], row['Y'], row['Z'], row['dish_dia'], row['el_low'], row['el_high'], row['SEFD'], row['altitude']) for index, row in array.iterrows()]
    return stations

