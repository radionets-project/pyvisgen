import torch
from astropy.utils.decorators import lazyproperty


class Array:
    def __init__(self, array_layout):
        self.array_layout = array_layout

    @lazyproperty
    def calc_relative_pos(self):
        # from geocentric coordinates to relative coordinates inside array
        delta_x, delta_y, delta_z = self.get_pairs
        return delta_x, delta_y, delta_z

    @lazyproperty
    def get_pairs(self):
        combs_x = torch.combinations(self.array_layout.x)
        delta_x = (combs_x[:, 0] - combs_x[:, 1]).reshape(-1, 1)

        combs_y = torch.combinations(self.array_layout.y)
        delta_y = (combs_y[:, 0] - combs_y[:, 1]).reshape(-1, 1)

        combs_z = torch.combinations(self.array_layout.z)
        delta_z = (combs_z[:, 0] - combs_z[:, 1]).reshape(-1, 1)

        return delta_x, delta_y, delta_z

    @lazyproperty
    def calc_ant_pair_vals(self):
        """Calculates station number, low elevation, and high
        elevation pairs.
        """
        st_num_pairs = torch.combinations(self.array_layout.st_num)
        els_low_pairs = torch.combinations(self.array_layout.el_low)
        els_high_pairs = torch.combinations(self.array_layout.el_high)

        return st_num_pairs, els_low_pairs, els_high_pairs
