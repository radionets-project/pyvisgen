import torch
from astropy.utils.decorators import lazyproperty

__all__ = ["Array"]


class Array:
    """Class that handles antenna array operations such
    as calculating antenna pairs for baselines.

    Parameters
    ----------
    array_layout : :class:`~pyvisgen.layouts.Stations`
        :class:`~pyvisgen.layouts.Stations` dataclass object
        containing station data.
    """

    def __init__(self, array_layout):
        """Initializes the class with a given array layout.

        Parameters
        ----------
        array_layout : :class:`~pyvisgen.layouts.Stations`
            :class:`~pyvisgen.layouts.Stations` dataclass object
            containing station data.
        """
        self.array_layout = array_layout

    @lazyproperty
    def calc_relative_pos(self) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Calculates the relative positions of the antennas
        from geocentric coordinates.

        Returns
        -------
        delta_x : :func:`~torch.tensor`
            Relative x positions.
        delta_y : :func:`~torch.tensor`
            Relative y positions.
        delta_z : :func:`~torch.tensor`
            Relative z positions.
        """
        # from geocentric coordinates to relative coordinates inside array
        delta_x, delta_y, delta_z = self.get_pairs

        return delta_x, delta_y, delta_z

    @lazyproperty
    def get_pairs(self) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Computes antenna pair combinations and calculates
        the relative positions of the antennas.

        Returns
        -------
        delta_x : :func:`~torch.tensor`
            Relative x positions.
        delta_y : :func:`~torch.tensor`
            Relative y positions.
        delta_z : :func:`~torch.tensor`
            Relative z positions.
        """
        combs_x = torch.combinations(self.array_layout.x)
        delta_x = (combs_x[:, 0] - combs_x[:, 1]).reshape(-1, 1)

        combs_y = torch.combinations(self.array_layout.y)
        delta_y = (combs_y[:, 0] - combs_y[:, 1]).reshape(-1, 1)

        combs_z = torch.combinations(self.array_layout.z)
        delta_z = (combs_z[:, 0] - combs_z[:, 1]).reshape(-1, 1)

        return delta_x, delta_y, delta_z

    @lazyproperty
    def calc_ant_pair_vals(self) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Calculates station number, low elevation, and high
        elevation pairs.

        Returns
        -------
        st_num_pairs : :func:`~torch.tensor`
            Station number pair combinations.
        els_low_pairs : :func:`~torch.tensor`
            Station elevation pairs.
        els_high_pairs : :func:`~torch.tensor`
            Station elevation pairs.
        """
        st_num_pairs = torch.combinations(self.array_layout.st_num)
        els_low_pairs = torch.combinations(self.array_layout.el_low)
        els_high_pairs = torch.combinations(self.array_layout.el_high)

        return st_num_pairs, els_low_pairs, els_high_pairs
