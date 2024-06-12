import torch
from astropy.utils.decorators import lazyproperty


class Array:
    def __init__(self, array_layout):
        self.array_layout = array_layout

    @lazyproperty
    def calc_relative_pos(self):
        # from geocentric coordinates to relative coordinates inside array
        delta_x, delta_y, delta_z = self.get_pairs()
        return delta_x, delta_y, delta_z

    def get_pairs(self):
        combs_x = torch.combinations(self.array_layout.x)
        delta_x = (combs_x[:, 0] - combs_x[:, 1]).reshape(-1, 1)

        combs_y = torch.combinations(self.array_layout.y)
        delta_y = (combs_y[:, 0] - combs_y[:, 1]).reshape(-1, 1)

        combs_z = torch.combinations(self.array_layout.z)
        delta_z = (combs_z[:, 0] - combs_z[:, 1]).reshape(-1, 1)
        return delta_x, delta_y, delta_z

    def unique(self, x, dim=0):
        unique, inverse, counts = torch.unique(
            x, dim=dim, sorted=True, return_inverse=True, return_counts=True
        )
        inv_sorted = inverse.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]
        index = index
        return unique, index

    @lazyproperty
    def get_baseline_mask(self):
        # mask baselines between the same telescope
        self.mask = [
            i * len(self.array_layout.x) + i for i in range(len(self.array_layout.x))
        ]
        return self.mask

    def delete(self, arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
        skip = [i for i in range(arr.size(dim)) if i != ind]
        indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
        return arr.__getitem__(indices)

    @lazyproperty
    def calc_ant_pair_vals(self):
        st_num_pairs = self.delete(
            arr=torch.stack(
                torch.meshgrid(self.array_layout.st_num, self.array_layout.st_num)
            )
            .swapaxes(0, 2)
            .reshape(-1, 2),
            ind=self.mask,
            dim=0,
        )[self.indices]

        els_low_pairs = self.delete(
            arr=torch.stack(
                torch.meshgrid(self.array_layout.el_low, self.array_layout.el_low)
            )
            .swapaxes(0, 2)
            .reshape(-1, 2),
            ind=self.mask,
            dim=0,
        )[self.indices]

        els_high_pairs = self.delete(
            arr=torch.stack(
                torch.meshgrid(self.array_layout.el_high, self.array_layout.el_high)
            )
            .swapaxes(0, 2)
            .reshape(-1, 2),
            ind=self.mask,
            dim=0,
        )[self.indices]
        return st_num_pairs, els_low_pairs, els_high_pairs
