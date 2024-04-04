import torch
from astropy.utils.decorators import lazyproperty


class Array:
    def __init__(self, array_layout):
        self.array_layout = array_layout

    @lazyproperty
    def calc_relative_pos(self):
        # from geocentric coordinates to relative coordinates inside array
        delta_x, delta_y, delta_z = self.get_pairs()
        self.indices = self.single_occurance(delta_x)
        delta_x = delta_x[self.indices]
        delta_y = delta_y[self.indices]
        delta_z = delta_z[self.indices]
        return delta_x, delta_y, delta_z, self.indices

    def get_pairs(self):
        delta_x = (
            torch.stack(
                [
                    val
                    - self.array_layout.x[
                        ~(torch.arange(len(self.array_layout.x)) == i)
                    ]
                    for i, val in enumerate(self.array_layout.x)
                ]
            )
            .ravel()
            .reshape(-1, 1)
        )
        delta_y = (
            torch.stack(
                [
                    val
                    - self.array_layout.y[
                        ~(torch.arange(len(self.array_layout.y)) == i)
                    ]
                    for i, val in enumerate(self.array_layout.y)
                ]
            )
            .ravel()
            .reshape(-1, 1)
        )
        delta_z = (
            torch.stack(
                [
                    val
                    - self.array_layout.z[
                        ~(torch.arange(len(self.array_layout.z)) == i)
                    ]
                    for i, val in enumerate(self.array_layout.z)
                ]
            )
            .ravel()
            .reshape(-1, 1)
        )
        return delta_x, delta_y, delta_z

    def single_occurance(self, tensor):
        # only calc one half of visibility because of Fourier symmetry
        vals, index = self.unique(torch.abs(tensor))
        return index

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
