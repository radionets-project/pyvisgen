"""Simple default gridder as fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.constants as const
import numpy as np

if TYPE_CHECKING:
    from pyvisgen.simulation import Observation


@dataclass
class GriddedData:
    mask_real: np.ndarray | None = None
    mask_imag: np.ndarray | None = None

    def get_mask_real_imag(self):
        return self.mask_real, self.mask_imag


class DefaultGridder:
    @classmethod
    def pyvisgen(
        cls,
        vis_data,
        obs: Observation,
        img_size: int,
        fov: float,
        stokes_components: int = "I",
        polarizations: str | None = None,
    ):
        instance = cls()
        instance.vis_data = vis_data
        instance.obs = obs
        instance.img_size = img_size
        instance.fov = fov

        # NOTE: stokes_comp and polarization are not implemented
        # and only kept for compatibility reasons. The default
        # gridder will always grid Stokes I regardless
        instance.stokes_comp = stokes_components
        instance.polarization = polarizations

        instance.grid_data = GriddedData()

        return instance

    def grid(self):
        uu = self.vis_data.u
        vv = self.vis_data.v
        uu /= const.c
        vv /= const.c

        freq = self.obs.ref_frequency.cpu().numpy()

        u = uu * np.array(freq)
        v = vv * np.array(freq)

        vis = self.vis_data.get_values()

        if vis.shape[-2] < 4:
            raise ValueError(
                "Expected shape at index -2 to be 4 for vis_data.get_values()!"
            )

        real = vis[..., 0, 0] + vis[..., 3, 0]
        imag = vis[..., 0, 1] + vis[..., 3, 1]
        real = real.real.ravel()
        imag = imag.real.ravel()

        if isinstance(u, complex):
            u = u.real
        if isinstance(v, complex):
            v = v.real

        samps = np.array(
            [
                np.append(-u, u),
                np.append(-v, v),
                np.append(real, real),
                np.append(-imag, imag),
            ]
        )
        # Generate Mask
        N = self.img_size  # image size
        fov = np.deg2rad(self.fov / 3600)

        delta = 1 / fov

        # bins are shifted by delta/2 so that maximum in uv space matches maximum
        # in numpy fft
        bins = (
            np.arange(
                start=-(N / 2) * delta,
                stop=(N / 2 + 1) * delta,
                step=delta,
            )
            - delta / 2
        )

        mask, *_ = np.histogram2d(samps[1], samps[0], bins=[bins, bins], density=False)
        mask[mask == 0] = 1

        mask_real, x_edges, y_edges = np.histogram2d(
            samps[1], samps[0], bins=[bins, bins], weights=samps[2], density=False
        )
        mask_imag, x_edges, y_edges = np.histogram2d(
            samps[1], samps[0], bins=[bins, bins], weights=samps[3], density=False
        )

        mask_real /= mask
        mask_imag /= mask

        self.grid_data.mask_real = mask_real
        self.grid_data.mask_imag = mask_imag

        return self.grid_data
