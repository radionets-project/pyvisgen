from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import astropy.units as un
import numpy as np
import torch
from astropy.coordinates import EarthLocation
from h5py import File
from natsort import natsorted

from pyvisgen.layouts import layouts
from pyvisgen.simulation.visibility import Visibilities

__all__ = [
    "UVH5Baselines",
    "UVH5Data",
    "UVH5Observation",
    "UVH5Reader",
    "read_uvh5",
]


@dataclass
class UVH5Baselines:
    """Baseline information stored in a UVH5 file.

    Stand-in for :class:`~pyvisgen.simulation.observation.Baselines`
    and the get_valid_subset method, so that code calling
    ``obs.baselines.get_valid_subset(...)`` works unchanged.

    Attributes
    ----------
    time : :class:`~torch.Tensor`, shape ``(n_vis,)``
        Time of each visibility row in JD.
    st_id_pairs : :class:`~torch.Tensor`, shape ``(n_vis, 2)``
        Station indices of each row.
    """

    time: torch.Tensor
    st_id_pairs: torch.Tensor

    def get_valid_subset(
        self,
        num_baselines,
        device: str | torch.device = "cpu",
    ) -> "UVH5Baselines":
        """Return the stored rows and move them to ``device``.

        The UVH5 file format already contains only valid baselines, so no selection
        is done. ``num_baselines`` is accepted for API compatibility with
        :meth:`~pyvisgen.simulation.observation.Baselines.get_valid_subset`
        and is thus ignored.
        """
        return UVH5Baselines(
            time=self.time.to(device), st_id_pairs=self.st_id_pairs.to(device)
        )


@dataclass
class UVH5Observation:
    """Observation metadata stored in a UVH5 file.

    Provides the subset of :class:`~pyvisgen.simulation.Observation`
    attributes that :meth:`pyvisgrid.Gridder.from_pyvisgen` uses.

    Attributes
    ----------
    ra, dec : :class:`~torch.Tensor`
        Pointing coordinates of the array.
    ref_frequency : :class:`~torch.Tensor`
        Reference frequency in Hz (i.e. the first frequency stored).
    frequency_offsets : :class:`~torch.Tensor`
        Offsets of each spectral window from ``ref_frequency`` in Hz.
    bandwidths : :class:`~torch.Tensor`
        Width of each spectral window in Hz.
    layout : str or :class:`~pathlib.Path` or None
        Name of a built-in array layout or path to a layout file.
    lm : :class:`~torch.Tensor`
        Direction cosines of the image grid, shape ``(H, W, 2)``.
    baselines : :class:`UVH5Baselines`
        Times and station pairs of the visibility rows.
    normalize : bool
        Whether the visibilities were normalised when written.

    Notes
    -----
    Start time, scan durations, noise and beam settings are not
    stored in the UVH5 file format, so a full :class:`~pyvisgen.simulation.Observation`
    cannot be rebuilt.
    """

    ra: torch.Tensor
    dec: torch.Tensor
    ref_frequency: torch.Tensor
    frequency_offsets: torch.Tensor
    bandwidths: torch.Tensor
    layout: str | Path | None
    lm: torch.Tensor
    baselines: UVH5Baselines
    normalize: bool
    _array: layouts.Stations | None = field(default=None, repr=False)

    @property
    def num_baselines(self) -> int:
        """Number of baselines between the stations in the layout.

        Uses the layout if it can be loaded, else the stations that
        appear in the file.
        """
        try:
            n_st = len(self.array.st_num)
        except (ValueError, FileNotFoundError):
            n_st = int(torch.unique(self.baselines.st_id_pairs).numel())

        return n_st * (n_st - 1) // 2

    @property
    def array(self) -> layouts.Stations:
        """Array layout, loaded from :attr:`layout`.

        Raises
        ------
        ValueError
            If the file stores no layout and none was passed to the reader.
        """
        if self._array is None:
            if self.layout is None:
                raise ValueError(
                    "The UVH5 file contains no array layout. Pass ``layout`` to the "
                    "reader to use obs.array."
                )
            layout = self.layout
            # Check if the a stored string is a path to a layout file
            if isinstance(layout, str) and Path(layout).is_file():
                layout = Path(layout)

            self._array = layouts.get_array_layout(layout)  # ty: ignore[invalid-assignment]

        return self._array  # ty: ignore[invalid-return-type]

    @cached_property
    def array_earth_loc(self) -> EarthLocation:
        """Geocentric station positions of :attr:`array`."""
        return EarthLocation.from_geocentric(
            self.array.x,
            self.array.y,
            self.array.z,
            unit=un.m,
        )


@dataclass
class UVH5Data:
    """Contents of a single UVH5 file.

    Attributes
    ----------
    vis : :class:`~pyvisgen.simulation.Visibilities`
        Visibilities as returned by :func:`~pyvisgen.simulation.visibility.vis_loop`.
    obs : :class:`UVH5Observation`
        Observation metadata.
    sky : :class:`~torch.Tensor` or None
        Simulated sky ``(C, H, W)`` if it was written, else ``None``.
    path : :class:`~pathlib.Path`
        File the data was read from.
    """

    vis: Visibilities
    obs: UVH5Observation
    sky: torch.Tensor | None
    path: Path

    @property
    def img_size(self) -> int | None:
        """Side length of the stored sky image in pixels, if available."""
        return None if self.sky is None else int(self.sky.shape[-1])


def read_uvh5(path: str | Path, layout: str | Path | None = None) -> UVH5Data:
    """Read a UVH5 file written by :class:`~pyvisgen.io.datawriters.UVH5Writer`.

    The returned ``vis`` and ``obs`` can be passed to the
    :meth:`~pyvisgrid.Gridder.from_pyvisgen` gridder method from pyvisgrid.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        Path to the ``.uvh5`` file.
    layout : str or :class:`~pathlib.Path` or None, optional
        Array layout to use instead of the one stored in the file, e.g.
        when the file was written by an older pyvisgen version or the
        layout was a custom file that has been moved. Default: ``None``.

    Returns
    -------
    :class:`UVH5Data`
        Visibilities, observation metadata and, if stored, the sky image.

    Raises
    ------
    KeyError
        If the file has no ``obs`` group, so the observation data such
        as the pointing is unknown.

    Notes
    -----
    ``num``, ``linear_dop`` and ``circular_dop`` are not stored in the
    UVH5 file format. ``num`` is rebuilt as ``1 … n_vis`` and ``base_num`` as
    ``256 (a1 + 1) + a2 + 1`` from the station pairs, as in
    :meth:`~pyvisgen.simulation.observation.Baselines.get_valid_subset`.
    The degrees of polarisation (dop) are returned as empty tensors.

    Examples
    --------
    >>> from pyvisgrid.core.gridder import Gridder
    >>> data = read_uvh5("train_0.uvh5")
    >>> grid_data = Gridder.from_pyvisgen(
    ...     vis_data=data.vis,
    ...     obs=data.obs,
    ...     img_size=data.img_size,
    ...     fov=fov,
    ... ).grid()
    """
    path = Path(path)

    with File(path, "r") as f:
        if "obs" not in f:
            raise KeyError(
                f"{path} has no 'obs' group, so observation data like the pointing "
                "direction is unknown."
            )

        vis_grp = f["visibilities"]
        uvw_grp = f["uvw"]

        V_11 = torch.from_numpy(vis_grp["V_11"][()])
        st_id_pairs = torch.from_numpy(uvw_grp["st_id_pairs"][()]).long()
        date = torch.from_numpy(np.asarray(f["times"][()], dtype=np.float64))
        n_vis = V_11.shape[0]

        vis = Visibilities(
            V_11=V_11,
            V_22=torch.from_numpy(vis_grp["V_22"][()]),
            V_12=torch.from_numpy(vis_grp["V_12"][()]),
            V_21=torch.from_numpy(vis_grp["V_21"][()]),
            weights=torch.from_numpy(vis_grp["weights"][()]),
            num=torch.arange(1, n_vis + 1, dtype=torch.float64),
            base_num=(256 * (st_id_pairs[:, 0] + 1) + st_id_pairs[:, 1] + 1).to(
                torch.float64
            ),
            u=torch.from_numpy(uvw_grp["u"][()]),
            v=torch.from_numpy(uvw_grp["v"][()]),
            w=torch.from_numpy(uvw_grp["w"][()]),
            date=date,
            st_id_pairs=st_id_pairs,
            linear_dop=torch.tensor([]),
            circular_dop=torch.tensor([]),
        )

        freq_bands = torch.from_numpy(f["frequency_bands"][()])
        ref_frequency = freq_bands[0]

        obs_grp = f["obs"]
        if layout is None and "layout" in obs_grp:
            stored = obs_grp["layout"][()]
            layout = stored.decode() if isinstance(stored, bytes) else str(stored)

        lm = torch.from_numpy(np.stack([f["lmn"]["l"][()], f["lmn"]["m"][()]], axis=-1))

        obs = UVH5Observation(
            ra=torch.from_numpy(obs_grp["ra"][()]),
            dec=torch.from_numpy(obs_grp["dec"][()]),
            ref_frequency=ref_frequency,
            frequency_offsets=freq_bands - ref_frequency,
            bandwidths=torch.from_numpy(f["channel_widths"][()]),
            layout=layout,
            lm=lm,
            baselines=UVH5Baselines(date=date, st_id_pairs=st_id_pairs),
            normalize=bool(f["normalize"][()]),
        )

        sky = torch.from_numpy(f["sky"]["SI"][()]) if "sky" in f else None

    return UVH5Data(vis=vis, obs=obs, sky=sky, path=path)


class UVH5Reader:
    """(Iterable) reader for a single UVH5 file or a directory of them.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        A ``.uvh5`` file, or a directory searched for ``*.uvh5`` files.
    layout : str or :class:`~pathlib.Path` or None, optional
        Array layout that overrides the one stored in the files.
        Default: ``None``.

    Examples
    --------
    Using the Gridder class from pyvisgrid:

    >>> reader = UVH5Reader("build/uvh5")
    >>> for data in reader:
    ...     grid_data = Gridder.from_pyvisgen(
    ...         vis_data=data.vis,
    ...         obs=data.obs,
    ...         img_size=data.img_size,
    ...         fov=fov,
    ...     ).grid()

    Or, for a single file:
    >>> reader = UVH5Reader("build/train_0.uvh5")
    >>> data = reader[0]
    >>> grid_data = Gridder.from_pyvisgen(
    ...     vis_data=data.vis,
    ...     obs=data.obs,
    ...     img_size=data.img_size,
    ...     fov=fov,
    ... ).grid()
    """

    def __init__(self, path: str | Path, layout: str | Path | None = None) -> None:
        path = Path(path)

        if path.is_dir():
            self.files = natsorted(path.glob("*.uvh5"), key=lambda p: p.name)
        elif path.is_file():
            self.files = [path]
        else:
            raise FileNotFoundError(f"{path} does not exist.")

        if not self.files:
            raise FileNotFoundError(f"No .uvh5 files found in {path}.")

        self.layout = layout

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> UVH5Data:
        return read_uvh5(self.files[i], layout=self.layout)

    def __iter__(self) -> Iterator[UVH5Data]:
        for file in self.files:
            yield read_uvh5(file, layout=self.layout)
