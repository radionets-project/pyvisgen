from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import ITRS, AltAz, EarthLocation, SkyCoord

from ..exceptions import OptionalDependencyMissing

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["ArrayDisplay"]


def _mean_geodetic_reference(locations: EarthLocation):
    lon = locations.lon.to_value(u.rad)
    lat = locations.lat.to_value(u.rad)
    height = locations.height.to_value(u.m)

    mean_lon = np.arctan2(np.mean(np.sin(lon)), np.mean(np.cos(lon))) * u.rad
    mean_lat = np.mean(lat) * u.rad
    mean_height = np.mean(height) * u.m

    return EarthLocation.from_geodetic(
        lon=mean_lon,
        lat=mean_lat,
        height=mean_height,
    )


def _earthlocation_to_enu(
    locations: EarthLocation,
    reference: EarthLocation | None = None,
):
    if reference is None:
        reference = _mean_geodetic_reference(locations)

    tel_cart = locations.get_itrs().cartesian
    ref_cart = reference.get_itrs().cartesian

    local_itrs = ITRS(
        tel_cart - ref_cart,
        location=reference,
    )

    local_altaz = local_itrs.transform_to(AltAz(location=reference))

    return local_altaz.cartesian, reference


def _earthlocation_deg_offsets(
    locations: EarthLocation,
    reference: EarthLocation,
) -> tuple[u.Quantity, u.Quantity]:
    ref = SkyCoord(ra=reference.lon, dec=reference.lat, frame="icrs")
    loc = SkyCoord(ra=locations.lon, dec=locations.lat, frame="icrs")

    sep = ref.separation(loc)
    pa = ref.position_angle(loc)

    return (sep * np.sin(pa)).to(u.deg), (sep * np.cos(pa)).to(u.deg)


class ArrayDisplay:
    """Display a top-down view of any given :class:`~pyvisgen.layouts.Stations`
    object.

    Per default, `ArrayDisplay` will scale the unit automatically
    and change from Northing/Easting in km to lon/lat in deg if the
    distance between two telescopes/antennas is more than 100 km. The plot will be
    autoscaled but can be zoomed even further on the array if ``zoomed`` is
    set to ``True``.

    Parameters
    ----------
    stations : :class:`~pyvisgen.layouts.Stations`
        pyvisgen Stations object containing coordinates and
        antenna parameters.
    axes : :class:`~matplotlib.axes.Axes` or None, optional
        matplotlib axes to plot on. If ``None``, will use current
        axes using matplotlib pyplots :func:`~matplotlib.pyplot.gca`
        function. Default: ``None``
    title : str or None, optional
        Optional title for the plot. Default: ``None``
    unit : str, optional
        Unit for the axes. Can be `m`, `km`, `deg`, or `auto`.
        If set to `auto`, `ArrayDisplay` will change the unit automatically
        according to the threshold set in ``deg_km_thresh``.
        Default: ``"auto"``
    deg_km_thresh : float, optional
        Threshold that determines when the `ArrayDisplay` switches from
        km to deg. Default: ``100.0``
    reference_location : EarthLocation or None, optional
        Reference location for which the relative positions will
        be calculated. Ideally this is the array center. If set to
        ``None``, the center will be computed automatically.
        Default: ``None``
    zoomed : bool, optional
        Whether to zoom further onto the array. Per default the plot
        is autoscaled (limited by the radial grid). Depending on the array,
        zooming further in may be desirable. Default: ``False``
    padding_fraction : float, optional
        Border padding for the zoomed plot. Default: ``0.12``
    marker_type : str, optional
        Either ``"fixed"`` or ``"diameter"``. Whether to have a
        `fixed` marker size for each telescope/antenna or a `diameter`-scaled size.
        Default: ``"fixed"``
    marker_size : float, optional
        Scatter plot marker size. If `marker_type` is set to `diameter`, `marker_size`
        will be the baseline for the scaled markers. Default: ``20.0``
    diameter_scale : float, optional
        Additional scaling factor for diameter-scaled markers. Default: ``1.0``
    marker_color : str | None, optional
        Scatter plot marker color. If set to ``None``, the color will selected
        automatically from the current matplotlib rcParams. Default: ``None``
    """

    def __init__(
        self,
        stations,
        axes: Axes | None = None,
        title: str | None = None,
        unit: str = "auto",
        deg_km_thresh: float = 100.0,
        reference_location: EarthLocation | None = None,
        zoomed: bool = False,
        padding_fraction: float = 0.12,
        marker_type: str = "fixed",
        marker_size: float = 20.0,
        diameter_scale: float = 1.0,
        marker_color: str | None = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as e:
            raise OptionalDependencyMissing("plot") from e

        self.stations = stations
        self.axes = axes or plt.gca()
        self.unit = unit
        self.deg_km_thresh = deg_km_thresh

        self.marker_type = marker_type
        self.marker_size = marker_size
        self.diameter_scale = diameter_scale

        self.marker_color = marker_color

        self.stations_el = EarthLocation.from_geocentric(
            stations.x,
            stations.y,
            stations.z,
            unit="m",
        )

        self.st_coords, self.reference = _earthlocation_to_enu(
            self.stations_el, reference=reference_location
        )

        self.max_radius_km = np.nanmax(
            np.hypot(self.st_coords.x.to_value("km"), self.st_coords.y.to_value("km"))
        )

        self._set_unit()
        self._add_radial_grid()

        self._add_antennas()
        self._labels = []
        self.axes.set_aspect(1.0)
        self.axes.set(
            xlabel=f"{self.xlabel} / {self.axis_unit_label}",
            ylabel=f"{self.ylabel} / {self.axis_unit_label}",
            title=title,
        )

        if zoomed:
            self._pad_limits(padding_fraction)
        else:
            self.axes.autoscale_view()

    def _set_unit(self) -> None:
        if self.unit == "auto":
            self.display_unit = (
                "deg"
                if self.max_radius_km > self.deg_km_thresh
                else "km"
                if self.max_radius_km > 1
                else "m"
            )
        else:
            self.display_unit = self.unit

        if self.display_unit == "deg":
            self.x, self.y = _earthlocation_deg_offsets(
                self.stations_el, self.reference
            )
            self.z = self.st_coords.z
            self.axis_unit_label = self.display_unit
            self.r = self.stations.diam
            self.xlabel = "Approx. Rel. Longitude"
            self.ylabel = "Approx. Rel. Latitude"
        elif self.display_unit in {"m", "km"}:
            self.x = self.st_coords.y.to(self.display_unit)  # Easting
            self.y = self.st_coords.x.to(self.display_unit)  # Northing
            self.z = self.st_coords.z.to(self.display_unit)
            self.r = self.stations.diam
            self.axis_unit_label = self.display_unit
            self.xlabel = "Easting"
            self.ylabel = "Northing"
        else:
            raise ValueError(
                f"unit {self.display_unit!r} is unknown. Choose one of 'm', "
                "'km', or 'deg'"
            )

    def _dish_marker_sizes(self) -> np.ndarray:
        diam_m = u.Quantity(self.stations.diam, u.m).to_value(u.m)

        if np.any(~np.isfinite(diam_m)) or np.any(diam_m <= 0):
            raise ValueError("station diameters must be finite and > 0")

        min_diam = np.min(diam_m)

        self.diameter_factor = 1.0 + (diam_m / min_diam - 1.0) * self.diameter_scale

        if np.any(self.diameter_factor <= 0):
            raise ValueError("chosen diameter_scale creates non-positive marker sizes")

        return self.marker_size * self.diameter_factor**2

    def _add_antennas(self) -> None:
        from matplotlib.pyplot import rcParams

        x_plot = np.asarray(self.x.value)
        y_plot = np.asarray(self.y.value)

        self.axes.update_datalim(np.column_stack([x_plot, y_plot]))

        if self.marker_type == "fixed":
            self.marker_sizes = np.full_like(x_plot, self.marker_size)

        elif self.marker_type == "diameter":
            self.marker_sizes = self._dish_marker_sizes()

        else:
            raise ValueError("marker_type must be either 'fixed' or 'diameter'")

        if not self.marker_color:
            color_cycle = cycle(rcParams["axes.prop_cycle"].by_key()["color"])
            self.marker_color = next(color_cycle)

        self.antennas = self.axes.scatter(
            x_plot,
            y_plot,
            s=self.marker_sizes,
            c=self.marker_color,
            alpha=0.7,
            linewidths=2.0,
            zorder=3,
        )

    def _add_radial_grid(
        self,
        *,
        center=(0.0, 0.0),
        color="0.85",
        alpha=0.45,
        linestyle="dotted",
        linewidth=2.5,
    ) -> None:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle

        max_radius = np.nanmax(np.hypot(self.x - center[0], self.y - center[1])).value
        step = self._grid_step(max_radius / 6.0)

        if step <= 0:
            return None

        max_grid_radius = np.ceil(max_radius / step) * step
        radii = np.arange(step, max_grid_radius + 0.5 * step, step)

        circle_patches = PatchCollection(
            [
                Circle(
                    xy=center,
                    radius=r,
                    fill=False,
                )
                for r in radii
            ],
            color=color,
            ls=linestyle,
            fc="none",
            lw=linewidth,
            alpha=alpha,
        )

        self.axes.add_collection(circle_patches)

    @staticmethod
    def _grid_step(step) -> float:
        if step <= 0 or not np.isfinite(step):
            return 1.0

        exp = np.floor(np.log10(step))
        _fraction = step / 10**exp

        if _fraction <= 1:
            fraction = 1
        elif _fraction <= 2:
            fraction = 2
        elif _fraction <= 5:
            fraction = 5
        else:
            fraction = 10

        return fraction * 10**exp

    def _pad_limits(self, padding_fraction: float = 0.12) -> None:
        xmin = np.nanmin(self.x)
        xmax = np.nanmax(self.x)
        ymin = np.nanmin(self.y)
        ymax = np.nanmax(self.y)

        xspan = xmax - xmin
        yspan = ymax - ymin
        span = max(xspan, yspan).value

        if span == 0:  # pragma: no cover
            span = 1.0

        xmid = 0.5 * (xmin + xmax).value
        ymid = 0.5 * (ymin + ymax).value

        half = 0.5 * span * (1.0 + padding_fraction)

        self.axes.set_xlim(xmid - half, xmid + half)
        self.axes.set_ylim(ymid - half, ymid + half)

    def add_labels(self, ids: bool = False, pad_points: float = 2.0, **kwargs):
        """Add telescope/antenna names or ids to the scatter points.

        Parameters
        ----------
        ids : bool, optional
            If ``True``, use station ids (0, ..., N) instead of names.
            May be useful for denser arrays. Default: ``False``
        pad_points : float, optional
            Additional padding, if required, e.g. for different marker linewidths.
            Default: ``2.0``
        **kwargs
            Extra arguments to :meth:`~matplotlib.axes.Axes.annotate``. Refer to
            the matplotlib documentation for a full list of all possible arguments.

        Notes
        -----
        For very dense arrays such as the DSA-2000, the labels will overlap
        and not be legible.
        """
        px = self.x.value
        py = self.y.value

        marker_radius_points = 0.5 * np.sqrt(self.marker_sizes)

        labels = (
            self.stations.st_name
            if not ids
            else self.stations.st_num.numpy().astype(int)
        )

        annotate_kwargs = dict(
            fontsize=8,
            clip_on=True,
            ha="center",
            va="top",
            zorder=4,
        )
        annotate_kwargs.update(**kwargs)

        for label, x, y, r_pt in zip(
            labels,
            px,
            py,
            marker_radius_points,
        ):
            lab = self.axes.annotate(
                label,
                xy=(x, y),
                xycoords="data",
                xytext=(0, -(r_pt + pad_points)),
                textcoords="offset points",
                **annotate_kwargs,  # ty:ignore[invalid-argument-type]
            )
            self._labels.append(lab)

    def remove_labels(self):
        """Remove labels set by the `add_labels` method."""
        for lab in self._labels:
            lab.remove()
        self._labels = []
