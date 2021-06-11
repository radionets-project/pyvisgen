import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as un
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def source_reference(src_crd):
    s_ref = EarthLocation.from_geodetic(
        lon=src_crd.ra,
        lat=src_crd.dec,
        height=0,
    )
    return s_ref


def get_enu(ants, src_crd):
    s_ref = source_reference(src_crd)
    source_ref = np.array([s_ref.x.value, s_ref.y.value, s_ref.z.value])
    # define offset
    B = (ants - source_ref[:, None]).T
    # define rotation
    R = rot(s_ref.lon.deg, s_ref.lat.deg)
    R = np.repeat([R], ants.shape[-1], axis=0)
    # calc new coords
    enu = np.einsum("bjk, bk->bj", R, B)
    return enu


def rot(lon, lat):
    """
    Calculates roytation matrix
    """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0],
            [
                -np.sin(lat) * np.cos(lon),
                -np.sin(lat) * np.sin(lon),
                np.cos(lat),
            ],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def update_src_coord(src_crd, hr_offset):
    s_ref = source_reference(src_crd)
    src_crd = SkyCoord(
        ra=(s_ref.to_geodetic().lon.hour + hr_offset) * 15,
        dec=src_crd.dec.deg,
        unit=(un.deg, un.deg),
    )
    return src_crd


def plot_vlba(array_layout):
    extent = [-155, -65, 10, 45.5]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    stamen_terrain = cimgt.Stamen("terrain-background")

    plt.figure(figsize=(5.78 * 2, 3.57), dpi=600)
    ax = plt.axes(projection=ccrs.Orthographic(central_lon, central_lat))
    ax.set_extent(extent)

    ants = np.array([array_layout.x, array_layout.y, array_layout.z])
    enu = get_enu(
        ants,
        SkyCoord(
            ra=central_lon,
            dec=central_lat,
            unit=(un.deg, un.deg),
        ),
    )

    ant = enu[:, :2]

    ax.plot(
        ant[:, 0],
        ant[:, 1],
        marker=".",
        color="black",
        linestyle="none",
        markersize=6,
        zorder=10,
        label="Antenna positions",
    )

    [
        ax.plot(
            [ant[i, 0], ant[j, 0]],
            [ant[i, 1], ant[j, 1]],
            zorder=1,
            marker="None",
            linestyle="-",
            color="#d62728",
            linewidth=0.8,
        )
        for i in range(10)
        for j in range(10)
    ]

    ax.add_image(stamen_terrain, 4)

    leg = plt.legend(markerscale=1.5, fontsize=7, loc=2)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)


def plot_eart(array_layout, src_crd, hr_offset=None):
    if hr_offset:
        src_crd = update_src_coord(src_crd, hr_offset)
    ants = np.array([array_layout.x, array_layout.y, array_layout.z])
    enu = get_enu(ants, src_crd)
    s_ref = source_reference(src_crd)
    central_lon = s_ref.to_geodetic().lon.deg
    central_lat = s_ref.to_geodetic().lat.deg

    proj = ccrs.NearsidePerspective(
        central_longitude=central_lon,
        central_latitude=central_lat,
        satellite_height=1e10,
    )

    fig = plt.figure(figsize=(5.78 * 2, 3.57 * 2), dpi=300)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.7)

    plt.plot(
        enu[:, 0],
        enu[:, 1],
        marker="o",
        linestyle="none",
        color="black",
        markersize=5,
        zorder=7,
        label="Projected antenna positions",
    )

    plt.plot(
        central_lon,
        central_lat,
        marker="*",
        linestyle="none",
        color="#ff7f0e",
        markersize=12,
        zorder=7,
        label="Projected source",
    )

    ant = enu[:, :2]
    [
        ax.plot(
            [ant[i, 0], ant[j, 0]],
            [ant[i, 1], ant[j, 1]],
            zorder=1,
            marker="None",
            linestyle="-",
            color="#d62728",
            linewidth=0.8,
        )
        for i in range(10)
        for j in range(10)
    ]

    ax.stock_img()
    fig.tight_layout()
