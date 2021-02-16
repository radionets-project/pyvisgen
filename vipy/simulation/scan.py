from dataclasses import dataclass
from astropy import units as un
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import numpy as np
from scipy.special import j1
import scipy.constants as const
import scipy.signal as sig
from vipy.simulation.utils import single_occurance, get_pairs


@dataclass
class Baselines:
    name: [str]
    u: [float]
    v: [float]
    w: [float]
    valid: [bool]

    def __getitem__(self, i):
        baseline = Baseline(
            self.name[i],
            self.u[i],
            self.v[i],
            self.w[i],
            self.valid[i],
        )
        return baseline

    def add(self, baselines):
        self.name = np.concatenate([self.name, baselines.name])
        self.u = np.concatenate([self.u, baselines.u])
        self.v = np.concatenate([self.v, baselines.v])
        self.w = np.concatenate([self.w, baselines.w])
        self.valid = np.concatenate([self.valid, baselines.valid])


@dataclass
class Baseline:
    name: str
    u: float
    v: float
    w: float
    valid: bool


def get_baselines(src_crd, time, array_layout):
    """Calculates baselines from source coordinates and time of observation for
    every antenna station in array_layout.
    (Calculation for 1 timestep?)

    Parameters
    ----------
    src_crd : astropy SkyCoord object (?)
        ra and dec of source location / pointing center
    time : astropy time object
        time of observation
    array_layout : dataclass object
        station information

    Returns
    -------
    dataclass object
        baselines between telescopes with visinility flags
    """
    # Calculate for all times
    # calculate GHA, Greenwich as reference for EHT
    ha_all = Angle(
        [t.sidereal_time("apparent", "greenwich") - src_crd.ra for t in time]
    )

    # calculate elevations
    el_st_all = src_crd.transform_to(
        AltAz(
            obstime=time.reshape(len(time), -1),
            location=EarthLocation.from_geocentric(
                np.repeat([array_layout.x], len(time), axis=0),
                np.repeat([array_layout.y], len(time), axis=0),
                np.repeat([array_layout.z], len(time), axis=0),
                unit=un.m,
            ),
        )
    ).alt.degree

    # fails for 1 timestep
    assert len(ha_all.value) == len(el_st_all)

    # always the same
    delta_x, delta_y, delta_z = get_pairs(array_layout)
    indices = single_occurance(delta_x)
    delta_x = delta_x[indices]
    delta_y = delta_y[indices]
    delta_z = delta_z[indices]
    mask = [i * len(array_layout.x) + i for i in range(len(array_layout.x))]
    pairs = np.delete(
        np.array(np.meshgrid(array_layout.name, array_layout.name)).T.reshape(-1, 2),
        mask,
        axis=0,
    )[indices]

    els_low = np.delete(
        np.array(np.meshgrid(array_layout.el_low, array_layout.el_low)).T.reshape(
            -1, 2
        ),
        mask,
        axis=0,
    )[indices]

    els_high = np.delete(
        np.array(np.meshgrid(array_layout.el_high, array_layout.el_high)).T.reshape(
            -1, 2
        ),
        mask,
        axis=0,
    )[indices]

    # Loop over ha and el_st
    baselines = Baselines([], [], [], [], [])
    for ha, el_st in zip(ha_all, el_st_all):
        u = np.sin(ha) * delta_x + np.cos(ha) * delta_y
        v = (
            -np.sin(src_crd.ra) * np.cos(ha) * delta_x
            + np.sin(src_crd.ra) * np.sin(ha) * delta_y
            + np.cos(src_crd.ra) * delta_z
        )
        w = (
            np.cos(src_crd.ra) * np.cos(ha) * delta_x
            - np.cos(src_crd.ra) * np.sin(ha) * delta_y
            + np.sin(src_crd.ra) * delta_z
        )
        assert u.shape == v.shape == w.shape

        els_st = np.delete(
            np.array(np.meshgrid(el_st, el_st)).T.reshape(-1, 2),
            mask,
            axis=0,
        )[indices]

        valid = np.ones(u.shape).astype(bool)

        m1 = (els_st < els_low).any(axis=1)
        m2 = (els_st > els_high).any(axis=1)
        valid_mask = np.ma.mask_or(m1, m2)
        valid[valid_mask] = False

        names = pairs[:, 0] + "-" + pairs[:, 1]

        names = names
        u = u.reshape(-1)
        v = v.reshape(-1)
        w = w.reshape(-1)
        valid = valid.reshape(-1)

        # collect baselines
        base = Baselines(names, u, v, w, valid)
        baselines.add(base)
    return baselines


def create_bgrid(fov, samples, src_crd):
    """Calculates beam grids of telescopes depending on field of view,
    number of grid samples, and source coordinates.

    Parameters
    ----------
    fov : float
        filed of view of observation (telescope?)
    samples : int
        number of grid samples
    src_crd : astropy SkyCoord object
        ra and dec of source location / pointing center

    Returns
    -------
    2d array
        2d beam grids correponding to field of view
    """
    spacing = fov / samples

    bgrid = np.zeros((samples, samples, 2))
    mgrid = np.meshgrid(np.arange(32), np.arange(32))

    sd = np.sin(src_crd.dec.rad)
    cd = np.cos(src_crd.dec.rad)

    delx = (mgrid[1] - samples / 2.0) * spacing
    dely = (mgrid[0] - samples / 2.0) * spacing

    alpha = np.arctan(delx / (cd - dely * sd)) + src_crd.ra.rad
    delta = np.arcsin((sd + dely * cd) / np.sqrt(1 + delx ** 2 + dely ** 2))

    bgrid[..., 0] = alpha
    bgrid[..., 1] = delta
    return bgrid


def jinc(x):
    """Create jinc function.

    Parameters
    ----------
    x : array
        value of (?)

    Returns
    -------
    array
        value of jinc function at x
    """
    if x == 0:
        return 1
    return 2 * j1(x) / x


def get_beam(src_crd, array, bgrid, freq):
    """Calculates telescope beam from source coordinate, telescope array,
    and observation frequency using bgrid.

    Parameters
    ----------
    src_crd : astropy SkyCoord object
        ra and dec of source location / pointing center
    array : dataclass object (?)
        station information
    bgrid : 2d array
        2d grid correponding to field of view
    freq : float
        observation frequency

    Returns
    -------
    array
        telescope beam
    """
    beam = np.zeros((len(array), bgrid.shape[0], bgrid.shape[0]))
    for i in range(bgrid.shape[0]):
        for j in range(bgrid.shape[0]):
            crd = SkyCoord(ra=bgrid[i, j, 0], dec=bgrid[i, j, 1], unit=(un.rad, un.rad))
            dist = src_crd.separation(crd)
            wave = const.c / freq
            for n, st in enumerate(array):
                beam[n, i, j] = jinc(np.pi * st.diam / wave * np.sin(dist))
    return beam


def get_beam_conv(beam, num_telescopes):
    """Calculate convolution beam in Fourier space.

    Parameters
    ----------
    beam : array
        beams of different stations
    num_telescopes : int
        number of stations

    Returns
    -------
    array
        convolution beams in Fourier space
    """
    M = np.array([])
    for i, j1 in enumerate(beam):
        for j, j2 in enumerate(beam[i + 1 :]):
            M = np.append(M, j1 * j2)

    num_basel = int((num_telescopes ** 2 - num_telescopes) / 2)
    M = np.reshape(M, (num_basel, beam.shape[1], beam.shape[1]))
    M_ft = np.fft.fft2(M)
    return M_ft


def get_uvPatch(img_ft, bgrid, freq, bw, start_uv, stop_uv, cellsize):
    start_freq = freq - bw / 2
    stop_freq = freq + bw / 2

    u_11 = [st.u * start_freq / const.c for st in start_uv]  # start freq, start uv
    v_11 = [st.v * start_freq / const.c for st in start_uv]  # start freq, start uv
    u_12 = [st.u * stop_freq / const.c for st in start_uv]  # stop freq, start uv
    v_12 = [st.v * stop_freq / const.c for st in start_uv]  # stop freq, start uv
    u_21 = [st.u * start_freq / const.c for st in stop_uv]  # start freq, stop uv
    v_21 = [st.v * start_freq / const.c for st in stop_uv]  # start freq, stop uv
    u_22 = [st.u * stop_freq / const.c for st in stop_uv]  # stop freq, stop uv
    v_22 = [st.v * stop_freq / const.c for st in stop_uv]  # stop freq, stop uv

    # get corners of rectangular patch
    u_max = max(max(u_11, u_12), max(u_21, u_22))
    v_max = max(max(v_11, v_12), max(v_21, v_22))
    u_min = min(min(u_11, u_12), min(u_21, u_22))
    v_min = min(min(v_11, v_12), min(v_21, v_22))

    # get patch
    udim = [
        abs(u_max[i] - u_min[i]) + 4.0 * cellsize + bgrid.shape[1] * cellsize
        for i in range(len(u_max))
    ]
    vdim = [
        abs(v_max[i] - v_min[i]) + 4.0 * cellsize + bgrid.shape[1] * cellsize
        for i in range(len(u_max))
    ]

    npu = np.ceil(udim / cellsize)  # defined but unused
    npv = np.ceil(vdim / cellsize)  # defined but unused

    u0 = [b.u for b in start_uv]
    v0 = [b.v for b in start_uv]

    u1 = np.array(u0) + np.array(udim)
    v1 = np.array(v0) + np.array(vdim)

    spu = (np.ceil(u0 / cellsize) + int(img_ft.shape[0] / 2)).astype(int)
    spv = (np.ceil(v0 / cellsize) + int(img_ft.shape[0] / 2)).astype(int)

    epu = (np.ceil(u1 / cellsize) + int(img_ft.shape[0] / 2)).astype(int)
    epv = (np.ceil(v1 / cellsize) + int(img_ft.shape[0] / 2)).astype(int)

    patch = [img_ft[spu[i] : epu[i], spv[i] : epv[i]] for i in range(spu.shape[0])]
    return patch


def conv(patch, M_ft):
    """Calculates convolution between sky patch and telescope beam.

    Parameters
    ----------
    patch : array
        sky patches
    M_ft : array
        telescope's convolution beams in Fourier space

    Returns
    -------
    array
        convolved sky patch
    """
    conv = [sig.convolve2d(patch[i], M_ft[i], mode="valid") for i in range(len(patch))]
    return conv


def integrate(conv):
    """Integrates convolved sky patch to calculate visibility.

    Parameters
    ----------
    conv : array
        sky patch convolved with telescope beam

    Returns
    -------
    complex
        visibility
    """
    vis = [np.sum(conv[i]) for i in range(len(conv))]
    return vis
