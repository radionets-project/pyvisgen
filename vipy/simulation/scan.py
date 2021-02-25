from dataclasses import dataclass
from astropy import units as un
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import numpy as np
from scipy.special import j1
import scipy.constants as const
import scipy.signal as sig
from astroplan import Observer
from vipy.simulation.utils import single_occurance, get_pairs
from vipy.layouts import layouts
import torch
import itertools


@dataclass
class Baselines:
    name: [str]
    st1: [object]
    st2: [object]
    u: [float]
    v: [float]
    w: [float]
    valid: [bool]

    def __getitem__(self, i):
        baseline = Baseline(
            self.name[i],
            self.st1[i],
            self.st2[i],
            self.u[i],
            self.v[i],
            self.w[i],
            self.valid[i],
        )
        return baseline

    def add(self, baselines):
        self.name = np.concatenate([self.name, baselines.name])
        self.st1 = np.concatenate([self.st1, baselines.st1])
        self.st2 = np.concatenate([self.st2, baselines.st2])
        self.u = np.concatenate([self.u, baselines.u])
        self.v = np.concatenate([self.v, baselines.v])
        self.w = np.concatenate([self.w, baselines.w])
        self.valid = np.concatenate([self.valid, baselines.valid])


@dataclass
class Baseline:
    name: str
    st1: object
    st2: object
    u: float
    v: float
    w: float
    valid: bool

    def baselineNum(self):
        return 256 * (self.st1.st_num + 1) + self.st2.st_num + 1


def get_baselines(src_crd, time, array_layout):
    """Calculates baselines from source coordinates and time of observation for
    every antenna station in array_layout.

    Parameters
    ----------
    src_crd : astropy SkyCoord object
        ra and dec of source location / pointing center
    time : w time object
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

    st_nums = np.delete(
        np.array(np.meshgrid(array_layout.st_num, array_layout.st_num)).T.reshape(
            -1, 2
        ),
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
    baselines = Baselines([], [], [], [], [], [], [])
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

        u = u.reshape(-1)
        v = v.reshape(-1)
        w = w.reshape(-1)
        valid = valid.reshape(-1)

        # collect baselines
        base = Baselines(
            names,
            array_layout[st_nums[:, 0]],
            array_layout[st_nums[:, 1]],
            u,
            v,
            w,
            valid,
        )
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
    mgrid = np.meshgrid(np.arange(samples), np.arange(samples))

    sd = np.sin(src_crd.dec.rad)
    cd = np.cos(src_crd.dec.rad)

    delx = (mgrid[1] - samples / 2.0) * spacing
    dely = (mgrid[0] - samples / 2.0) * spacing

    alpha = np.arctan(delx / (cd - dely * sd)) + src_crd.ra.rad
    delta = np.arcsin((sd + dely * cd) / np.sqrt(1 + delx ** 2 + dely ** 2))

    bgrid[..., 0] = alpha
    bgrid[..., 1] = delta
    return bgrid


def lm(grid, src_crd):
    """Calculates lm grid of the source. Depends on beam (rename to FOV grid?)
    and source position

    Parameters
    ----------
    grid : 2d array
        beam/fov grid of source
    src_crd : astropy SkyCoord
        Position of source

    Returns
    -------
    2d array
        fov grid in lm/cosine coordinates for FOV
    """
    lm = np.zeros(grid.shape)
    ra = grid[..., 0]
    dec = grid[..., 1]
    lm[..., 0] = np.cos(src_crd.dec.rad) * np.sin(src_crd.ra.rad - ra)
    lm[..., 1] = np.sin(dec) * np.cos(src_crd.dec.rad) - np.cos(dec) * np.sin(
        src_crd.dec.rad
    ) * np.cos(src_crd.ra.rad - ra)
    return lm


def getJones(lm, baselines, wave, time, src_crd, array_layout):
    """Calculate all Jones matrices for stations in every baselines given and returns
    Kronecker product (Mueller matrix).

    Parameters
    ----------
    lm : 2d array
        lm grid for FOV
    baselines : dataclass object
        all calculated baselines for measurement
    wave : float
        wavelenght for observation
    time : astropy time object
        times for every set of baselines measured
    src_crd : astropy SkyCoord
        position of source
    array_layout : dataclass object
        station information

    Returns
    -------
    5d array
        Shape of first two axes is given by fov grid. Shape of third axis is given by number of baselines.
        Last two axes have shape of 4 (Mueller matrix).
        This output return a Mueller matrix for every baseline and pixel in lm grid.
    """
    # J = E P K
    # for every entry in baselines exists one station pair (st1,st2)
    # st1 has a jones matrix, st2 has a jones matrix
    # Calculate Jones matrices for every baseline
    JJ = np.zeros(
        (lm.shape[0], lm.shape[1], baselines.name.shape[0], 4, 4), dtype=complex
    )

    # parallactic angle
    beta = np.array(
        [
            Observer(
                EarthLocation(st.x * un.m, st.y * un.m, st.z * un.m)
            ).parallactic_angle(time, src_crd)
            for st in array_layout
        ]
    )

    # calculate E Matrices
    E_st = getE(lm, array_layout, wave)

    vectorized_num = np.vectorize(lambda st: st.st_num, otypes=[int])
    valid = baselines.valid.astype(bool)

    stat_num = array_layout.st_num.shape[0]
    base_num = int(stat_num * (stat_num - 1) / 2)

    st1, st2 = get_valid_baselines(baselines, base_num)

    st1_num = vectorized_num(st1)
    st2_num = vectorized_num(st2)

    if st1_num.shape[0] == 0:
        return torch.zeros(1)

    E1 = torch.tensor(E_st[:, :, st1_num, :, :])
    E2 = torch.tensor(E_st[:, :, st2_num, :, :])

    # calculate P Matrices
    tsob = time_step_of_baseline(baselines, base_num)
    b1 = np.array([beta[st1_num[i], tsob[i]] for i in range(st1_num.shape[0])])
    b2 = np.array([beta[st2_num[i], tsob[i]] for i in range(st2_num.shape[0])])
    P1 = torch.tensor(getP(b1))
    P2 = torch.tensor(getP(b2))

    # calculate K matrices
    K = getK(baselines, lm, wave, base_num)

    J1 = torch.matmul(E1, P1)
    J2 = torch.matmul(E2, P2)

    # Kronecker product
    JJ = torch.einsum("...lm,...no->...lnmo", J1, J2).reshape(
        lm.shape[0], lm.shape[1], st1_num.shape[0], 4, 4
    )
    JJ = torch.einsum("lmbij,lmb->lmbij", JJ, K)

    return JJ


def getE(lm, array_layout, wave):
    """Calculates Jones matrix E for every pixel in lm grid and every station given.

    Parameters
    ----------
    lm : 2d array
        lm grid for FOV
    array_layout : dataclass object
        station information
    wave : float
        wavelenght

    Returns
    -------
    5d array
        Returns Jones matrix for every pixel in lm grid and every station.
        Shape is given by lm-grid axes, station axis, and (2,2) Jones matrix axes
    """
    # calculate matrix E for every point in grid
    E = np.zeros((lm.shape[0], lm.shape[1], array_layout.st_num.shape[0], 2, 2))

    # get diameters of all stations and do vectorizing stuff
    diameters = array_layout.diam
    di = np.zeros((lm.shape[0], lm.shape[1], array_layout.st_num.shape[0]))
    di[:, :] = diameters

    E[..., 0, 0] = jinc(
        np.pi
        * di
        / wave
        * np.repeat(
            np.sin(np.sqrt(lm[..., 0] ** 2 + lm[..., 1] ** 2)),
            array_layout.st_num.shape[0],
            1,
        ).reshape((lm.shape[0], lm.shape[1], array_layout.st_num.shape[0]))
    )
    E[..., 1, 1] = E[..., 0, 0]
    return E


def getP(beta):
    """Calculates Jones matrix P for given parallactic angles beta

    Parameters
    ----------
    beta : float array
        parallactic angles

    Returns
    -------
    3d array
        Return Jones matrix for every angle.
        Shape is given by beta axis and (2,2) Jones matrix axes
    """
    # calculate matrix P with parallactic angle beta
    P = np.zeros((beta.shape[0], 2, 2))

    P[:, 0, 0] = np.cos(beta)
    P[:, 0, 1] = -np.sin(beta)
    P[:, 1, 0] = np.sin(beta)
    P[:, 1, 1] = np.cos(beta)
    return P


def getK(baselines, lm, wave, base_num):
    """Calculates Fouriertransformation Kernel for every baseline and pixel in lm grid.

    Parameters
    ----------
    baselines : dataclass object
        basline information
    lm : 2d array
        lm grid for FOV
    wave : float
        wavelength

    Returns
    -------
    3d array
        Return Fourier Kernel for every pixel in lm grid and given baselines.
        Shape is given by lm axes and baseline axis
    """
    # new valid baseline calculus. for details see function get_valid_baselines()
    valid = baselines.valid.reshape(-1, base_num)
    mask = np.array(valid[:-1]).astype(bool) & np.array(valid[1:]).astype(bool)

    u = baselines.u.reshape(-1, base_num) / wave
    v = baselines.v.reshape(-1, base_num) / wave

    u_start = u[:-1][mask]
    u_stop = u[1:][mask]
    v_start = v[:-1][mask]
    v_stop = v[1:][mask]

    u_cmplt = np.append(u_start, u_stop)
    v_cmplt = np.append(v_start, v_stop)

    l = torch.tensor(lm[:, :, 0])
    m = torch.tensor(lm[:, :, 1])

    ul = torch.einsum("b,ij->ijb", torch.tensor(u_cmplt), l)
    vm = torch.einsum("b,ij->ijb", torch.tensor(v_cmplt), m)

    K = torch.exp(2 * np.pi * 1j * (ul + vm))

    return K


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
    # if x == 0:
    #     return 1
    # jinc = 2 * j1(x) / x
    # jinc[x == 0] = 1
    jinc = np.ones(x.shape)
    jinc[x != 0] = 2 * j1(x[x != 0]) / x[x != 0]
    return jinc


def integrate(JJ_f1, JJ_f2, I, base_num, delta_t, delta_f, delta_l, delta_m):
    """Integration over frequency, time, l, m

    Parameters
    ----------
    JJ_f1 : 5d array
        Mueller matrices for frequency-bw
    JJ_f2 : 5d array
        Mueller matrices for frequency-bw
    I : 3d array
        Stokes vector
    base_num : int
        number of baselines per corr_int_time
    delta_t : float
        t1-t0
    delta_f : float
        bandwidth bw
    delta_l : float
        l width
    delta_m : float
        m width

    Returns
    -------
    2d array
        Returns visibility vector for each baseline calculated.
    """
    # Stokes matrix
    S = 0.5 * torch.tensor(
        [[1, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j], [1, -1, 0, 0]],
        dtype=torch.cdouble,
    )
    JJ_f1 = torch.einsum("lmbij,lmj->lmbi", JJ_f1 @ S, I)
    JJ_f2 = torch.einsum("lmbij,lmj->lmbi", JJ_f2 @ S, I)

    JJ_f = torch.stack((JJ_f1, JJ_f2))
    del JJ_f1, JJ_f2

    int_m = torch.trapz(JJ_f, axis=2)
    del JJ_f
    int_l = torch.trapz(int_m, axis=1)
    int_f = torch.trapz(int_l, axis=0)
    int_t = torch.trapz(
        torch.stack(torch.split(int_f, int(int_f.shape[0] / 2), dim=0)), axis=0
    )

    integral = int_t / (delta_t * delta_f * delta_l * delta_m)

    return integral


def get_valid_baselines(baselines, base_num):
    """Calculates all valid baselines. This depens on the baselines that are visible at start and stop times

    Parameters
    ----------
    baselines : dataclass object
        baseline spec
    base_num : number of all baselines per time step
        N*(N-1)/2

    Returns
    -------
    2 1d arrays
        Returns valid stations for every baselines as array
    """
    # reshape valid mask to (time, total baselines per time)
    valid = baselines.valid.reshape(-1, base_num)

    # generate a mask to only take baselines that are visible at start and stop time
    # example:  telescope is visible at time t_0 but not visible at time t_1, therefore throw away baseline
    # this is checked for every pair of time: t_0-t_1, t_1-t_2,...
    # t_0<-mask[0]->t_1, t_1<-mask[1]->t_2,...
    mask = np.array(valid[:-1]).astype(bool) & np.array(valid[1:]).astype(bool)

    # reshape stations to apply mask
    st1 = baselines.st1.reshape(-1, base_num)
    st2 = baselines.st2.reshape(-1, base_num)

    # apply mask
    # bas_stx[:-1][mask] gives all start stx
    # bas_stx[1:][mask] gives all stop stx
    st1_start = st1[:-1][mask]
    st1_stop = st1[1:][mask]
    st2_start = st2[:-1][mask]
    st2_stop = st2[1:][mask]

    st1_cmplt = np.append(st1_start, st1_stop)
    st2_cmplt = np.append(st2_start, st2_stop)

    return st1_cmplt, st2_cmplt


def time_step_of_baseline(baselines, base_num):
    """Calculates the time step for every valid baseline

    Parameters
    ----------
    baselines : dataclass object
        baseline specs
    base_num : number of all baselines per time step
        N*(N-1)/2

    Returns
    -------
    1d array
        Return array with every time step repeated N times, where N is the number of valid baselines per time step
    """
    # reshape valid mask to (time, total baselines per time)
    valid = baselines.valid.reshape(-1, base_num)

    # generate a mask to only take baselines that are visible at start and stop time
    # example:  telescope is visible at time t_0 but not visible at time t_1, therefore throw away baseline
    # this is checked for every pair of time: t_0-t_1, t_1-t_2,...
    # t_0<-mask[0]->t_1, t_1<-mask[1]->t_2,...
    mask = np.array(valid[:-1]).astype(bool) & np.array(valid[1:]).astype(bool)

    # DIFFERENCE TO get_valid_baselines
    # calculate sum over axis 1 to get number of valid baselines at each time step
    valid_per_step = np.sum(mask, axis=1)

    # write time for every valid basline into list and reshape
    time_step = [[t_idx] * vps for t_idx, vps in enumerate(valid_per_step)]
    time_step = np.array(list(itertools.chain(*time_step)))
    time_step = np.append(time_step, time_step + 1)  # +1???

    return time_step
