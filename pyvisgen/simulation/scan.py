from dataclasses import dataclass
from astropy import units as un
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import numpy as np
from scipy.special import j1
import scipy.constants as const
import scipy.signal as sig
from astroplan import Observer
from pyvisgen.simulation.utils import single_occurance, get_pairs
from pyvisgen.layouts import layouts
import torch
import itertools
import time as t
import numexpr as ne  # fast exponential
from einsumt import einsumt as einsum


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


def rd_grid(fov, samples, src_crd):
    """Calculates RA and Dec values for a given fov around a source position

    Parameters
    ----------
    fov : float
        FOV size
    samples : int
        number of pixels
    src_crd : astropy SkyCoord
        position of source

    Returns
    -------
    3d array
        Returns a 3d array with every pixel containing a RA and Dec value
    """
    res = fov / samples

    rd_grid = np.zeros((samples, samples, 2))
    for i in range(samples):
        rd_grid[i, :, 0] = np.array(
            [(i - samples / 2) * res + src_crd.ra.rad for i in range(samples)]
        )
        rd_grid[:, i, 1] = np.array(
            [-(i - samples / 2) * res + src_crd.dec.rad for i in range(samples)]
        )

    return rd_grid


def lm_grid(rd_grid, src_crd):
    """Calculates sine projection for fov

    Parameters
    ----------
    rd_grid : 3d array
        array containing a RA and Dec value in every pixel
    src_crd : astropy SkyCoord
        source position

    Returns
    -------
    3d array
        Returns a 3d array with every pixel containing a l and m value
    """
    lm_grid = np.zeros(rd_grid.shape)
    lm_grid[:, :, 0] = np.cos(rd_grid[:, :, 1]) * np.sin(
        rd_grid[:, :, 0] - src_crd.ra.rad
    )
    lm_grid[:, :, 1] = np.sin(rd_grid[:, :, 1]) * np.cos(src_crd.dec.rad) - np.cos(
        src_crd.dec.rad
    ) * np.sin(src_crd.dec.rad) * np.cos(rd_grid[:, :, 0] - src_crd.ra.rad)

    return lm_grid


def uncorrupted(lm, baselines, wave, time, src_crd, array_layout, SI):
    """Calculates uncorrupted visibility

    Parameters
    ----------
    lm : 3d array
        every pixel containing a l and m value
    baselines : dataclass
        baseline information
    wave : float
        wavelength of observation
    time : astropy Time
        Time steps of observation
    src_crd : astropy SkyCoord
        source position
    array_layout : dataclass
        station information
    I : 2d array
        source brightness distribution / input img

    Returns
    -------
    4d array
        Returns visibility for every lm and baseline
    """
    stat_num = array_layout.st_num.shape[0]
    base_num = int(stat_num * (stat_num - 1) / 2)

    vectorized_num = np.vectorize(lambda st: st.st_num, otypes=[int])
    st1, st2 = get_valid_baselines(baselines, base_num)

    st1_num = vectorized_num(st1)

    if st1_num.shape[0] == 0:
        return torch.zeros(1)

    K = getK(baselines, lm, wave, base_num)

    B = np.zeros((lm.shape[0], lm.shape[1], 1), dtype=complex)

    B[:, :, 0] = SI + SI
    # # only calculate without polarization for the moment
    # B[:, :, 0, 0] = SI[:, :, 0] + SI[:, :, 1]
    # B[:, :, 0, 1] = SI[:, :, 2] + 1j * SI[:, :, 3]
    # B[:, :, 1, 0] = SI[:, :, 2] - 1j * SI[:, :, 3]
    # B[:, :, 1, 1] = SI[:, :, 0] - SI[:, :, 1]

    X = torch.einsum("lmi,lmb->lmbi", torch.tensor(B), K)

    return X


def corrupted(lm, baselines, wave, time, src_crd, array_layout, I, rd):
    """Calculates corrupted visibility

    Parameters
    ----------
    lm : 3d array
        every pixel containing a l and m value
    baselines : dataclass
        baseline information
    wave : float
        wavelength of observation
    time : astropy Time
        Time steps of observation
    src_crd : astropy SkyCoord
        source position
    array_layout : dataclass
        station information
    I : 2d array
        source brightness distribution / input img
    rd : 3d array
        RA and dec values for every pixel

    Returns
    -------
    4d array
        Returns visibility for every lm and baseline
    """

    stat_num = array_layout.st_num.shape[0]
    base_num = int(stat_num * (stat_num - 1) / 2)

    vectorized_num = np.vectorize(lambda st: st.st_num, otypes=[int])
    st1, st2 = get_valid_baselines(baselines, base_num)
    st1_num = vectorized_num(st1)
    st2_num = vectorized_num(st2)
    if st1_num.shape[0] == 0:
        return torch.zeros(1)

    K = getK(baselines, lm, wave, base_num)

    B = np.zeros((lm.shape[0], lm.shape[1], 2, 2), dtype=complex)

    B[:, :, 0, 0] = I[:, :, 0] + I[:, :, 1]
    B[:, :, 0, 1] = I[:, :, 2] + 1j * I[:, :, 3]
    B[:, :, 1, 0] = I[:, :, 2] - 1j * I[:, :, 3]
    B[:, :, 1, 1] = I[:, :, 0] - I[:, :, 1]

    # coherency
    X = torch.einsum("lmij,lmb->lmbij", torch.tensor(B), K)
    # X = np.einsum('lmij,lmb->lmbij', B, K, optimize=True)
    # X = torch.tensor(B)[:,:,None,:,:] * K[:,:,:,None,None]

    del K

    # telescope response
    E_st = getE(rd, array_layout, wave, src_crd)
    # E1 = torch.tensor(E_st[:, :, st1_num, :, :], dtype=torch.cdouble)
    # E2 = torch.tensor(E_st[:, :, st2_num, :, :], dtype=torch.cdouble)
    E1 = torch.tensor(E_st[:, :, st1_num], dtype=torch.cdouble)
    E2 = torch.tensor(E_st[:, :, st2_num], dtype=torch.cdouble)

    EX = torch.einsum("lmb,lmbij->lmbij", E1, X)

    del E1, X
    # EXE = torch.einsum('lmbij,lmbjk->lmbik',EX,torch.transpose(torch.conj(E2),3,4))
    EXE = torch.einsum("lmbij,lmb->lmbij", EX, E2)
    del EX, E2

    # P matrix
    # parallactic angle

    beta = np.array(
        [
            Observer(
                EarthLocation(st.x * un.m, st.y * un.m, st.z * un.m)
            ).parallactic_angle(time, src_crd)
            for st in array_layout
        ]
    )
    tsob = time_step_of_baseline(baselines, base_num)
    b1 = np.array([beta[st1_num[i], tsob[i]] for i in range(st1_num.shape[0])])
    b2 = np.array([beta[st2_num[i], tsob[i]] for i in range(st2_num.shape[0])])
    P1 = torch.tensor(getP(b1), dtype=torch.cdouble)
    P2 = torch.tensor(getP(b2), dtype=torch.cdouble)

    PEXE = torch.einsum("bij,lmbjk->lmbik", P1, EXE)
    del EXE
    PEXEP = torch.einsum(
        "lmbij,bjk->lmbik", PEXE, torch.transpose(torch.conj(P2), 1, 2)
    )
    del PEXE

    return PEXEP


def direction_independent(lm, baselines, wave, time, src_crd, array_layout, I, rd):
    """Calculates direction independet visibility

    Parameters
    ----------
    lm : 3d array
        every pixel containing a l and m value
    baselines : dataclass
        baseline information
    wave : float
        wavelength of observation
    time : astropy Time
        Time steps of observation
    src_crd : astropy SkyCoord
        source position
    array_layout : dataclass
        station information
    I : 2d array
        source brightness distribution / input img
    rd : 3d array
        RA and dec values for every pixel

    Returns
    -------
    4d array
        Returns visibility for every lm and baseline
    """
    stat_num = array_layout.st_num.shape[0]
    base_num = int(stat_num * (stat_num - 1) / 2)

    vectorized_num = np.vectorize(lambda st: st.st_num, otypes=[int])
    st1, st2 = get_valid_baselines(baselines, base_num)
    st1_num = vectorized_num(st1)
    st2_num = vectorized_num(st2)
    if st1_num.shape[0] == 0:
        return torch.zeros(1)

    K = getK(baselines, lm, wave, base_num)

    B = np.zeros((lm.shape[0], lm.shape[1], 2, 2), dtype=complex)

    B[:, :, 0, 0] = I[:, :, 0] + I[:, :, 1]
    B[:, :, 0, 1] = I[:, :, 2] + 1j * I[:, :, 3]
    B[:, :, 1, 0] = I[:, :, 2] - 1j * I[:, :, 3]
    B[:, :, 1, 1] = I[:, :, 0] - I[:, :, 1]

    # coherency
    X = torch.einsum("lmij,lmb->lmbij", torch.tensor(B), K)

    del K

    # telescope response
    E_st = getE(rd, array_layout, wave, src_crd)

    E1 = torch.tensor(E_st[:, :, st1_num], dtype=torch.cdouble)
    E2 = torch.tensor(E_st[:, :, st2_num], dtype=torch.cdouble)

    EX = torch.einsum("lmb,lmbij->lmbij", E1, X)

    del E1, X

    EXE = torch.einsum("lmbij,lmb->lmbij", EX, E2)
    del EX, E2

    return EXE


def integrate(X1, X2):
    """Summation over l and m and avering over time and freq

    Parameters
    ----------
    X1 : 4d array
        visibility for every l,m and baseline for freq1
    X2 : 4d array
        visibility for every l,m and baseline for freq2

    Returns
    -------
    2d array
        Returns visibility for every baseline
    """
    X_f = torch.stack((X1, X2))

    int_m = torch.sum(X_f, dim=2)
    del X_f
    int_l = torch.sum(int_m, dim=1)
    del int_m
    int_f = 0.5 * torch.sum(int_l, dim=0)
    del int_l

    X_t = torch.stack(torch.split(int_f, int(int_f.shape[0] / 2), dim=0))
    del int_f
    int_t = 0.5 * torch.sum(X_t, dim=0)
    del X_t

    return int_t


def getE(rd, array_layout, wave, src_crd):
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
    # E = np.zeros((rd.shape[0], rd.shape[1], array_layout.st_num.shape[0], 2, 2))
    E = np.zeros((rd.shape[0], rd.shape[1], array_layout.st_num.shape[0]))

    # get diameters of all stations and do vectorizing stuff
    diameters = array_layout.diam

    theta = angularDistance(rd, src_crd)

    x = 2 * np.pi / wave * np.einsum("s,rd->rds", diameters, theta)

    E[:, :, :] = jinc(x)
    # E[:,:,:,0,0] = jinc(x)
    # E[..., 1, 1] = E[..., 0, 0]

    return E


def angularDistance(rd, src_crd):
    """Calculates angular distance from source position

    Parameters
    ----------
    rd : 3d array
        every pixel containing ra and dec
    src_crd : astropy SkyCoord
        source position

    Returns
    -------
    2d array
        Returns angular Distance for every pixel in rd grid with respect
        to source position
    """
    r = rd[:, :, 0] - src_crd.ra.rad
    d = rd[:, :, 1] - src_crd.dec.rad

    theta = np.arcsin(np.sqrt(r ** 2 + d ** 2))

    return theta


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
    w = baselines.w.reshape(-1, base_num) / wave

    u_start = u[:-1][mask]
    u_stop = u[1:][mask]
    v_start = v[:-1][mask]
    v_stop = v[1:][mask]
    w_start = w[:-1][mask]
    w_stop = w[1:][mask]

    u_cmplt = np.append(u_start, u_stop)
    v_cmplt = np.append(v_start, v_stop)
    w_cmplt = np.append(w_start, w_stop)

    l = torch.tensor(lm[:, :, 0])
    m = torch.tensor(lm[:, :, 1])
    n = torch.sqrt(1 - l ** 2 - m ** 2)

    ul = torch.einsum("b,ij->ijb", torch.tensor(u_cmplt), l)
    vm = torch.einsum("b,ij->ijb", torch.tensor(v_cmplt), m)
    wn = torch.einsum("b,ij->ijb", torch.tensor(w_cmplt), (n - 1))

    pi = np.pi
    test = ul + vm + wn
    K = ne.evaluate("exp(-2 * pi * 1j * (ul + vm + wn))")  # -0.4 secs for vlba
    return torch.tensor(K)


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
