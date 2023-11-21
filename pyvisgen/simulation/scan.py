import itertools
from math import pi

import numpy as np
import torch
from astroplan import Observer
from astropy import units as un
from astropy.coordinates import EarthLocation
from scipy.special import j1


def uncorrupted(bas, obs, spw, time, SI):
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
    SI : 2d array
        source brightness distribution / input img

    Returns
    -------
    4d array
        Returns visibility for every lm and baseline
    """
    K = getK(bas, obs, spw, time)
    B = torch.zeros((obs.lm.shape[0], obs.lm.shape[1], 1))

    B[:, :, 0] = torch.tensor(SI) + torch.tensor(SI)

    X = torch.einsum("lmi,lmb->lmbi", B, K)
    del K, B
    return X


def corrupted(lm, baselines, wave, time, src_crd, array_layout, SI, rd):
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
    SI : 2d array
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

    B = np.zeros((lm.shape[0], lm.shape[1], 1), dtype=complex)

    B[:, :, 0] = SI + SI
    # # only calculate without polarization for the moment
    # B[:, :, 0, 0] = SI[:, :, 0] + SI[:, :, 1]
    # B[:, :, 0, 1] = SI[:, :, 2] + 1j * SI[:, :, 3]
    # B[:, :, 1, 0] = SI[:, :, 2] - 1j * SI[:, :, 3]
    # B[:, :, 1, 1] = SI[:, :, 0] - SI[:, :, 1]

    X = torch.einsum("lmi,lmb->lmbi", torch.tensor(B), K)
    # X = np.einsum('lmij,lmb->lmbij', B, K, optimize=True)
    # X = torch.tensor(B)[:,:,None,:,:] * K[:,:,:,None,None]

    del K, B

    # telescope response
    E_st = getE(rd, array_layout, wave, src_crd)
    # E1 = torch.tensor(E_st[:, :, st1_num, :, :], dtype=torch.cdouble)
    # E2 = torch.tensor(E_st[:, :, st2_num, :, :], dtype=torch.cdouble)
    E1 = torch.tensor(E_st[:, :, st1_num], dtype=torch.cdouble)
    E2 = torch.tensor(E_st[:, :, st2_num], dtype=torch.cdouble)

    EX = torch.einsum("lmb,lmbi->lmbi", E1, X)

    del E1, X, E_st
    # EXE = torch.einsum('lmbij,lmbjk->lmbik',EX,torch.transpose(torch.conj(E2),3,4))
    EXE = torch.einsum("lmbi,lmb->lmbi", EX, E2)
    del EX, E2

    # return EXE

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

    PEXE = torch.einsum("bi,lmbi->lmbi", P1, EXE)
    del EXE, P1, beta, tsob
    PEXEP = torch.einsum("lmbi,bi->lmbi", PEXE, torch.conj(P2))
    del PEXE, P2

    return PEXEP


def direction_independent(lm, baselines, wave, time, src_crd, array_layout, SI, rd):
    """Calculates direction independent visibility

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
    SI : 2d array
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

    B = np.zeros((lm.shape[0], lm.shape[1], 1), dtype=complex)

    B[:, :, 0] = SI + SI
    # B[:, :, 0, 0] = I[:, :, 0] + I[:, :, 1]
    # B[:, :, 0, 1] = I[:, :, 2] + 1j * I[:, :, 3]
    # B[:, :, 1, 0] = I[:, :, 2] - 1j * I[:, :, 3]
    # B[:, :, 1, 1] = I[:, :, 0] - I[:, :, 1]

    # coherency
    X = torch.einsum("lmi,lmb->lmbi", torch.tensor(B), K)

    del K

    # telescope response
    E_st = getE(rd, array_layout, wave, src_crd)

    E1 = torch.tensor(E_st[:, :, st1_num], dtype=torch.cdouble)
    E2 = torch.tensor(E_st[:, :, st2_num], dtype=torch.cdouble)

    EX = torch.einsum("lmb,lmbi->lmbi", E1, X)

    del E1, X

    EXE = torch.einsum("lmbi,lmb->lmbi", EX, E2)
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

    theta = np.arcsin(np.sqrt(r**2 + d**2))

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
    P = np.zeros((beta.shape[0], 1))

    P[:, 0] = np.cos(beta)
    # P[:, 0, 1] = -np.sin(beta)
    # P[:, 1, 0] = np.sin(beta)
    # P[:, 1, 1] = np.cos(beta)
    return P


def getK(bas, obs, spw, time):
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
    u_cmplt = torch.cat((bas.u_start, bas.u_stop)) / 3e8 / spw
    v_cmplt = torch.cat((bas.v_start, bas.v_stop)) / 3e8 / spw
    w_cmplt = torch.cat((bas.w_start, bas.w_stop)) / 3e8 / spw

    l = obs.lm[:, :, 0]
    m = obs.lm[:, :, 1]
    n = torch.sqrt(1 - l**2 - m**2)

    ul = torch.einsum("b,ij->ijb", u_cmplt, l)
    vm = torch.einsum("b,ij->ijb", v_cmplt, m)
    wn = torch.einsum("b,ij->ijb", w_cmplt, (n - 1))
    del l, m, n, u_cmplt, v_cmplt, w_cmplt

    K = torch.exp(-2 * pi * 1j * (ul + vm + wn))
    del ul, vm, wn
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
    jinc = np.ones(x.shape)
    jinc[x != 0] = 2 * j1(x[x != 0]) / x[x != 0]
    return jinc


def get_valid_baselines(baselines, base_num):
    """Calculates all valid baselines. This depens on the baselines that are visible at
    start and stop times.

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
    # example:  telescope is visible at time t_0 but not visible at time t_1, therefore
    # throw away baseline
    # this is checked for every pair of time: t_0-t_1, t_1-t_2,...
    # t_0<-mask[0]->t_1, t_1<-mask[1]->t_2,...
    mask = np.array(valid[:-1]).astype(bool) & np.array(valid[1:]).astype(bool)

    print(mask.shape)
    # reshape stations to apply mask
    print(baselines.st1.shape)
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
        Return array with every time step repeated N times, where N is the number of
        valid baselines per time step
    """
    # reshape valid mask to (time, total baselines per time)
    valid = baselines.valid.reshape(-1, base_num)

    # generate a mask to only take baselines that are visible at start and stop time
    # example:  telescope is visible at time t_0 but not visible at time t_1, therefore
    # throw away baseline
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
