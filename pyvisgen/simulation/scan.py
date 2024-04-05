from math import pi

import torch
from torch.special import bessel_j1


@torch.compile
def rime(img, bas, lm, rd, ra, dec, ant_diam, spw_low, spw_high, corrupted=False):
    """Calculates visibilities using RIME

    Parameters
    ----------
    img: torch.tensor
        sky distribution
    bas : dataclass object
        baselines dataclass
    lm : 2d array
        lm grid for FOV
    spw_low : float
        lower wavelength
    spw_high : float
        higher wavelength

    Returns
    -------
    2d tensor
        Returns visibility for every baseline
    """
    with torch.no_grad():
        X1, X2 = calc_fourier(img, bas, lm, spw_low, spw_high)
        if corrupted:
            X1, X2 = calc_beam(X1, X2, rd, ra, dec, ant_diam, spw_low, spw_high)
        vis = integrate(X1, X2)
    return vis


@torch.compile
def calc_fourier(img, bas, lm, spw_low, spw_high):
    """Calculates Fouriertransformation Kernel for every baseline and pixel in lm grid.

    Parameters
    ----------
    img: torch.tensor
        sky distribution
    bas : dataclass object
        baseline information
    lm : 2d array
        lm grid for FOV
    spw_low : float
        lower wavelength
    spw_high : float
        higher wavelength

    Returns
    -------
    3d tensor
        Return Fourier Kernel for every pixel in lm grid and given baselines.
        Shape is given by lm axes and baseline axis
    """
    u_cmplt = torch.cat((bas[0], bas[1]))
    v_cmplt = torch.cat((bas[3], bas[4]))
    w_cmplt = torch.cat((bas[6], bas[7]))

    l = lm[..., 0]
    m = lm[..., 1]
    n = torch.sqrt(1 - l**2 - m**2)

    ul = torch.einsum("u,ij->uij", u_cmplt, l)
    vm = torch.einsum("v,ij->vij", v_cmplt, m)
    wn = torch.einsum("w,ij->wij", w_cmplt, (n - 1))
    del l, m, n, u_cmplt, v_cmplt, w_cmplt

    K1 = torch.exp(
        -2 * pi * 1j * (ul / 3e8 * spw_low + vm / 3e8 * spw_low + wn / 3e8 * spw_low)
    )[..., None, None]
    K2 = torch.exp(
        -2 * pi * 1j * (ul / 3e8 * spw_high + vm / 3e8 * spw_high + wn / 3e8 * spw_high)
    )[..., None, None]
    del ul, vm, wn
    return img * K1, img * K2


# return torch.einsum("li,lb->lbi", img, K1), torch.einsum("li,lb->lbi", img, K2)


@torch.compile
def calc_beam(X1, X2, rd, ra, dec, ant_diam, spw_low, spw_high):
    diameters = ant_diam.to(rd.device)
    theta = angularDistance(rd, ra, dec)
    rds = torch.einsum("s,rd->rds", diameters, theta)

    E1 = jinc(2 * pi / 3e8 * spw_low * rds)
    E2 = jinc(2 * pi / 3e8 * spw_high * rds)

    assert E1.shape == E2.shape
    EXE1 = E1[..., None] * X1 * E1[..., None]
    # torch.einsum("lmb,nlmbi->lbi", E1, X1)
    # del X1
    # EXE1 =
    # torch.einsum("lbi,lb->lbi", EX1, E1)
    del E1, X1
    EXE2 = E2[..., None] * X2 * E2[..., None]
    # torch.einsum("lb,lbi->lbi", E2, X2)
    del E2, X2
    # EXE2 = torch.einsum("lbi,lb->lbi", EX2, E2)
    # del EX2, E2
    return EXE1, EXE2


@torch.compile
def angularDistance(rd, ra, dec):
    """Calculates angular distance from source position

    Parameters
    ----------
    rd : 3d tensor
        every pixel containing ra and dec
    ra : float
        right ascension of source position
    dec : float
        declination of source position

    Returns
    -------
    2d array
        Returns angular Distance for every pixel in rd grid with respect
        to source position
    """
    r = rd[:, :, 0] - torch.deg2rad(ra.to(rd.device))
    d = rd[:, :, 1] - torch.deg2rad(dec.to(rd.device))
    theta = torch.arcsin(torch.sqrt(r**2 + d**2))
    return theta


@torch.compile
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
    jinc = torch.ones(x.shape, device=x.device).double()
    jinc[x != 0] = 2 * bessel_j1(x[x != 0]) / x[x != 0]
    return jinc


@torch.compile
def integrate(X1, X2):
    """Summation over (l,m) and avering over time and freq

    Parameters
    ----------
    X1 : 3d tensor
        visibility for every (l,m) and baseline for freq1
    X2 : 3d tensor
        visibility for every (l,m) and baseline for freq2

    Returns
    -------
    2d tensor
    Returns visibility for every baseline
    """
    X_f = torch.stack((X1, X2))
    int_m = torch.sum(X_f, dim=2)
    del X_f
    # only integrate for 1 sky dimension
    # 2d sky is reshaped to 1d by sensitivity mask
    int_l = torch.sum(int_m, dim=2)
    del int_m
    int_f = 0.5 * torch.sum(int_l, dim=0)
    del int_l
    X_t = torch.stack(torch.split(int_f, int(int_f.shape[0] / 2), dim=0))
    del int_f
    int_t = 0.5 * torch.sum(X_t, dim=0)
    del X_t
    return int_t


'''
        W = torch.zeros(U.shape, device="cuda:0")
        src_crd = SkyCoord(ra=self.ra, dec=self.dec, unit=(un.deg, un.deg))
        ha = Angle(self.start.sidereal_time("apparent", "greenwich") - src_crd.ra)
        dec = torch.deg2rad(self.dec)  # self.rd[:, :int(N/2), 1]
        ha = torch.deg2rad(torch.tensor(ha.deg))  #self.rd[:, :int(N/2), 0]
        w = torch.cos(dec) * torch.cos(ha) * U - torch.cos(dec) *
        torch.sin(ha) * V + torch.sin(dec) * W
        w_start = w.flatten() - delta / 2
        w_stop = w.flatten() + delta / 2
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
'''
