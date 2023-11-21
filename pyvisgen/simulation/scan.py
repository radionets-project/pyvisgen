from math import pi

import torch
from torch import nn


class FourierKernel(nn.Module):
    def __init__(self, bas, obs, spw, device):
        super().__init__()
        self.K = self.getK(bas, obs, spw, device)

    def getK(self, bas, obs, spw, device):
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
        device = torch.device(device)
        u_cmplt = torch.cat((bas.u_start, bas.u_stop)).to(device) / 3e8 / spw
        v_cmplt = torch.cat((bas.v_start, bas.v_stop)).to(device) / 3e8 / spw
        w_cmplt = torch.cat((bas.w_start, bas.w_stop)).to(device) / 3e8 / spw

        l = obs.lm[:, :, 0].to(device)
        m = obs.lm[:, :, 1].to(device)
        n = torch.sqrt(1 - l**2 - m**2)

        ul = torch.einsum("b,ij->ijb", u_cmplt, l)
        vm = torch.einsum("b,ij->ijb", v_cmplt, m)
        wn = torch.einsum("b,ij->ijb", w_cmplt, (n - 1))
        del l, m, n, u_cmplt, v_cmplt, w_cmplt

        K = torch.exp(-2 * pi * 1j * (ul + vm + wn))
        del ul, vm, wn
        return K

    def forward(self, img):
        return torch.einsum("lmi,lmb->lmbi", img, self.K)


class Integrate(nn.Module):
    def __init__(self):
        """Summation over l and m and avering over time and freq

        Parameters
        ----------
        X1 : 4d tensor
            visibility for every l,m and baseline for freq1
        X2 : 4d tensor
            visibility for every l,m and baseline for freq2

        Returns
        -------
        1d tensor
        Returns visibility for every baseline
        """
        super().__init__()

    def forward(self, X1, X2):
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
        return int_t.cpu()


class RIME_uncorrupted(nn.Module):
    def __init__(self, bas, obs, spw, device, grad):
        """Calculates uncorrupted visibility

        Parameters
        ----------
        bas : dataclass
            baselines dataclass
        obs : class
            observation class
        spw : float
            spectral window

        Returns
        -------
        4d array
            Returns visibility for every lm and baseline
        """
        super().__init__()
        self.bas = bas
        self.obs = obs
        self.fourier = FourierKernel(bas, obs, spw, device)
        self.integrate = Integrate()

    def forward(self, img):
        K = self.fourier(img)
        vis = self.integrate(K, K)
        return vis


'''
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
