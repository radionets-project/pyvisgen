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
        base = Baselines(names, array_layout[st_nums[:, 0]], array_layout[st_nums[:, 1]], u, v, w, valid)
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
    ra = grid[...,0]
    dec = grid[...,1]
    lm[...,0] = np.cos(src_crd.dec.rad) * np.sin(src_crd.ra.rad-ra)
    lm[...,1] = np.sin(dec)*np.cos(src_crd.dec.rad) - np.cos(dec)*np.sin(src_crd.dec.rad)*np.cos(src_crd.ra.rad-ra)
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
    JJ = np.zeros((lm.shape[0],lm.shape[1],baselines.name.shape[0],4,4),dtype=complex)

    # parallactic angle
    beta = np.array([Observer(EarthLocation(st.x*un.m,st.y*un.m,st.z*un.m)).parallactic_angle(time, src_crd) for st in array_layout])


    # calculate E Matrices
    E1 = getE(lm, baselines.st1, wave)
    E2 = getE(lm, baselines.st2, wave)

    

    # calculate P Matrices
    vectorized_num = np.vectorize(lambda st: st.st_num)
    P1 = getP(vectorized_num(baselines.st1))
    P2 = getP(vectorized_num(baselines.st2))

    

    # calculate K matrices
    K = getK(baselines,lm,wave)
    K = np.repeat(K, 4, axis=2).reshape(K.shape[0], K.shape[1], K.shape[2], 4)
    K = np.repeat(K, 4, axis=3).reshape(K.shape[0], K.shape[1], K.shape[2], 4, 4)

    J1 = E1@P1
    J2 = E2@P2

    #Kronecker product
    JJ = np.einsum('...lm,...no->...lnmo',J1,J2).reshape(lm.shape[0],lm.shape[1],baselines.name.shape[0],4,4)*K

    return JJ

def getE(lm, stations, wave):
    """Calculates Jones matrix E for every pixel in lm grid and every station given.

    Parameters
    ----------
    lm : 2d array
        lm grid for FOV
    stations : dataclass object
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
    E = np.zeros((lm.shape[0],lm.shape[1], stations.shape[0],2,2))

    #get diameters of all stations and do vectorizing stuff
    vectorized_diam = np.vectorize(lambda st: st.diam)
    diameters = vectorized_diam(stations)
    di = np.zeros((lm.shape[0],lm.shape[1], stations.shape[0]))
    di[:,:] = diameters 

    E[...,0,0] = jinc(np.pi * di / wave * np.repeat(np.sin(np.sqrt(lm[...,0]**2+lm[...,1]**2)), stations.shape[0], 1).reshape((lm.shape[0],lm.shape[1], stations.shape[0])))
    E[...,1,1] = E[...,0,0]
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
    #calculate matrix P with parallactic angle beta
    P = np.zeros((beta.shape[0],2,2))

    P[:,0,0] = np.cos(beta)
    P[:,0,1] = -np.sin(beta)
    P[:,1,0] = np.sin(beta)
    P[:,1,1] = np.cos(beta)
    return P


def getK(baselines,lm,wave):
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
        Return Fourier Kernel for every pixel in lm grid and given baslines.
        Shape is given by lm axes and baseline axis
    """
    u = np.zeros((lm.shape[0],lm.shape[1], baselines.name.shape[0]))
    v = np.zeros((lm.shape[0],lm.shape[1], baselines.name.shape[0]))
    u[:,:] = baselines.u/wave
    v[:,:] = baselines.v/wave

    return np.exp(1j*2*np.pi*(u*np.repeat(lm[...,0],baselines.name.shape[0],1).reshape(lm.shape[0],lm.shape[1],baselines.name.shape[0])+v*np.repeat(lm[...,1],baselines.name.shape[0],1).reshape(lm.shape[0],lm.shape[1],baselines.name.shape[0]))/wave)


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
    jinc = 2 * j1(x) / x
    jinc[x==0] = 1
    return jinc


def integrate(JJ_f1, JJ_f2, I, base_num, delta_t, delta_f, delta_l, delta_m):
    #Stokes matrix
    S = 0.5* np.array([[1,1,0,0],[0,0,1,1j],[0,0,1,-1j],[1,-1,0,0]])
    JJ_f1 = np.einsum('lmbij,lmj->lmbi',JJ_f1@S, I)
    JJ_f2 = np.einsum('lmbij,lmj->lmbi',JJ_f2@S, I)

    JJ_f = np.stack((JJ_f1, JJ_f2))
    del JJ_f1, JJ_f2

    int_m = np.trapz(JJ_f, axis=2)
    int_l = np.trapz(int_m, axis=1)
    int_f = np.trapz(int_l, axis=0)
    int_t = np.trapz(np.stack((int_f[:-base_num], int_f[base_num:])), axis=0)

    integral = int_t/(delta_t*delta_f*delta_l*delta_m)

    return integral
