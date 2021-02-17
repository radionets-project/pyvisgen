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
    lm = np.zeros(grid.shape)
    ra = grid[...,0]
    dec = grid[...,1]
    lm[...,0] = np.cos(src_crd.dec.rad) * np.sin(src_crd.ra.rad-ra)
    lm[...,1] = np.sin(dec)*np.cos(src_crd.dec.rad) - np.cos(dec)*np.sin(src_crd.dec.rad)*np.cos(src_crd.ra.rad-ra)
    return lm


def getJones(lm, baselines, wave, time, src_crd, array):
    # J = E P K
    # for every entry in baselines exists one station pair (st1,st2)
    # st1 has a jones matrix, st2 has a jones matrix
    # Calculate Jones matrices for every baseline
    JJ = np.zeros((lm.shape[0],lm.shape[1],baselines.name.shape[0],4,4),dtype=complex)

    # parallactic angle
    beta = np.array([Observer(EarthLocation(st.x*un.m,st.y*un.m,st.z*un.m)).parallactic_angle(time, src_crd) for st in array])


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

    JJ = np.einsum('...lm,...no->...lnmo',J1,J2).reshape(256,256,28,4,4)*K
    # for k in range(lm.shape[0]):
    #     for l in range(lm.shape[1]):
    #         JJ[k,l] = np.kron(E1[k,l]@P1,(E2[k,l]@P2).conj().T)#*K1[k,l]
    return JJ

def getE(lm, stations, wave):
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
    #calculate matrix P with parallactic angle beta
    P = np.zeros((beta.shape[0],2,2))

    P[:,0,0] = np.cos(beta)
    P[:,0,1] = -np.sin(beta)
    P[:,1,0] = np.sin(beta)
    P[:,1,1] = np.cos(beta)
    return P


def getK(baselines,lm,wave):
    u = np.zeros((lm.shape[0],lm.shape[1], baselines.name.shape[0]))
    v = np.zeros((lm.shape[0],lm.shape[1], baselines.name.shape[0]))
    u[:,:] = baselines.u
    v[:,:] = baselines.v

    return np.exp(1j*2*np.pi*(u*np.repeat(lm[...,0],baselines.name.shape[0],1).reshape(lm.shape[0],lm.shape[1],baselines.name.shape[0])+v*np.repeat(lm[...,1],baselines.name.shape[0],1).reshape(lm.shape[0],lm.shape[1],baselines.name.shape[0]))/wave)


def integrateV(JJ, I, dl, dm):
    S = 0.5* np.array([[1,1,0,0],[0,0,1,1j],[0,0,1,-1j],[1,-1,0,0]])
    vec = np.einsum('lmbij,lmj->lmbi',JJ@S, I)
    return np.sum(vec, axis=(0,1))#/(dl*dm)



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


def get_beam(src_crd, time, array, bgrid, freq):
    """Calculates telescope beam from source coordinate, telescope array,
    and observation frequency using bgrid. Uses Jones matrix calculus.
    J = D E P K

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
    beam = np.zeros((len(array), bgrid.shape[0], bgrid.shape[0], 2, 2))

    D = np.eye(2)


    for i in range(bgrid.shape[0]):
        for j in range(bgrid.shape[0]):
            crd = SkyCoord(ra=bgrid[i, j, 0], dec=bgrid[i, j, 1], unit=(un.rad, un.rad))
            dist = src_crd.separation(crd)
            wave = const.c / freq

            locs = [EarthLocation(st.x*un.m,st.y*un.m,st.z*un.m) for st in array]
            beta = [Observer(locs[i]).parallactic_angle(time, src_crd) for i in range(len(locs))]
            for n, st in enumerate(array):
                P = np.array([[np.cos(beta[n]), -np.sin(beta[n])],[np.sin(beta[n]), np.cos(beta[n])]])

                E = np.array([[jinc(np.pi * st.diam / wave * np.sin(dist)),0],[0,jinc(np.pi * st.diam / wave * np.sin(dist))]])

                beam[n, i, j] = D@E@P
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
    num_basel = int((num_telescopes ** 2 - num_telescopes) / 2)
    M = np.zeros((num_basel, beam.shape[1], beam.shape[1], 4, 4))
    num_tel = 0
    for i, b1 in enumerate(beam):
        for j, b2 in enumerate(beam[i + 1 :]):
            m = np.zeros((beam.shape[1], beam.shape[1], 4, 4))
            for k in range(beam.shape[1]):
                for l in range(beam.shape[1]):
                    m[k,l] = np.kron(b1[k,l],np.conjugate(b1[k,l]))
            M[num_tel] = m
            print(num_tel)
            num_tel += 1

    # num_basel = int((num_telescopes ** 2 - num_telescopes) / 2)
    # M = np.reshape(M, (num_basel, beam.shape[1], beam.shape[1], 4, 4))
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

    spu = (np.ceil(u0 / cellsize) + int(img_ft.shape[1] / 2)).astype(int)
    spv = (np.ceil(v0 / cellsize) + int(img_ft.shape[1] / 2)).astype(int)

    epu = (np.ceil(u1 / cellsize) + int(img_ft.shape[1] / 2)).astype(int)
    epv = (np.ceil(v1 / cellsize) + int(img_ft.shape[1] / 2)).astype(int)

    patch = [img_ft[:,spu[i] : epu[i], spv[i] : epv[i]] for i in range(spu.shape[0])]
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
    conv = [np.array([sig.convolve2d(patch[i][0], M_ft[i][0], mode="valid"),sig.convolve2d(patch[i][1], M_ft[i][1], mode="valid"),sig.convolve2d(patch[i][2], M_ft[i][2], mode="valid"),sig.convolve2d(patch[i][3], M_ft[i][3], mode="valid")]) for i in range(len(patch))]
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
