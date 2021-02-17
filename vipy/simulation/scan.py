from dataclasses import dataclass
from astropy import units as un
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import numpy as np
from scipy.special import j1
import scipy.constants as const
import scipy.signal as sig
from astroplan import Observer
import vipy.simulation.layouts.layouts as layouts


@dataclass
class baseline:
    name: str
    st1: layouts.Station
    st2: layouts.Station
    u: float
    v: float
    w: float
    valid: bool

    def baselineNum(self):
        #baseline = 256*ant1 + ant2 + (array#-1)/100
        return 256*(self.st1.st_num+1) + self.st2.st_num+1


def get_baselines(src_crd, time, array):
    """Calculates baselines from source coordinates and time of observation for
    every antenna station in station array. 
    

    Parameters
    ----------
    src_crd : astropy SkyCoord object 
        ra and dec of source location / pointing center
    time : astropy time object
        time of observation
    array : list of dataclass object Station
        station information

    Returns
    -------
    dataclass object
        baselines between telescopes with visinility flags
    """
    # calculate GHA, Greenwich as reference for EHT
    lst = time.sidereal_time("apparent", "greenwich")
    ha = lst - src_crd.ra

    # calculate elevations
    el_st = [
        src_crd.transform_to(
            AltAz(
                obstime=time,
                location=EarthLocation.from_geocentric(st.x, st.y, st.z, unit=un.m),
            )
        ).alt.degree
        for st in array
    ]

    # calculate baselines
    baselines = []
    for i, st1 in enumerate(array):
        for j, st2 in enumerate(array[i + 1 :]):
            delta_x = st1.x - st2.x
            delta_y = st1.y - st2.y
            delta_z = st1.z - st2.z

            # coord transformation uvw
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

            # check baseline
            valid = True
            if (
                el_st[i] < st1.el_low
                or el_st[i] > st1.el_high
                or el_st[i + j + 1] < st2.el_low
                or el_st[i + j + 1] > st2.el_high
            ):
                valid = False

            #collect baselines
            baselines.append(baseline(st1.name + '-' + st2.name, st1, st2, u, v, w, valid))
            
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

    sr = np.sin(src_crd.ra.rad)  # calculated but unused
    cr = np.cos(src_crd.ra.rad)  # calculated but unused
    sd = np.sin(src_crd.dec.rad)
    cd = np.cos(src_crd.dec.rad)

    for i in range(samples):
        for j in range(samples):
            delx = (i - samples / 2.0) * spacing
            dely = (j - samples / 2.0) * spacing

            alpha = np.arctan(delx / (cd - dely * sd)) + src_crd.ra.rad
            delta = np.arcsin((sd + dely * cd) / np.sqrt(1 + delx ** 2 + dely ** 2))

            bgrid[i, j, 0] = alpha
            bgrid[i, j, 1] = delta

    return bgrid


def lm(grid, src_crd):
    lm = np.zeros(grid.shape)
    ra = grid[...,0]
    dec = grid[...,1]
    lm[...,0] = np.cos(src_crd.dec.rad) * np.sin(src_crd.ra.rad-ra)
    lm[...,1] = np.sin(dec)*np.cos(src_crd.dec.rad) - np.cos(dec)*np.sin(src_crd.dec.rad)*np.cos(src_crd.ra.rad-ra)
    return lm


def getJones(lm, baselines, wave, time, src_crd, array):
    # J = D E P K
    # for every entry in baselines exists one station pair (st1,st2)
    # st1 has a jones matrix, st2 has a jones matrix
    # Calculate Jones matrices for every baseline
    JJ = np.zeros((lm.shape[0],lm.shape[1],len(baselines),4,4),dtype=complex)
    D = np.eye(2)

    # parallactic angle
    beta = [Observer(EarthLocation(st.x*un.m,st.y*un.m,st.z*un.m)).parallactic_angle(time, src_crd) for st in array]

    for i, b in enumerate(baselines):
        # calculate E Matrices
        E1 = getE(lm, b.st1, wave)
        E2 = getE(lm, b.st2, wave)

        # calculate P Matrices
        P1 = getP(beta[b.st1.st_num])
        P2 = getP(beta[b.st2.st_num])
        

        # calculate K matrices
        K1 = getK(b,lm,wave)
        K2 = K1


        for k in range(lm.shape[0]):
            for l in range(lm.shape[1]):
                JJ[k,l,i] = np.kron(D@E1[k,l]@P1,(D@E2[k,l]@P2).conj().T)*K1[k,l]
    return JJ

def getE(lm, station, wave):
    # calculate matrix E for every point in grid
    E = np.zeros((lm.shape[0],lm.shape[1],2,2))
    E[...,0,0] = jinc(np.pi * station.diam / wave * np.sin(np.sqrt(lm[...,0]**2+lm[...,1]**2)))
    E[...,1,1] = E[...,0,0]
    return E


def getP(beta):
    #calculate matrix P with parallactic angle beta
    P = np.zeros((2,2))

    P[0,0] = np.cos(beta)
    P[0,1] = -np.sin(beta)
    P[1,0] = np.sin(beta)
    P[1,1] = np.cos(beta)
    return P


def getK(b,lm,wave):
    # K = np.zeros((lm.shape[0],lm.shape[1],2,2),dtype=complex)
    # K[...,0,0] = np.exp(1j*2*np.pi*(b.u*lm[...,0]+b.v*lm[...,1])/wave)
    # K[...,0,1] = 0
    # K[...,1,0] = 0
    # K[...,1,1] = K[...,0,0]
    return np.exp(1j*2*np.pi*(b.u*lm[...,0]+b.v*lm[...,1])/wave)


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
