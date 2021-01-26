from dataclasses import dataclass
from astropy import units as un
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import numpy as np
from scipy.special import j1
import scipy.constants as const
import scipy.signal as sig

@dataclass
class baseline():
    name: str
    u: float
    v: float
    w: float
    valid: bool


def get_baselines(src_crd, time, array):
    #calculate GHA
    lst = time.sidereal_time('apparent', 'greenwich')
    ha = lst - src_crd.ra

    #calculate elevations
    el_st = [src_crd.transform_to(AltAz(obstime=time,location=EarthLocation.from_geocentric(st.x,st.y,st.z,unit=un.m))).alt.degree for st in array]

    #calculate baselines
    baselines = []
    for i, st1 in enumerate(array):
        for j, st2 in enumerate(array[i+1:]):
            delta_x = st1.x - st2.x
            delta_y = st1.y - st2.y
            delta_z = st1.z - st2.z

            #coord transformation uvw
            u = np.sin(ha)*delta_x + np.cos(ha)*delta_y
            v = -np.sin(src_crd.ra)*np.cos(ha)*delta_x + np.sin(src_crd.ra)*np.sin(ha)*delta_y + np.cos(src_crd.ra)*delta_z
            w = np.cos(src_crd.ra)*np.cos(ha)*delta_x - np.cos(src_crd.ra)*np.sin(ha)*delta_y + np.sin(src_crd.ra)*delta_z

            #check baseline
            valid = True
            if el_st[i] < st1.el_low or el_st[i] > st1.el_high or el_st[i+j+1] < st2.el_low or el_st[i+j+1] > st2.el_high:
                valid = False

            #collect baselines
            baselines.append(baseline(st1.name + '-' + st2.name, u, v, w, valid))
            
    return baselines


def create_bgrid(fov, samples, src_crd):
    spacing = fov/samples

    bgrid = np.zeros((samples,samples,2))


    sr = np.sin(src_crd.ra.rad)
    cr = np.cos(src_crd.ra.rad)
    sd = np.sin(src_crd.dec.rad)
    cd = np.cos(src_crd.dec.rad)

    for i in range(samples):
        for j in range(samples):
            delx = (i-samples/2.0)*spacing
            dely = (j-samples/2.0)*spacing

            alpha = np.arctan(delx/(cd-dely*sd))+src_crd.ra.rad
            delta = np.arcsin((sd+dely*cd)/np.sqrt(1+delx**2+dely**2))

            bgrid[i,j,0] = alpha
            bgrid[i,j,1] = delta

    return bgrid

def jinc(x):
    if x == 0:
        return 1
    return 2*j1(x)/x

def get_beam(src_crd, array, bgrid, freq):
    beam = np.zeros((len(array), bgrid.shape[0], bgrid.shape[0]))
    for i in range(bgrid.shape[0]):
        for j in range(bgrid.shape[0]):
            crd = SkyCoord(ra=bgrid[i,j,0], dec=bgrid[i,j,1], unit=(un.rad, un.rad))
            dist = src_crd.separation(crd)
            wave = const.c/freq
            for n, st in enumerate(array):
                beam[n, i, j] = jinc(np.pi*st.diam/wave*np.sin(dist))
    
    return beam


def get_beam_conv(beam, num_telescopes):
    M = np.array([])
    for i, j1 in enumerate(beam):
            for j, j2 in enumerate(beam[i+1:]):
                M = np.append(M, j1*j2)
                
    num_basel = int((num_telescopes**2-num_telescopes)/2)
    M = np.reshape(M,(num_basel, beam.shape[1], beam.shape[1]))
    M_ft = np.fft.fft2(M)

    return M_ft
    

def get_uvPatch(img_ft, bgrid, freq, bw, start_uv, stop_uv, cellsize):
    start_freq = freq-bw/2
    stop_freq = freq+bw/2

    u_11 = [st.u*start_freq/const.c for st in start_uv] #start freq, start uv
    v_11 = [st.v*start_freq/const.c for st in start_uv] #start freq, start uv
    u_12 = [st.u*stop_freq/const.c for st in start_uv] #stop freq, start uv
    v_12 = [st.v*stop_freq/const.c for st in start_uv] #stop freq, start uv
    u_21 = [st.u*start_freq/const.c for st in stop_uv] #start freq, stop uv
    v_21 = [st.v*start_freq/const.c for st in stop_uv] #start freq, stop uv
    u_22 = [st.u*stop_freq/const.c for st in stop_uv] #stop freq, stop uv
    v_22 = [st.v*stop_freq/const.c for st in stop_uv] #stop freq, stop uv

    #get corners of rectangular patch
    u_max = max(max(u_11,u_12),max(u_21,u_22))
    v_max = max(max(v_11,v_12),max(v_21,v_22))
    u_min = min(min(u_11,u_12),min(u_21,u_22))
    v_min = min(min(v_11,v_12),min(v_21,v_22))

    #get patch
    udim = [abs(u_max[i]-u_min[i])+ 4.0*cellsize+bgrid.shape[1]*cellsize for i in range(len(u_max))]
    vdim = [abs(v_max[i]-v_min[i])+ 4.0*cellsize+bgrid.shape[1]*cellsize for i in range(len(u_max))]


    npu = np.ceil(udim/cellsize)
    npv = np.ceil(vdim/cellsize)

    u0 = [b.u for b in start_uv]
    v0 = [b.v for b in start_uv]

    u1 = np.array(u0) + np.array(udim)
    v1 = np.array(v0) + np.array(vdim)

    spu = (np.ceil(u0/cellsize) + int(img_ft.shape[0]/2)).astype(int)
    spv = (np.ceil(v0/cellsize) + int(img_ft.shape[0]/2)).astype(int)

    epu = (np.ceil(u1/cellsize) + int(img_ft.shape[0]/2)).astype(int)
    epv = (np.ceil(v1/cellsize) + int(img_ft.shape[0]/2)).astype(int)

    patch = [img_ft[spu[i]:epu[i],spv[i]:epv[i]] for i in range(spu.shape[0])]

    return patch


def conv(patch, M_ft):
    conv = [sig.convolve2d(patch[i], M_ft[i], mode='valid') for i in range(len(patch))]
    
    return conv


def integrate(conv):
    vis = [np.sum(conv[i]) for i in range(len(conv))]

    return vis