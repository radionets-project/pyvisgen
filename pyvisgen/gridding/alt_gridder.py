import astropy.constants as const
import numpy as np
from scipy.special import p_roots

speedoflight = const.c.value


def init(
    uvw, freq, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding, ms=None
):
    ofac = 2
    x, y = np.meshgrid(
        *[-ss / 2 + np.arange(ss) for ss in (nxdirty, nydirty)], indexing="ij"
    )
    x *= pixsizex
    y *= pixsizey
    eps = x**2 + y**2
    nm1 = -eps / (np.sqrt(1.0 - eps) + 1.0)
    ng = ofac * nxdirty, ofac * nydirty
    supp = int(np.ceil(np.log10(1 / epsilon * (3 if do_wgridding else 2)))) + 1
    kernel = es_kernel(supp, 2.3 * supp)
    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1)).reshape(-1, 3)
    conjind = uvw[:, 2] < 0
    uvw[conjind] *= -1
    u, v, w = uvw.T
    if do_wgridding:
        wmin, wmax = np.min(w), np.max(w)
        dw = 1 / ofac / np.max(np.abs(nm1)) / 2
        nwplanes = int(np.ceil((wmax - wmin) / dw + supp)) if do_wgridding else 1
        w0 = (wmin + wmax) / 2 - dw * (nwplanes - 1) / 2
    else:
        nwplanes, w0, dw = 1, None, None
    gridcoord = [np.linspace(-0.5, 0.5, nn, endpoint=False) for nn in ng]
    slc0, slc1 = (
        slice(nxdirty // 2, nxdirty * 3 // 2),
        slice(nydirty // 2, nydirty * 3 // 2),
    )
    u *= pixsizex
    v *= pixsizey
    if ms is not None:
        ms[conjind] = ms[conjind].conjugate()
        return u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, ms
    return u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, conjind


class Kernel:
    def __init__(self, supp, func):
        self._func = func
        self._supp = supp

    def ft(self, x):
        x = x * self._supp * np.pi
        nroots = 2 * self._supp
        if self._supp % 2 == 0:
            nroots += 1
        q, weights = p_roots(nroots)
        ind = q > 0
        weights = weights[ind]
        q = q[ind]
        kq = np.outer(x, q) if len(x.shape) == 1 else np.einsum("ij,k->ijk", x, q)
        arr = np.sum(weights * self._raw(q) * np.cos(kq), axis=-1)
        return self._supp * arr

    def __call__(self, x):
        return self._raw(x / self._supp * 2)

    def _raw(self, x):
        ind = np.logical_and(x <= 1, x >= -1)
        res = np.zeros_like(x)
        res[ind] = self._func(x[ind])
        return res


def es_kernel(supp, beta):
    return Kernel(supp, lambda x: np.exp(beta * (pow((1 - x) * (1 + x), 0.5) - 1)))


def ms2dirty_python_fast(
    uvw, freq, vis, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding
):
    u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, vis = init(
        uvw, freq, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding, vis
    )
    supp = kernel._supp
    for ii in range(nwplanes):
        grid = np.zeros(ng, dtype=vis.dtype)
        for uu, vv, vis in zip(
            u, v, vis * kernel(ii - (w - w0) / dw) if do_wgridding else vis
        ):
            if vis == 0:
                continue
            ratposx = (uu * ng[0]) % ng[0]
            ratposy = (vv * ng[1]) % ng[1]
            xle = int(np.round(ratposx)) - supp // 2
            yle = int(np.round(ratposy)) - supp // 2
            # pos = np.arange(0, supp)
            # xkernel = kernel(pos - ratposx + xle)
            # ykernel = kernel(pos - ratposy + yle)
            for xx in range(supp):
                foo = vis  # * xkernel[xx]
                myxpos = (xle + xx) % ng[0]
                for yy in range(supp):
                    myypos = (yle + yy) % ng[1]
                    grid[myxpos, myypos] += foo  # * ykernel[yy]
    #     loopim = np.fft.fftshift(np.fft.ifft2(grid)*np.prod(ng))
    #     loopim = loopim[slc0, slc1]
    #     if do_wgridding:
    #         loopim *= np.exp(-2j*np.pi*nm1*(w0+ii*dw))
    #     im += loopim.real
    # im /= kernel.ft(gridcoord[0][slc0])[:, None]
    # im /= kernel.ft(gridcoord[1][slc1])
    # if do_wgridding:
    #     im /= (nm1+1)*kernel.ft(nm1*dw)
    return grid


def get_npixdirty(uvw, freq, fov_deg, mask):
    speedOfLight = 299792458.0
    bl = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2 + uvw[:, 2] ** 2)
    bluvw = bl.reshape((-1, 1)) * freq.reshape((1, -1)) / speedOfLight
    maxbluvw = np.max(bluvw * mask)
    minsize = int((2 * fov_deg * np.pi / 180 * maxbluvw)) + 1
    return minsize + (minsize % 2)  # make even
