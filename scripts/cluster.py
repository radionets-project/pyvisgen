import pyvisgen.simulation.utils as ut
import pyvisgen.layouts.layouts as layouts
import astropy.constants as const
from astropy import units as un
import time as t
import numpy as np
import pyvisgen.simulation.scan as scan
import torch
from astropy.io import fits
from pyvisgen.simulation.scan import get_valid_baselines
from astropy.time import Time
import pyvisgen.fits.writer as writer
from dataclasses import dataclass

import sys

from PIL import Image
# jobid = sys.getenv('SLURM_ARRAY_TASK_ID')

@dataclass
class Visibilities:
    I: [complex]
    Q: [complex]
    U: [complex]
    V: [complex]
    num: [int]
    scan: [int]
    base_num: [int]
    u: [float]
    v: [float]
    w: [float]
    date: [float]
    _date: [float]

    def __getitem__(self, i):
        baseline = Vis(
            self.I[i],
            self.Q[i],
            self.U[i],
            self.V[i],
            self.num[i],
            self.scan[i],
            self.base_num[i],
            self.u[i],
            self.v[i],
            self.w[i],
            self.date[i],
            self._date[i],
        )
        return baseline

    def get_values(self):
        return np.array([self.I, self.Q, self.U, self.V])
    
    def add(self, visibilities):
        self.I = np.concatenate([self.I, visibilities.I])
        self.Q = np.concatenate([self.Q, visibilities.Q])
        self.U = np.concatenate([self.U, visibilities.U])
        self.V = np.concatenate([self.V, visibilities.V])
        self.num = np.concatenate([self.num, visibilities.num])
        self.scan = np.concatenate([self.scan, visibilities.scan])
        self.base_num = np.concatenate([self.base_num, visibilities.base_num])
        self.u = np.concatenate([self.u, visibilities.u])
        self.v = np.concatenate([self.v, visibilities.v])
        self.w = np.concatenate([self.w, visibilities.w])
        self.date = np.concatenate([self.date, visibilities.date])
        self._date = np.concatenate([self._date, visibilities._date])
        

@dataclass
class Vis:
    I: complex
    Q: complex
    U: complex
    V: complex
    num: int
    scan: int
    base_num: int
    u: float
    v: float
    w: float
    date: float
    _date: float


# set number of threads

def loop():
    torch.set_num_threads(48)

    # read config
    rc = ut.read_config("../config/vlba.toml")
    array_layout = layouts.get_array_layout('vlba')
    src_crd = rc['src_coord']

    wave1 = const.c/((float(rc['channel'].split(':')[0])-float(rc['channel'].split(':')[1])/2)*10**6/un.second)/un.meter
    wave2 = const.c/((float(rc['channel'].split(':')[0])+float(rc['channel'].split(':')[1])/2)*10**6/un.second)/un.meter

    # calculate rd, lm
    rd = scan.rd_grid(rc['fov_size']*np.pi/(3600*180),256, src_crd)
    lm = scan.lm_grid(rd, src_crd)

    # calculate time steps
    time = ut.calc_time_steps(rc)

    # open image
    img = np.asarray(Image.open('/net/nfshome/home/sfroese/150.jpg'))
    norm = np.sum(img)
    img = img/norm
    img = torch.tensor(img)
    I = torch.zeros((img.shape[0],img.shape[1],4), dtype=torch.cdouble)
    I[...,0] = img
    I[...,1] = img

    stat_num = array_layout.st_num.shape[0]
    base_num = int(stat_num * (stat_num - 1) / 2)

    # calculate vis
    visibilities = Visibilities([], [], [], [], [], [], [], [], [], [], [], [])
    vis_num = np.zeros(1)
#i in total number of scans
    for i in range(72):
        # i=30
        t = time[i*31:(i+1)*31]
        baselines = scan.get_baselines(src_crd, t, array_layout)
        
        valid = baselines.valid.reshape(-1, base_num)
        mask = np.array(valid[:-1]).astype(bool) & np.array(valid[1:]).astype(bool)
        u = baselines.u.reshape(-1, base_num)
        v = baselines.v.reshape(-1, base_num)
        w = baselines.w.reshape(-1, base_num)
        base_valid = np.arange(len(baselines.u)).reshape(-1, base_num)[:-1][mask]
        u_valid = u[:-1][mask]
        v_valid = v[:-1][mask]
        w_valid = w[:-1][mask]
        date = np.repeat((t[:-1]+rc['corr_int_time']*un.second/2).jd.reshape(-1, 1), base_num, axis=1)[mask] 
        _date = np.zeros(len(u_valid))    
        
        
        X1 = scan.corrupted(lm, baselines, wave1, time, src_crd, array_layout, I, rd)
        if X1.shape[0] == 1:
            continue
        X2 = scan.corrupted(lm, baselines, wave1, time, src_crd, array_layout, I, rd)
        
        vis_num = np.arange(X1.shape[2]//2) + 1 + vis_num.max()
        

        int_values = scan.integrate(X1, X2)
        del X1, X2
        int_values = int_values.reshape(-1,4)
        
        vis = Visibilities(
            int_values[:, 0],
            int_values[:, 1],
            int_values[:, 2],
            int_values[:, 3],
            vis_num,
            np.repeat(i+1, len(vis_num)),
            np.array([baselines[i].baselineNum() for i in base_valid]),
            u_valid,
            v_valid,
            w_valid,
            date,
            _date,      
        )
        
        visibilities.add(vis)
        # break
    return visibilities

# save vis
hdu_list = writer.create_hdu_list(loop(), rc)
hdu_list.writeto('test07_vlba.fits', overwrite=True)