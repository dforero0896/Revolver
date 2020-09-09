#!/usr/bin/env/bash 
import numpy as np
import pandas as pd
import os
import subprocess
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import legendre

def rr_analytic(bin_low_bound, bin_high_bound, box_size):
    volume = 4 * np.pi * (bin_high_bound**3 - bin_low_bound**3) / 3
    normed_volume = volume / box_size **3
    return normed_volume

def rr_analytic2d(bin_low_bound, bin_high_bound, box_size, nmu_bins=80):
    volume = 4 * np.pi * (bin_high_bound**3 - bin_low_bound**3) / 3
    normed_volume = volume / box_size **3
    mu_edges = np.linspace(0,1,nmu_bins+1)
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    mono = normed_volume[:,None] / (nmu_bins*np.ones(nmu_bins)[None,:])
    quad = mono*(2*2+1) * legendre(2)(mu)[None, :]
    hexa = mono*(4*2+1) * legendre(4)(mu)[None, :]

    RR0 = np.sum(mono, axis=1)
    RR2 = np.sum(quad, axis=1)
    RR4 = np.sum(hexa, axis=1)
    return pd.DataFrame(dict(zip(['mono', 'quad', 'hexa'], [RR0, RR2, RR4])))

def compute_2pcf(fileroot, estimator, denominator, use_rr=False):

    dd = pd.read_csv(fileroot+".dd", delim_whitespace=True, engine='c', names = ['s_low', 's_high', 'mono_pc', 'mono', 'quad_pc', 'quad'], usecols=[0,1,2,3,4,5])
    ss = pd.read_csv(fileroot+".ss", delim_whitespace=True, engine='c', names = ['s_low', 's_high', 'mono_pc', 'mono', 'quad_pc', 'quad'], usecols=[0,1,2,3,4,5])
    if denominator != 'ss' or use_rr:
        try:
            rr = pd.read_csv(denominator, delim_whitespace=True, engine='c', names = ['s_low', 's_high', 'mono_pc', 'mono', 'quad_pc', 'quad'], usecols=[0,1,2,3,4,5])
            
        except OSError:
            print("RR counts file not found, falling back to analytical.")
            rr = rr_analytic2d(dd['s_low'], dd['s_high'], 2500)
        denom = rr['mono'].values[:,None]
    else:
        denom = ss['mono'].values[:,None]


    if not use_rr:
        if estimator == 'natural':
            numer = dd - ss
        elif estimator == 'ls':
            ds = pd.read_csv(fileroot+".ds", delim_whitespace=True, engine='c', names = ['s_low', 's_high', 'mono_pc', 'mono', 'quad_pc', 'quad'], usecols=[0,1,2,3,4,5])
            numer = dd - 2 * ds + ss
    else:
        if estimator == 'natural':
            numer = dd - rr
        elif estimator == 'ls':
            ds = pd.read_csv(fileroot+".ds", delim_whitespace=True, engine='c', names = ['s_low', 's_high', 'mono_pc', 'mono', 'quad_pc', 'quad'], usecols=[0,1,2,3,4,5])
            numer = dd - 2 * ds + rr
 
    
    tpcf = numer/denom
    tpcf['s'] = 0.5 * (dd['s_low'] + dd['s_high'])

    return tpcf

def plot_2pcf(tpcf, ax, **kwargs):

    s = tpcf['s']
    ax[0].plot(s, s**2*tpcf['mono'], **kwargs)
    try:
        label = kwargs.pop('label')
    except KeyError:
        pass
    ax[1].plot(s, s**2*tpcf['quad'], **kwargs)

fig, ax = plt.subplots(1,2, figsize=(20,10))

plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_pos_shift', 'natural', 'ss'), ax, label='No rand natural ss')
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_pos_shift', 'ls', 'ss'), ax, label = 'No rand ls ss')
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_pos_shift', 'natural', 'rr'), ax, label='No rand natural rranalyt')
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_wran_pos_shift', 'natural', 'ss'), ax, label='Ran ss natural')
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_wran_pos_shift', 'natural', 'rr'), ax, label='Ran rranalyt natural')
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_wran_pos_shift', 'ls', 'ss'), ax, label='Ran ss ls')
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_wran_pos_shift', 'ls', 'rr'), ax, label='Ran rranalyt ls')
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_wran_nopad_pos_shift', 'ls', 'ss'), ax, label='Ran ss nopad ls', ls = ':', lw=4)
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_wran_nopad_pos_shift', 'ls', 'rr'), ax, label='Ran rranalyt nopad ls', ls = ':', lw=4)

plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_wran_nopad_pos_shift', 'ls', 'rr', use_rr=True), ax, label='Ran only rranalyt nopad ls', ls='--', lw=3)
plot_2pcf(compute_2pcf('tests/CATALPTCICz0.466G960S1005638091_zspace_wran_nopad_pos_shift', 'natural', 'rr', use_rr=True), ax, label='Ran only rranalyt nopad natural', ls='--', lw=3)
ax[0].legend()
fig.savefig('notebooks/estimators.png', dpi=200)s