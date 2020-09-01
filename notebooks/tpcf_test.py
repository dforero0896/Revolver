#!/usr/bin/env python

import numpy as np
from astroML.correlation import two_point, bootstrap_two_point
from sklearn.neighbors import KDTree
import pandas as pd
import os
import subprocess
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from Corrfunc.theory.DD import DD
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.io import read_catalog
from Corrfunc.utils import convert_3d_counts_to_cf
from scipy.special import legendre

def two_point_box(data, bins, box_size):
    """ From astroML two_point"""
    print("Building data KDTree", flush=True)
    KDT_D = KDTree(data)
    print("Counting DD.", flush=True)
    counts_DD = KDT_D.two_point_correlation(data, bins)
    #counts_DD = np.zeros(len(bins))
    DD = np.diff(counts_DD)
    bins_low = bins[:-1]
    bins_high = bins[1:]
    volume = 4 * np.pi * (bins_high**3 - bins_low**3) / 3
    RR = volume/(box_size**3)

    corr = DD/RR -1

    return corr, DD, RR

def main_fcfc_box():
    pre_recon = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.dat"
    pre_recon_fn = [os.path.splitext(pre_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fn = [os.path.splitext(post_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    #subprocess.check_call(["bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={pre_recon}", f"--dd={pre_recon_fn[0]}", f"--dr={pre_recon_fn[1]}", f"--rr={pre_recon_fn[2]}", f"--output={pre_recon_fn[3]}"])
    subprocess.check_call(["bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon}", f"--dd={post_recon_fn[0]}", f"--dr={post_recon_fn[1]}", f"--rr={post_recon_fn[2]}", f"--output={post_recon_fn[3]}"])
    pre_2pcf = pd.read_csv(pre_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf = pd.read_csv(post_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['mono'], label='Pre')
    ax[0].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['mono'], label='Post')
    ax[1].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['quad'])
    ax[1].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['quad'])
    plt.gcf()
    plt.savefig("notebooks/before_after_test.png")
def main_fcfc_lc():
    pre_recon = "revolver_test/Patchy-Mocks-DR12CMASSLOWZTOT-N-V6C-Portsmouth-mass_0001.dat"
    post_recon = "revolver_test/Patchy-Mocks-DR12CMASSLOWZTOT-N-V6C-Portsmouth-mass_0001_pos_shift.dat"
    pre_recon_ran = "revolver_test/Random-DR12CMASSLOWZTOT-N-V6C-x20-VT-FC.dat"
    post_recon_ran = "revolver_test/Random-DR12CMASSLOWZTOT-N-V6C-x20-VT-FC_pos_shift.dat"
    post_recon_prev = "revolver_test/GCF-Patchy-Mocks-DR12CMASSLOWZTOT-N-V6C-Portsmouth-mass_0001.2pcf"
    pre_recon_fn = [os.path.splitext(pre_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fn = [os.path.splitext(post_recon)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    if not os.path.isfile(pre_recon_fn[-1]):
        subprocess.check_call(["bin/2pcf_lc", "--conf=notebooks/fcfc_lc.conf", f"--data={pre_recon}", f"--rand={pre_recon_ran}", f"--dd={pre_recon_fn[0]}", f"--dr={pre_recon_fn[1]}", f"--rr={pre_recon_fn[2]}", f"--output={pre_recon_fn[3]}"])
    if True:#not os.path.isfile(post_recon_fn[-1]):
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_lc", "--conf=notebooks/fcfc_lc.conf", f"--data={post_recon}", f"--rand={post_recon_ran}", f"--dd={post_recon_fn[0]}", f"--dr={post_recon_fn[1]}", f"--rr={post_recon_fn[2]}", f"--output={post_recon_fn[3]}"])
    pre_2pcf = pd.read_csv(pre_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf = pd.read_csv(post_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf_prev = pd.read_csv(post_recon_prev, delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['mono'], label='Pre')
    ax[0].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['mono'], label='Post')
    ax[0].plot(post_2pcf_prev['s'], post_2pcf_prev['s']**2*post_2pcf_prev['mono'], label='Post prev')
    ax[1].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['quad'])
    ax[1].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['quad'])
    ax[0].legend(loc=0)
    plt.gcf()
    plt.savefig("notebooks/before_after_test_lc.png")
def main_fcfc_ran_box():
    pre_recon = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace_wran_pos_shift.dat"
    pre_recon_ran = "revolver_test/box_uniform_random_seed1_0-2500.dat"
    post_recon_ran = "revolver_test/box_uniform_random_seed1_0-2500_pos_shift.dat"
    pre_recon_fn = [os.path.splitext(pre_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fn = [os.path.splitext(post_recon)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    if not os.path.isfile(pre_recon_fn[-1]):
        subprocess.check_call(["bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={pre_recon}", f"--rand={pre_recon_ran}", f"--dd={pre_recon_fn[0]}", f"--dr={pre_recon_fn[1]}", f"--rr={pre_recon_fn[2]}", f"--output={pre_recon_fn[3]}", f"--count-mode=7"])
    if True:#not os.path.isfile(post_recon_fn[-1]):
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon}", f"--rand={post_recon_ran}", f"--dd={post_recon_fn[0]}", f"--dr={post_recon_fn[1]}", f"--rr={post_recon_fn[2]}", f"--output={post_recon_fn[3]}", f"--count-mode=7", f"--cf-mode=1"])
    pre_2pcf = pd.read_csv(pre_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf = pd.read_csv(post_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['mono'], label='Pre')
    ax[0].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['mono'], label='Post')
    ax[1].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['quad'])
    ax[1].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['quad'])
    ax[0].legend(loc=0)
    plt.gcf()
    plt.savefig("notebooks/before_after_test_ran_box.png")
def get_multipoles(results, bins, nmu_bins, RR):
    counts = results['npairs'].reshape(len(bins)-1, nmu_bins)
    s = results['savg'].reshape(len(bins)-1, nmu_bins)
    mu = results['mu_max'].reshape(len(bins)-1, nmu_bins)
    print(s)
    print(mu)
    mono = counts/RR[:,None] -1
    quad = mono*(2*2+1) * legendre(2)(mu)
    hexa = mono*(4*2+1) * legendre(4)(mu)
    xi0 = np.trapz(mono, dx=1./nmu_bins, axis=0)
    xi2 = np.trapz(quad, dx=1./nmu_bins, axis=0)
    xi4 = np.trapz(hexa, dx=1./nmu_bins, axis=0)
    return s[:,0], xi0, xi2, xi4
def main_corrfunc():
    pre_recon_fn = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon_fn = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.dat"
    print("Importing prerecon dataset", flush=True)
    pre_recon = pd.read_csv(pre_recon_fn, delim_whitespace=True, engine='c', names = ['x', 'y', 'z'], usecols=[0,1,2]).values
    print("Importing postrecon dataset", flush=True)
    post_recon = pd.read_csv(post_recon_fn, delim_whitespace=True, engine='c', names = ['x', 'y', 'z'], usecols=[0,1,2]).values
    box_size=2500
    bins = np.linspace(0.1, 200, 41)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    autocorr=1
    nthreads=32
    mu_max = 1.
    nmu_bins=40
    DD_post = DDsmu(autocorr, nthreads, bins, mu_max, nmu_bins, post_recon[:,0], post_recon[:,1], post_recon[:,2], periodic=True, verbose=True, boxsize=box_size, output_savg=True) 
    DD_pre = DDsmu(autocorr, nthreads, bins, mu_max, nmu_bins, pre_recon[:,0], pre_recon[:,1], pre_recon[:,2], periodic=True, verbose=True, boxsize=box_size, output_savg=True) 
    N_post = post_recon.shape[0]
    N_pre = pre_recon.shape[0]
    volume = 4 * np.pi * (bins[1:]**3 - bins[:-1]**3) / 3
    RR_post = N_post * volume * (N_post -1)/(box_size**3)
    RR_pre = N_pre * volume * (N_post -1)/(box_size**3)
    s, xi0, xi2, xi4 = get_multipoles(DD_post, bins, nmu_bins, RR_post)
    print(xi0.shape)
    print(s)
    

    
    corr_post = xi0
    corr_pre = xi2


    pre_recon_fns = [os.path.splitext(pre_recon_fn)[0]+"_corrfunc."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fns = [os.path.splitext(post_recon_fn)[0]+"_corrfunc."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    #np.savetxt(post_recon_fns[0], np.c_[bin_centers,DD_post['npairs']])
    np.savetxt(post_recon_fns[2], np.c_[bin_centers,RR_post])
    np.savetxt(post_recon_fns[-1], np.c_[bin_centers,corr_post])
    #np.savetxt(pre_recon_fns[0], np.c_[bin_centers,DD_pre['npairs']])
    np.savetxt(pre_recon_fns[2], np.c_[bin_centers,RR_pre])
    np.savetxt(pre_recon_fns[-1], np.c_[bin_centers,corr_pre])

    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(bin_centers, bin_centers**2*corr_pre, label='Pre')
    ax[0].plot(bin_centers, bin_centers**2*corr_post, label='Post')
    plt.gcf()
    plt.savefig("notebooks/before_after_test_corrfunc.png")

def main_astroml():
    pre_recon_fn = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon_fn = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.dat"
    print("Importing prerecon dataset", flush=True)
    pre_recon = pd.read_csv(pre_recon_fn, delim_whitespace=True, engine='c', names = ['x', 'y', 'z'], usecols=[0,1,2]).values
    print(f"Prerecon shape: {pre_recon.shape}")
    print("Importing postrecon dataset", flush=True)
    post_recon = pd.read_csv(post_recon_fn, delim_whitespace=True, engine='c', names = ['x', 'y', 'z'], usecols=[0,1,2]).values
    print(f"Postrecon shape: {post_recon.shape}")

    pre_recon_fns = [os.path.splitext(pre_recon_fn)[0]+"_py."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fns = [os.path.splitext(post_recon_fn)[0]+"_py."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    bins = np.linspace(0, 200, 41)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    print(f"Computing pre_recon correlation function.")
    corr_pre, DD_pre, RR_pre = two_point_box(pre_recon, bins=bins, box_size=2500)
    print(f"Computing post_recon correlation function.")
    corr_post, DD_post, RR_post = two_point_box(post_recon, bins=bins, box_size=2500)

    print(corr_pre.shape, corr_post.shape)

    np.savetxt(post_recon_fns[0], np.c_[bin_centers,DD_post])
    np.savetxt(post_recon_fns[-1], np.c_[bin_centers,corr_post])
    np.savetxt(pre_recon_fns[0], np.c_[bin_centers,DD_pre])
    np.savetxt(pre_recon_fns[-1], np.c_[bin_centers,corr_pre])
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(bin_centers, bin_centers**2*corr_pre, label='Pre')
    ax[0].plot(bin_centers, bin_centers**2*corr_post, label='Post')
    plt.gcf()
    plt.savefig("notebooks/before_after_test_py.png")

if __name__ == '__main__':
    #main_corrfunc()
    #main_fcfc_box()
    main_fcfc_lc()
    #main_fcfc_ran_box()
