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

def main_ext_tpcf():
    pre_recon = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon = "revolver_test/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.dat"
    pre_recon_fn = [os.path.splitext(pre_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fn = [os.path.splitext(post_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    #subprocess.check_call(["bin/2pcf", "--conf=notebooks/fcfc.conf", f"--data={pre_recon}", f"--dd={pre_recon_fn[0]}", f"--dr={pre_recon_fn[1]}", f"--rr={pre_recon_fn[2]}", f"--output={pre_recon_fn[3]}"])
    #subprocess.check_call(["bin/2pcf", "--conf=notebooks/fcfc.conf", f"--data={post_recon}", f"--dd={post_recon_fn[0]}", f"--dr={post_recon_fn[1]}", f"--rr={post_recon_fn[2]}", f"--output={post_recon_fn[3]}"])
    pre_2pcf = pd.read_csv(pre_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf = pd.read_csv(post_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['mono'], label='Pre')
    ax[0].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['mono'], label='Post')
    ax[1].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['quad'])
    ax[1].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['quad'])
    plt.gcf()
    plt.savefig("notebooks/before_after_test.png")
def main_py_tpcf():
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
    main_ext_tpcf()
