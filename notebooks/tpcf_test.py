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
    pre_recon = "tests/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon = "tests/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.dat"
    pre_recon_fn = [os.path.splitext(pre_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fn = [os.path.splitext(post_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    #subprocess.check_call(["bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={pre_recon}", f"--dd={pre_recon_fn[0]}", f"--dr={pre_recon_fn[1]}", f"--rr={pre_recon_fn[2]}", f"--output={pre_recon_fn[3]}"])
    subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon}", f"--dd={post_recon_fn[0]}", f"--dr={post_recon_fn[1]}", f"--rr={post_recon_fn[2]}", f"--output={post_recon_fn[3]}"])
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
    pre_recon = "tests/Patchy-Mocks-DR12CMASSLOWZTOT-N-V6C-Portsmouth-mass_0595.dat"
    post_recon = "tests/Patchy-Mocks-DR12CMASSLOWZTOT-N-V6C-Portsmouth-mass_0595_pos_shift.dat"
    post_recon_used =  "tests/Patchy-Mocks-DR12CMASSLOWZTOT-N-V6C-Portsmouth-mass_0595.xyzwzi.recon"
    pre_recon_ran = "tests/Random-DR12CMASSLOWZTOT-N-V6C-x20-VT-FC.dat"
    post_recon_ran = "tests/Random-DR12CMASSLOWZTOT-N-V6C-x20-VT-FC_pos_shift.dat"
    post_recon_used_ran = "tests/Random-DR12CMASSLOWZTOT-N-V6C-x20_0595.xyzwzi.recon"
    #post_recon_prev = "tests/GCF-Patchy-Mocks-DR12CMASSLOWZTOT-N-V6C-Portsmouth-mass_0595.2pcf" #These two do not coincide (should they?)
    post_recon_prev = "tests/2PCF/2PCF_Patchy-Mocks-DR12CMASSLOWZTOT-N-V6C-Portsmouth-mass_0595_recon_z0.2z0.5.2pcf"
    pre_recon_fn = [os.path.splitext(pre_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fn = [os.path.splitext(post_recon)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    post_recon_used_fn = [os.path.splitext(post_recon_used)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    if not os.path.isfile(pre_recon_fn[-1]):
        subprocess.check_call(["bin/2pcf_lc", "--conf=notebooks/fcfc_lc.conf", f"--data={pre_recon}", f"--rand={pre_recon_ran}", f"--dd={pre_recon_fn[0]}", f"--dr={pre_recon_fn[1]}", f"--rr={pre_recon_fn[2]}", f"--output={pre_recon_fn[3]}"])
    if not os.path.isfile(post_recon_used_fn[-1]):
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_lc", "--conf=notebooks/fcfc_lc.conf", f"--data={post_recon_used}", f"--rand={post_recon_used_ran}", f"--dd={post_recon_used_fn[0]}", f"--dr={post_recon_used_fn[1]}", f"--rr={post_recon_used_fn[2]}", f"--output={post_recon_used_fn[3]}"])
    if True:#not os.path.isfile(post_recon_fn[-1]):
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_lc", "--conf=notebooks/fcfc_lc.conf", f"--data={post_recon}", f"--rand={post_recon_ran}", f"--dd={post_recon_fn[0]}", f"--dr={post_recon_fn[1]}", f"--rr={post_recon_fn[2]}", f"--output={post_recon_fn[3]}"])
        #subprocess.check_call(["bin/2pcf_lc", "--conf=notebooks/fcfc_lc.conf", f"--data={post_recon}", f"--rand={post_recon_ran}", f"--dd={post_recon_fn[0]}", f"--dr={post_recon_fn[1]}", f"--rr={post_recon_fn[2]}", f"--output={post_recon_fn[3]}"])
    pre_2pcf = pd.read_csv(pre_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf = pd.read_csv(post_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf_used = pd.read_csv(post_recon_used_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf_prev = pd.read_csv(post_recon_prev, delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['mono'], label='Pre')
    ax[0].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['mono'], label='Post')
    ax[0].plot(post_2pcf_prev['s'], post_2pcf_prev['s']**2*post_2pcf_prev['mono'], label='Post (Cheng test)')
    ax[0].plot(post_2pcf_used['s'], post_2pcf_used['s']**2*post_2pcf_used['mono'], label='Post (Used in analysis)')
    ax[1].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['quad'])
    ax[1].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['quad'])
    ax[1].plot(post_2pcf_prev['s'], post_2pcf_prev['s']**2*post_2pcf_prev['quad'])
    ax[1].plot(post_2pcf_used['s'], post_2pcf_used['s']**2*post_2pcf_used['quad'])
    ax[0].legend(loc=0)
    plt.gcf()
    plt.savefig("notebooks/before_after_test_lc.png")
def main_fcfc_ran_box():
    #Data redshift space
    pre_recon = "tests/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon = "tests/CATALPTCICz0.466G960S1005638091_zspace_wran_pos_shift.dat"
    post_recon_big = "tests/CATALPTCICz0.466G960S1005638091_zspace_wran_BIG_pos_shift.dat"
    post_recon_nopad = "tests/CATALPTCICz0.466G960S1005638091_zspace_wran_nopad_pos_shift.dat"
    post_recon_noran = "tests/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.dat"

    #Data real space
    pre_recon_r = "tests/CATALPTCICz0.466G960S1005638091.dat"
    post_recon_nopad_niter3_r = "tests/CATALPTCICz0.466G960S1005638091_pos_shift.dat"
    post_recon_nopad_niter1_r = "tests/CATALPTCICz0.466G960S1005638091_wran_nopad_niter1_pos_shift.dat"
    post_recon_nopad_norsd_r = "tests/CATALPTCICz0.466G960S1005638091_wran_nopad_norsd_pos_shift.dat"
    

    # Randoms redshift space
    pre_recon_ran = "tests/box_uniform_random_seed1_0-2500.dat"
    post_recon_ran = "tests/box_uniform_random_seed1_0-2500_pos_shift.dat"
    post_recon_ran_nopad = "tests/box_uniform_random_seed1_0-2500_nopad_pos_shift.dat"
    post_recon_ran_big = "tests/box_uniform_random_seed2_0-2500_BIG_pos_shift.dat"

    # Randoms real space
    pre_recon_ran_r = "tests/box_uniform_random_seed1_0-2500.dat"
    post_recon_ran_nopad_niter3_r = "tests/CATALPTCICz0.466G960S1005638091.ran_pos_shift.dat"
    post_recon_ran_nopad_niter1_r = "tests/box_uniform_random_seed1_0-2500_nopad_niter1_r_pos_shift.dat"
    post_recon_ran_nopad_norsd_r = "tests/box_uniform_random_seed1_0-2500_nopad_norsd_r_pos_shift.dat"

    # Filenames redshift space
    pre_recon_fn = [os.path.splitext(pre_recon)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_fn = [os.path.splitext(post_recon)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    post_recon_big_fn = [os.path.splitext(post_recon_big)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    post_recon_nopad_fn = [os.path.splitext(post_recon_nopad)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    post_recon_noran_fn = [os.path.splitext(post_recon_noran)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]

    # Filenames real space
    pre_recon_r_fn = [os.path.splitext(pre_recon_r)[0]+"."+s for s in ['dd', 'dr', 'rr', '2pcf']]
    post_recon_nopad_niter3_r_fn = [os.path.splitext(post_recon_nopad_niter3_r)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    post_recon_nopad_niter1_r_fn = [os.path.splitext(post_recon_nopad_niter1_r)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]
    post_recon_nopad_norsd_r_fn = [os.path.splitext(post_recon_nopad_norsd_r)[0]+"."+s for s in ['dd', 'ds', 'ss', '2pcf']]

    # Pre recon 2pcf zspace
    if not os.path.isfile(pre_recon_fn[-1]):
        print("Computing pre recon 2pcf")
        subprocess.check_call(["bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={pre_recon}", f"--rand={pre_recon_ran}", f"--dd={pre_recon_fn[0]}", f"--dr={pre_recon_fn[1]}", f"--rr={pre_recon_fn[2]}", f"--output={pre_recon_fn[3]}", f"--count-mode=7"])
    # Post recon 2pcf small rand zspace
    if not os.path.isfile(post_recon_fn[-1]):
        print("Computing post recon 2pcf small rand")
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon}", f"--rand={post_recon_ran}", f"--dd={post_recon_fn[0]}", f"--dr={post_recon_fn[1]}", f"--rr={post_recon_fn[2]}", f"--output={post_recon_fn[3]}", f"--count-mode=7", f"--cf-mode=1"])
    # Post recon 2pcf small rand nopad zspace
    if not os.path.isfile(post_recon_nopad_fn[-1]):
        print("Computing post recon 2pcf small rand nopad")
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon_nopad}", f"--rand={post_recon_ran_nopad}", f"--dd={post_recon_nopad_fn[0]}", f"--dr={post_recon_nopad_fn[1]}", f"--rr={post_recon_nopad_fn[2]}", f"--output={post_recon_nopad_fn[3]}", f"--count-mode=7", f"--cf-mode=1"])
    # Post recon 2pcf big rand zspace
    if not os.path.isfile(post_recon_big_fn[-1]):
        print("Computing post recon 2pcf big rand")
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon_big}", f"--rand={post_recon_ran_big}", f"--dd={post_recon_big_fn[0]}", f"--dr={post_recon_big_fn[1]}", f"--rr={post_recon_big_fn[2]}", f"--output={post_recon_big_fn[3]}", f"--count-mode=7", f"--cf-mode=1"])


    # Pre recon 2pcf rspace
    if not os.path.isfile(pre_recon_r_fn[-1]):
        print("Computing pre recon 2pcf")
        subprocess.check_call(["bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={pre_recon_r}", f"--rand={pre_recon_ran_r}", f"--dd={pre_recon_r_fn[0]}", f"--dr={pre_recon_r_fn[1]}", f"--rr={pre_recon_r_fn[2]}", f"--output={pre_recon_r_fn[3]}"])
    # Post recon 2pcf small rand nopad rspace niter3
    if not os.path.isfile(post_recon_nopad_niter3_r_fn[-1]):
        print("Computing post recon 2pcf small rand nopad niter3")
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon_nopad_niter3_r}", f"--rand={post_recon_ran_nopad_niter3_r}", f"--dd={post_recon_nopad_niter3_r_fn[0]}", f"--dr={post_recon_nopad_niter3_r_fn[1]}", f"--rr={post_recon_nopad_niter3_r_fn[2]}", f"--output={post_recon_nopad_niter3_r_fn[3]}", f"--count-mode=7", f"--cf-mode=1"])
    # Post recon 2pcf small rand nopad rspace niter1
    if not os.path.isfile(post_recon_nopad_niter1_r_fn[-1]):
        print("Computing post recon 2pcf small rand nopad niter1")
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon_nopad_niter1_r}", f"--rand={post_recon_ran_nopad_niter1_r}", f"--dd={post_recon_nopad_niter1_r_fn[0]}", f"--dr={post_recon_nopad_niter1_r_fn[1]}", f"--rr={post_recon_nopad_niter1_r_fn[2]}", f"--output={post_recon_nopad_niter1_r_fn[3]}", f"--count-mode=7", f"--cf-mode=1"])
    # Post recon 2pcf small rand nopad rspace norsd
    if not os.path.isfile(post_recon_nopad_norsd_r_fn[-1]):
        print("Computing post recon 2pcf small rand nopad norsd")
        subprocess.check_call(["srun", "-p", "p5", "-n1", "-c32", "bin/2pcf_box", "--conf=notebooks/fcfc.conf", f"--data={post_recon_nopad_norsd_r}", f"--rand={post_recon_ran_nopad_norsd_r}", f"--dd={post_recon_nopad_norsd_r_fn[0]}", f"--dr={post_recon_nopad_norsd_r_fn[1]}", f"--rr={post_recon_nopad_norsd_r_fn[2]}", f"--output={post_recon_nopad_norsd_r_fn[3]}", f"--count-mode=7", f"--cf-mode=1"])

    
    # Read computed 2pcfs zspace
    pre_2pcf = pd.read_csv(pre_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_2pcf = pd.read_csv(post_recon_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_big_2pcf = pd.read_csv(post_recon_big_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_nopad_2pcf = pd.read_csv(post_recon_nopad_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_noran_2pcf = pd.read_csv(post_recon_noran_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])

    # Read computed 2pcfs rspace
    pre_2pcf_r = pd.read_csv(pre_recon_r_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_nopad_niter3_r_2pcf = pd.read_csv(post_recon_nopad_niter3_r_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_nopad_niter1_r_2pcf = pd.read_csv(post_recon_nopad_niter1_r_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])
    post_nopad_norsd_r_2pcf = pd.read_csv(post_recon_nopad_norsd_r_fn[3], delim_whitespace=True, engine='c', names = ['s', 'mono', 'quad', 'hexa'])

    # Plot 2pcfs
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['mono'], label='Pre z')
    ax[0].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['mono'], label='Post ran small z')
    ax[0].plot(post_nopad_2pcf['s'], post_nopad_2pcf['s']**2*post_nopad_2pcf['mono'], label='Post nopad ran small z')
    ax[0].plot(post_big_2pcf['s'], post_big_2pcf['s']**2*post_big_2pcf['mono'], label='Post ran big z')
    ax[0].plot(post_noran_2pcf['s'], post_noran_2pcf['s']**2*post_noran_2pcf['mono'], label='Post no ran z')

    ax[0].plot(pre_2pcf_r['s'], pre_2pcf_r['s']**2*pre_2pcf_r['mono'], label='Pre r')
    ax[0].plot(post_nopad_niter3_r_2pcf['s'], post_nopad_niter3_r_2pcf['s']**2*post_nopad_niter3_r_2pcf['mono'], label='Post nopad ran small r niter3')
    ax[0].plot(post_nopad_niter1_r_2pcf['s'], post_nopad_niter1_r_2pcf['s']**2*post_nopad_niter1_r_2pcf['mono'], label='Post nopad ran small r niter1')
    ax[0].plot(post_nopad_norsd_r_2pcf['s'], post_nopad_norsd_r_2pcf['s']**2*post_nopad_norsd_r_2pcf['mono'], label='Post nopad ran small r norsd')

    ax[1].plot(pre_2pcf['s'], pre_2pcf['s']**2*pre_2pcf['quad'])
    ax[1].plot(post_2pcf['s'], post_2pcf['s']**2*post_2pcf['quad'])
    ax[1].plot(post_nopad_2pcf['s'], post_nopad_2pcf['s']**2*post_nopad_2pcf['quad'])
    ax[1].plot(post_big_2pcf['s'], post_big_2pcf['s']**2*post_big_2pcf['quad'])
    ax[1].plot(post_noran_2pcf['s'], post_noran_2pcf['s']**2*post_noran_2pcf['quad'])

    ax[1].plot(pre_2pcf_r['s'], pre_2pcf_r['s']**2*pre_2pcf_r['quad'])
    ax[1].plot(post_nopad_niter3_r_2pcf['s'], post_nopad_niter3_r_2pcf['s']**2*post_nopad_niter3_r_2pcf['quad'])
    ax[1].plot(post_nopad_niter1_r_2pcf['s'], post_nopad_niter1_r_2pcf['s']**2*post_nopad_niter1_r_2pcf['quad'])
    ax[1].plot(post_nopad_norsd_r_2pcf['s'], post_nopad_norsd_r_2pcf['s']**2*post_nopad_norsd_r_2pcf['quad'])
    
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
    xi0 = np.sum(mono, axis=0)
    xi2 = np.sum(quad, axis=0)
    xi4 = np.sum(hexa, axis=0)
    return s[:,0], xi0, xi2, xi4
def main_corrfunc():
    pre_recon_fn = "tests/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon_fn = "tests/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.dat"
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
    pre_recon_fn = "tests/CATALPTCICz0.466G960S1005638091_zspace.dat"
    post_recon_fn = "tests/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.dat"
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
    #main_fcfc_lc()
    main_fcfc_ran_box()
