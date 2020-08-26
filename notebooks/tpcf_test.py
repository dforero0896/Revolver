#!/usr/bin/env python

import numpy as np
from astroML.correlation import two_point, bootstrap_two_point
def two_point_box(data, bins, box_size):
    """ From astroML two_point"""
    print("Building data KDTree", flush=True)
    KDT_D = KDTree(data)
    print("Counting DD.", flush=True)
    counts_DD = KDT_D.two_point_correlation(data, bins)
    DD = np.diff(counts_DD)
    bins_low = bins[:-1]
    bins_high = bins[1:]
    volume = 4 * np.pi * (bin_high**3 - bin_low**3) / 3
    RR = volume/(box_size**3)

    corr = DD/RR -1

    return corr, DD, RR

if __name__ == '__main__':
    print("Importing prerecon dataset", flush=True)
    pre_recon = np.loadtxt("revolver_test/CATALPTCICz0.466G960S1005638091_zspace.dat")
    print(f"Prerecon shape: {pre_recon.shape}")
    print("Importing postrecon dataset", flush=True)
    post_recon = np.load("revolver_test/CATALPTCICz0.466G960S1005638091_zspace_pos_shift.npy")
    print(f"Postrecon shape: {post_recon.shape}")

    bins = np.linspace(0, 200, 41)
    print(f"Computing pre_recon correlation function.")
    corr_pre, DD_pre, RR_pre = two_point_box(pre_recon[:,:-1], bins=bins, box_size=2500)
    print(f"Computing post_recon correlation function.")
    corr_post, DD_post, RR_post = two_point_box(post_recon, bins=bins, box_size=2500)

    print(corr_pre.shape, corr_post.shape)