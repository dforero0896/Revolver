# This file explains the input parameters for the code and assigns reasonable default values to each of them.
# Don't alter this file – instead use a separate parameters file to overwrite any input values you wish to change
# Note: distance units in this code are calculated in Mpc/h by default

# ======= runtime options ======== #
verbose = True  # True for more informative output statements
debug = True    # True for output checks during reconstruction
nthreads = 8     # set to the number of CPUs available, more is better
# ================================ #

# ========= file handling options ========= #
handle = 'default'  # string to identify the run; used to set filenames
output_folder = 'revolver_test/'   # /path/to/folder/ where output should be placed
# ========================================= #

# ========== cosmology ============ #
omega_m = 0.307115  # used for reconstruction and to convert redshifts to distances (assumes flat Universe!)
# ================================= #

# ======= reconstruction options ========== #
do_recon = True     # if False, no reconstruction is performed and other recon options are ignored
nbins = 512     # the number of grid cells per side of the box
padding = 200.  # for survey data, the extra 'padding' for the cubic box, in Mpc/h
smooth = 10.    # smoothing scale in Mpc/h
bias = 1.92        # the linear galaxy/tracer bias value
f = 0.743         # the linear growth rate at the mean redshift
niter = 3       # number of iterations in the FFT reconstruction method, 3 is sufficient
# NOTE: for box data, reconstruction assumes plane-parallel approximation with single l-o-s along the box z-axis!!
# ========================================= #

# ======= input galaxy/tracer data options =========== #
tracer_file = 'revolver_test/CATALPTCICz0.466G960S1005638091_zspace.dat'     # /path/to/file with input data
tracer_file_type = 3  # 1 for FITS file, 2 for array in numpy pickle format (.npy), 3 for array in ASCII format
# NOTE: for FITS files, the tracer coordinates should be specified using appropriate field names
# current options are 'RA', 'DEC' and 'Z' for survey-like data on the sky, or 'X', 'Y', 'Z' for simulation boxes
# For array data (tracer_file_type = 2 or 3), specify which columns of the array contain the tracer coordinates
tracer_posn_cols = [0, 1, 2]  # columns of tracer input array containing 3D position information
# specify data type:
is_box = True       # True for cubic simulation box with periodic boundaries; False for survey-like data on the sky
box_length = 2500.   # if is_box, the box side length in Mpc/h; else ignored
# the following cuts useful for more efficient reconstruction and voxel void-finding for BOSS CMASS data, where a tiny
# fraction of data extends to very high or very low redshifts (and even redshifts < 0)
z_low_cut = 0.4      # lower redshift cut (ignored if not survey)
z_high_cut = 0.73    # higher redshift cut (ignored if not survey)
# what is the model for applying weights? 1 = like BOSS; 2 = like eBOSS; 3 = like joint BOSS+eBOSS LRG sample
# (unfortunately things change as surveys progress)
weights_model = 2
# 1. For FITS files (tracer_file_type = 1) weights are automatically extracted using field names based on BOSS/eBOSS data
# model (https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/galaxy_DRX_SAMPLE_NS.html)
# 2. for simulation box data (is_box = True) all weights information is ignored as assumed uniform
# -----------
# Most users will only use weights with survey data in FITS files
# If for some reason you have survey(-like) data in array format (tracer_file_type = 2 or 3), specify what info is
# present in the file using following flags (FKP, close-pair, missing redshift, total systematics, veto flag,
# completeness). Weights MUST be given in consecutive columns starting immediately after the column with redshifts,
# and with column numbers in the order fkp<cp<noz<systot<veto<comp
fkp = False     # FKP weights (used for reconstruction when n(z) is not constant)
cp = False      # close-pair or fibre collision weights
noz = False     # missing redshift / redshift failure weights
systot = False  # total systematic weights
veto = False    # veto mask (if present, galaxies with veto!=1 are discarded)
comp = False    # sector completeness
# ============================================= #

# ====== input randoms options ======= #
# for survey-like data, randoms characterize the window function and MUST be provided for reconstruction and
# voxel void-finding (not necessary for ZOBOV alone)
random_file = ''   # /path/to/file containing randoms data
random_file_type = 1  # 1 for FITS file, 2 for array in numpy pickle format (.npy), 3 for array in ASCII format
# if random_file_type = 2 or 3, specify which columns of the array contain the (RA, Dec, redshift) coordinates
random_posn_cols = [0, 1, 2]
# if galaxy data has FKP weights, randoms are assumed to have FKP weights too
# all other galaxy weights are ignored for randoms
# =========================== #
