# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: language_level=3
cimport numpy as np
cimport cython
from libc.math cimport exp
from libc.stdio cimport FILE, fopen, fread, fclose
from libc.stdlib cimport malloc, free
from cython.parallel import prange

cdef struct PARTICLE:
  int nadj
  int nadj_count
  int *adj

def mult_kx(double complex[:,:,:] deltaout, double complex[:,:,:] delta,
            double[:] k, double bias):
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      for iy in range(N):
        for iz in range(N):
          deltaout[ix,iy,iz] = delta[ix,iy,iz] * (-1.0j) * k[ix] / bias
    return deltaout

def mult_ky(double complex[:,:,:] deltaout, double complex[:,:,:] delta,
            double[:] k, double bias):
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      for iy in range(N):
        for iz in range(N):
          deltaout[ix,iy,iz] = delta[ix,iy,iz] * (-1.0j) * k[iy] / bias
    return deltaout

def mult_kz(double complex[:,:,:] deltaout, double complex[:,:,:] delta,
            double[:] k, double bias):
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      for iy in range(N):
        for iz in range(N):
          deltaout[ix,iy,iz] = delta[ix,iy,iz] * (-1.0j) * k[iz] / bias
    return deltaout

def mult_norm(double complex[:,:,:] rhoout, double complex[:,:,:] rhoin,
             double[:,:,:] norm):
    cdef int N = rhoin.shape[0]
    cdef int ix,iy,iz
    for ix in range(N):
      for iy in range(N):
        for iz in range(N):
          rhoout[ix,iy,iz] = rhoin[ix,iy,iz] * norm[ix,iy,iz]
    return rhoout

def divide_k2(double complex[:,:,:] deltaout, double complex[:,:,:] delta,
              double[:] k):              
    cdef int N = delta.shape[0]
    cdef int ix,iy,iz
    cdef double kx,ky,kz,k2
    for ix in range(N):
      kx = k[ix]
      for iy in range(N):
        ky = k[iy]
        for iz in range(N):
          kz = k[iz]
          k2 = kx*kx + ky*ky + kz*kz
          if(ix + iy + iz > 0):
            deltaout[ix,iy,iz] = delta[ix,iy,iz] / k2
    deltaout[0,0,0] = 0.
    return deltaout
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def allocate_gal_cic(
    double[:, :, :] delta,
    double[:] x,
    double[:] y,
    double[:] z,
    double[:] w,
    int npart,
    double xmin,
    double ymin,
    double zmin,
    double boxsize,
    int nbins,
    int wrap):

  cdef double xpos,ypos,zpos
  cdef Py_ssize_t i,j,k
  cdef int ix,iy,iz
  cdef int ixp,iyp,izp
  cdef double ddx,ddy,ddz
  cdef double mdx,mdy,mdz
  cdef double weight
  cdef double binsize
  cdef double oneoverbinsize

  binsize = boxsize / nbins
  oneoverbinsize = 1.0 / binsize
  weight = 1.0

  for i in prange(nbins, nogil=True):
    for j in range(nbins):
      for k in range(nbins):
        delta[i,j,k] = 0.

  for i in prange(npart, nogil=True):
    if(w is not None):
      weight = w[i]
    else:
      weight = 1
    
   
    xpos = (x[i] - xmin)*oneoverbinsize
    ypos = (y[i] - ymin)*oneoverbinsize
    zpos = (z[i] - zmin)*oneoverbinsize

    ix = <int>xpos % nbins
    iy = <int>ypos % nbins
    iz = <int>zpos % nbins
    

    ddx = xpos-ix
    ddy = ypos-iy
    ddz = zpos-iz

    mdx = (1.0 - ddx)
    mdy = (1.0 - ddy)
    mdz = (1.0 - ddz)

    ixp = ix + 1;
    iyp = iy + 1;
    izp = iz + 1;

    if(wrap):
      if(ixp >= nbins): ixp -= nbins
      if(iyp >= nbins): iyp -= nbins
      if(izp >= nbins): izp -= nbins
      #if(ix >= nbins): ix -= nbins
      #if(iy >= nbins): iy -= nbins
      #if(iz >= nbins): iz -= nbins
    else:
      if(ixp >= nbins):
        ixp = 0
        mdx = 0.0
      if(iyp >= nbins):
        iyp = 0
        mdy = 0.0
      if(izp >= nbins):
        izp = 0
        mdz = 0.0


    
    delta[ix,  iy,  iz]  += mdx * mdy * mdz * weight
    delta[ixp, iy,  iz]  += ddx * mdy * mdz * weight
    delta[ix,  iyp, iz]  += mdx * ddy * mdz * weight
    delta[ix,  iy,  izp] += mdx * mdy * ddz * weight
    delta[ixp, iyp, iz]  += ddx * ddy * mdz * weight
    delta[ixp, iy,  izp] += ddx * mdy * ddz * weight
    delta[ix,  iyp, izp] += mdx * ddy * ddz * weight
    delta[ixp, iyp, izp] += ddx * ddy * ddz * weight

  return delta

@cython.boundscheck(False)
@cython.wraparound(False)
def get_shift_array(double[:] x_arr, double[:] y_arr, double[:] z_arr, double[:,:,:] f_x, double[:,:,:] f_y,
                    double[:,:,:] f_z, double xmin, double ymin, double zmin, double binsize, bint is_box, int nbins,
                    double[:] shift_x, double[:] shift_y, double[:] shift_z, int n_threads):
        """Given grid of f_x, f_y and f_z values, uses interpolation scheme to compute
        appropriate values at the galaxy positions"""

        cdef int i, j, k, ii, jj, kk, posx, posy, posz
        cdef float xpos, ypos, zpos, ddx, ddy, ddz, weight
        cdef Py_ssize_t n
        cdef size_t npart = x_arr.shape[0]

        for n in prange(npart, nogil=True, num_threads=n_threads):

          xpos = (x_arr[n] - xmin) / binsize
          ypos = (y_arr[n] - ymin) / binsize
          zpos = (z_arr[n] - zmin) / binsize
        

          i = <int> xpos
          j = <int> ypos
          k = <int> zpos

          ddx = xpos - i
          ddy = ypos - j
          ddz = zpos - k

        
          shift_x[n] = 0
          shift_y[n] = 0
          shift_z[n] = 0
          for ii in range(2):
              for jj in range(2):
                  for kk in range(2):
                      weight = (((1 - ddx) + ii * (-1 + 2 * ddx)) *
                                ((1 - ddy) + jj * (-1 + 2 * ddy)) *
                                ((1 - ddz) + kk * (-1 + 2 * ddz)))
                      if is_box:
                          posx = (i + ii) % nbins
                          posy = (j + jj) % nbins
                          posz = (k + kk) % nbins
                      else:
                          posx = i + ii
                          posy = j + jj
                          posz = k + kk
                      shift_x[n] = shift_x[n] + f_x[posx, posy, posz] * weight
                      shift_y[n] = shift_y[n] + f_y[posx, posy, posz] * weight
                      shift_z[n] = shift_z[n] + f_z[posx, posy, posz] * weight

        

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_shift_array(double[:] x_arr, double[:] y_arr, double[:] z_arr, double[:,:,:] f_x, double[:,:,:] f_y,
                    double[:,:,:] f_z, double xmin, double ymin, double zmin, double binsize, bint is_box, int nbins,
                    float box_length, double[:] new_x_arr, double[:] new_y_arr, double[:] new_z_arr, int n_threads):
        """Given grid of f_x, f_y and f_z values, uses interpolation scheme to compute
        appropriate values at the galaxy positions"""

        cdef int i, j, k, ii, jj, kk, posx, posy, posz
        cdef float xpos, ypos, zpos, ddx, ddy, ddz, weight, shift_x, shift_y, shift_z
        cdef Py_ssize_t n
        cdef size_t npart = x_arr.shape[0]


        for n in prange(npart, nogil=True, num_threads=n_threads):

          xpos = (x_arr[n] - xmin) / binsize
          ypos = (y_arr[n] - ymin) / binsize
          zpos = (z_arr[n] - zmin) / binsize
        

          i = <int> xpos
          j = <int> ypos
          k = <int> zpos

          ddx = xpos - i
          ddy = ypos - j
          ddz = zpos - k

        
          shift_x = 0
          shift_y = 0
          shift_z = 0
          for ii in range(2):
              for jj in range(2):
                  for kk in range(2):
                      weight = (((1 - ddx) + ii * (-1 + 2 * ddx)) *
                                ((1 - ddy) + jj * (-1 + 2 * ddy)) *
                                ((1 - ddz) + kk * (-1 + 2 * ddz)))
                      if is_box:
                          posx = (i + ii) % nbins
                          posy = (j + jj) % nbins
                          posz = (k + kk) % nbins
                      else:
                          posx = i + ii
                          posy = j + jj
                          posz = k + kk
                      shift_x = shift_x + f_x[posx, posy, posz] * weight
                      shift_y = shift_y + f_y[posx, posy, posz] * weight
                      shift_z = shift_z + f_z[posx, posy, posz] * weight


          new_x_arr[n] = x_arr[n] + shift_x
          new_y_arr[n] = y_arr[n] + shift_y
          new_z_arr[n] = z_arr[n] + shift_z

          if is_box:
            new_x_arr[n] = (new_x_arr[n] + box_length) % box_length
            new_y_arr[n] = (new_y_arr[n] + box_length) % box_length
            new_z_arr[n] = (new_z_arr[n] + box_length) % box_length


def normalize_delta_survey(double complex[:,:,:] delta, double[:,:,:] rhog,
                        double[:,:,:] rhor, double alpha, double ran_min):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        if rhor[ix, iy, iz] > ran_min:
          delta[ix, iy, iz] = (rhog[ix, iy, iz] / (alpha * rhor[ix, iy, iz])) - 1. + 0.0j
        else:
          delta[ix, iy, iz] = 0. + 0.0j

  return delta

def normalize_delta_box(double complex[:,:,:] delta, double[:,:,:] rhog,
                        int npart):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        delta[ix, iy, iz] = (rhog[ix, iy, iz] * N**3) / npart - 1.0 + 0.0j

  return delta

def normalize_rho_survey(double[:,:,:] rho_out, double[:,:,:] rhog,
                        double[:,:,:] rhor, double alpha, double ran_min):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        if rhor[ix, iy, iz] > ran_min:
          rho_out[ix, iy, iz] = (rhog[ix, iy, iz] / (alpha * rhor[ix, iy, iz]))
        else:
          rho_out[ix, iy, iz] = 0.9e30

  return rho_out

def normalize_rho_box(double[:,:,:] rhog, int npart):

  cdef int N = rhog.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        rhog[ix, iy, iz] = (rhog[ix, iy, iz] * N**3) / npart

  return rhog

def survey_mask(int[:] mask, double[:,:,:] rhor, double ran_min):

  cdef int N = rhor.shape[0]
  cdef int ix, iy, iz
  for ix in range(N):
    for iy in range(N):
      for iz in range(N):
        if rhor[ix, iy, iz] <= ran_min:
          mask[ix*N*N + iy*N + iz] = 1

  return mask

def survey_cuts_logical(int[:] out, double[:] veto,
                        double[:] redshift, double zmin, double zmax):

  cdef int N = redshift.shape[0]
  cdef int i
  for i in range(N):
    if (veto[i] == 1.) and (redshift[i] > zmin) and (redshift[i] < zmax):
      out[i] = 1
    else:
      out[i] = 0

  return out

def voxelvoid_cuts(int[:] select, int[:] mask,
                   double[:,:] rawvoids, double min_dens_cut):

  cdef int N = rawvoids.shape[0]
  cdef int i, vox
  for i in range(N):
    vox = int(rawvoids[i, 2])
    if (mask[vox] == 0) and (rawvoids[i, 1] == 0) and (rawvoids[i, 3] < min_dens_cut):
    # if (mask[vox] == 0) and (rawvoids[i, 3] < min_dens_cut):
      select[i] = 1
    else:
      select[i] = 0

  return select

def voxelcluster_cuts(int[:] select, int[:] mask,
                   double[:,:] rawclusters, double max_dens_cut):

  cdef int N = rawclusters.shape[0]
  cdef int i, vox
  for i in range(N):
    vox = int(rawclusters[i, 2])
    if (mask[vox] == 0) and (rawclusters[i, 1] == 0) and (rawclusters[i, 3] > max_dens_cut):
      select[i] = 1
    else:
      select[i] = 0

  return select

def get_member_densities(double[:] member_dens, int[:] voxels,
                         double[:] rho):

  cdef int N = len(voxels)
  cdef int i
  for i in range(N):
    member_dens[i] = rho[voxels[i]]

  return member_dens
