from numba import cuda
import numpy as np
print(cuda.gpus)


@cuda.jit
def allocate_gal_cic_kernel(delta, x, y, z, w, npart, xmin, zmin, boxsize, nbins, wrap):


    binsize = boxsize / nbins
    oneoverbinsize = 1.0 / binsize
    weight = 1.0
    
    #delta assumed oto be initialized to zero (np.zeros)

    particle_index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(particle_index, npart, stride):
        if(w is not None): weight = w[i]
        else : weight=1
    
        xpos = (x[i] - xmin) * oneoverbinsize
        ypos = (y[i] - ymin) * oneoverbinsize
        zpos = (z[i] - zmin) * oneoverbinsize

        ix = int(xpos) % nbins
        iy = int(ypos) % nbins
        iz = int(zpos) % nbins

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
    

def main():
    # Create arrays
    x = np.ones(N)
    y = np.copy(x)*2
    z = np.empty_like(x)
    # Define blocks and threads
    n_threads_per_block = int(256)
    n_blocks = int((N + n_threads_per_block -1) / n_threads_per_block)
    #print(n_threads_per_block, n_blocks, n_threads_per_block*n_blocks)
    #print(z)
    add[n_blocks, n_threads_per_block](N, x, y, z)
    cuda.synchronize()

    print(z)
    

if __name__=='__main__':
    main()