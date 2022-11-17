import numpy as np
import pdb

def prop2d_fp(u_in, L_xy, lmb, z):
    """
    Performs wave propagation using Fresnel Propagation Method
    Inputs:
    u_in (np.ndarray): Complex wavefront profile at starting (input) plane
    L_xy (list): Two-element list stating length sizes of wavefront extent. In microns.
    lmb (double): Wavelength of light/wave. In microns. 
    z (double): length/distance to do free-space propagation. In microns.
    Outputs:
    u_out (np.ndarray): Complex wavefront profile at output plane
    """

    Ny, Nx = u_in.shape
    Lx, Ly = L_xy

    fs_x = Nx/Lx
    fs_y = Ny/Ly
    fx = fs_x * np.arange((-1*Nx+1)/2, (1*Nx)/2) / Nx
    fy = fs_y * np.arange((-1*Ny+1)/2, (1*Ny)/2) / Ny

    [Fx, Fy] = np.meshgrid(fx,fy)
    bp = (np.sqrt(Fx**2 + Fy**2)<(1/lmb))  # mediums bandpass
    # create 'propagation filter' in Fourier domain
    H = bp * np.exp(1j*2*np.pi*(z/lmb)*bp*(1-0.5*(lmb**2)*(Fx**2 + Fy**2)))

    A0 = np.fft.fft2(u_in)
    A0 = np.fft.fftshift(A0)
    Az = np.fft.ifftshift(A0*H)

    u_out = np.fft.ifft2(Az)
    return u_out

def prop2d_as(u_in, L_xy, lmb, z):
    """
    Performs wave propagation using Angular Spectrum Method
    Inputs:
    u_in (np.ndarray): Complex wavefront profile at starting (input) plane
    L_xy (list): Two-element list stating length sizes of wavefront extent. In microns.
    lmb (double): Wavelength of light/wave. In microns. 
    z (double): length/distance to do free-space propagation. In microns.
    Outputs:
    u_out (np.ndarray): Complex wavefront profile at output plane
    """

    Ny, Nx = u_in.shape
    Lx, Ly = L_xy

    fs_x = Nx/Lx
    fs_y = Ny/Ly
    fx = fs_x * np.arange((-1*Nx+1)/2, (1*Nx)/2) / Nx
    fy = fs_y * np.arange((-1*Ny+1)/2, (1*Ny)/2) / Ny

    [Fx, Fy] = np.meshgrid(fx,fy)
    bp = (np.sqrt(Fx**2 + Fy**2)<(1/lmb))  # mediums bandpass
    # create 'propagation filter' in Fourier domain
    map = 1 - (lmb**2)*(Fx**2 + Fy**2)
    map[np.where(map<0)] = 0
    map = np.sqrt(map)
    # for i in range(bp.shape[0]):
    #     for j in range(bp.shape[1]):
    #         if bp[i,j]:
    #             map[i,j] = np.sqrt(1 - (lmb**2)*(Fx[i,j]**2 + Fy[i,j]**2))
    #         else: 
    #             map[i,j] = 0
    H = bp * np.exp(1j*2*np.pi*(z/lmb)*bp*map)

    A0 = np.fft.fft2(u_in)
    A0 = np.fft.fftshift(A0)
    Az = np.fft.ifftshift(A0*H)

    u_out = np.fft.ifft2(Az)
    return u_out    
