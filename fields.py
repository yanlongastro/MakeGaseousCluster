import numpy as np

def get_rms_profile(Bx, By, Bz, shape):
    X = np.linspace(0, 1, shape[0])
    Y = np.linspace(0, 1, shape[1])
    Z = np.linspace(0, 1, shape[2])

    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')


    r = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)


    r_bins = np.linspace(0, r.max(), int(np.mean(shape)/2))
    B_rms_profile = []
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        B_rms_profile.append(np.sqrt(np.mean(B_mag[mask]**2)))
    return r_bins[:-1], B_rms_profile


def normalize_field(Bx, By, Bz, shape, envelope_func):
    r_bins, B_rms_profile = get_rms_profile(Bx, By, Bz, shape)
    B_rms_target = envelope_func(r_bins)

    fac = np.exp(np.mean(np.log(B_rms_target)-np.log(B_rms_profile)))
    Bx *= fac
    By *= fac
    Bz *= fac
    return Bx, By, Bz


def generate_periodic_B_with_envelope(shape, power_spectrum_func, envelope_func, seed=42):

    nx, ny, nz = shape
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    print('Shape:', shape)
    
    # 1. Initialize FFT
    kx = 2*np.pi * np.fft.fftfreq(nx, Lx/nx)
    ky = 2*np.pi * np.fft.fftfreq(ny, Ly/ny)
    kz = 2*np.pi * np.fft.fftfreq(nz, Lz/nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = 1
    
    np.random.seed(seed)
    
    # 2. Gaussian random amp. for A
    A_hat_x = (np.random.normal(0, 1, shape) + 1j*np.random.normal(0, 1, shape))
    A_hat_y = (np.random.normal(0, 1, shape) + 1j*np.random.normal(0, 1, shape))
    A_hat_z = (np.random.normal(0, 1, shape) + 1j*np.random.normal(0, 1, shape))
    
    # 3. Apply power sepctrum
    power = power_spectrum_func(np.sqrt(K2)) / np.sqrt(K2)  # note the additional k
    A_hat_x *= np.sqrt(power)
    A_hat_y *= np.sqrt(power)
    A_hat_z *= np.sqrt(power)
    
    # 4. A in the real space
    A_x = np.fft.ifftn(A_hat_x).real
    A_y = np.fft.ifftn(A_hat_y).real
    A_z = np.fft.ifftn(A_hat_z).real
    
    # 5. Apply envelope
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    z = np.linspace(-Lz/2, Lz/2, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    
    A_envelope = envelope_func(R)
    A_x *= A_envelope
    A_y *= A_envelope
    A_z *= A_envelope
    
    # 6. Calculate B=curl A in the Fourier space
    A_hat_x = np.fft.fftn(A_x)
    A_hat_y = np.fft.fftn(A_y)
    A_hat_z = np.fft.fftn(A_z)
    
    # curl in the Fourier space
    B_hat_x = 1j * (KY * A_hat_z - KZ * A_hat_y)
    B_hat_y = 1j * (KZ * A_hat_x - KX * A_hat_z)
    B_hat_z = 1j * (KX * A_hat_y - KY * A_hat_x)
    
    # IFFT
    B_x = np.fft.ifftn(B_hat_x).real
    B_y = np.fft.ifftn(B_hat_y).real
    B_z = np.fft.ifftn(B_hat_z).real
    
    print("Applying normalization.")
    B_x, B_y, B_z = normalize_field(B_x, B_y, B_z, shape, envelope_func)
    
    print("Done!")
    return B_x, B_y, B_z



def normalize_velocity(vx, vy, vz, shape, envelope_func):
    X = np.linspace(0, 1, shape[0])
    Y = np.linspace(0, 1, shape[1])
    Z = np.linspace(0, 1, shape[2])

    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

    r = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)


    r_bins = np.linspace(0, r.max(), int(np.max(shape)/2))
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        v_rms = np.sqrt(np.mean(v_mag[mask]**2))
        v_rms_target = envelope_func((r_bins[i]+r_bins[i+1])/2)
        
        vx[mask] *= v_rms_target/v_rms
        vy[mask] *= v_rms_target/v_rms
        vz[mask] *= v_rms_target/v_rms
    return vx, vy, vz



def generate_random_v_with_envelope(shape, power_spectrum_func, envelope_func, sol_ratio=0.25, seed=43):
    nx, ny, nz = shape
    kx = 2*np.pi * np.fft.fftfreq(nx, 1/nx)
    ky = 2*np.pi * np.fft.fftfreq(ny, 1/ny)
    kz = 2*np.pi * np.fft.fftfreq(nz, 1/nz)
    
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1

    total_power = power_spectrum_func(np.sqrt(k2))
    
    # 1. A for the div-free part
    A_hat_x = (np.random.normal(0, 1, shape) + 1j*np.random.normal(0, 1, shape))
    A_hat_y = (np.random.normal(0, 1, shape) + 1j*np.random.normal(0, 1, shape))
    A_hat_z = (np.random.normal(0, 1, shape) + 1j*np.random.normal(0, 1, shape))
    
    A_hat = np.array([A_hat_x, A_hat_y, A_hat_z]) * np.sqrt(total_power /np.sqrt(k2) /2 * sol_ratio)
    
    v_sol_hat = np.array([
        1j * (ky * A_hat[2] - kz * A_hat[1]),
        1j * (kz * A_hat[0] - kx * A_hat[2]),
        1j * (kx * A_hat[1] - ky * A_hat[0])
    ])
    
    # 2. phi for the curl-free part
    phi_hat = (np.random.normal(0, 1, shape) + 1j*np.random.normal(0, 1, shape))
    phi_hat = phi_hat * np.sqrt(total_power / np.sqrt(k2) * (1-sol_ratio))
    
    v_comp_hat = np.array([
        1j * kx * phi_hat,
        1j * ky * phi_hat,
        1j * kz * phi_hat
    ])
    
    # 3. merge
    v_hat = v_sol_hat + v_comp_hat
    
    # 
    vx, vy, vz = np.array([np.fft.ifftn(v_hat[i]).real for i in range(3)])
    
    vx, vy, vz = normalize_velocity(vx, vy, vz, shape, envelope_func)
    
    return vx, vy, vz


from scipy.interpolate import RegularGridInterpolator

def interpolate_vector_field_to_particles(vx, vy, vz, particles):
    """
    Interpolate the vector field (vx, vy, vz) defined on the grid (x, y, z) to the particle positions.
    
    Parameters:
    - vx, vy, vz: 3D arrays
    - x, y, z: 1D arrays
    - particles: (n_particles, 3) array
    
    Returns:
    - vx_p, vy_p, vz_p: (n_particles,) arrays
    """

    shape = vx.shape
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    z = np.linspace(0, 1, shape[2])

    interp_vx = RegularGridInterpolator((x, y, z), vx, bounds_error=False, fill_value=0.0)
    interp_vy = RegularGridInterpolator((x, y, z), vy, bounds_error=False, fill_value=0.0)
    interp_vz = RegularGridInterpolator((x, y, z), vz, bounds_error=False, fill_value=0.0)
    
    vx_p = interp_vx(particles)
    vy_p = interp_vy(particles)
    vz_p = interp_vz(particles)
    
    return np.array([vx_p, vy_p, vz_p]).T