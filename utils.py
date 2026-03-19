import numpy as np
from scipy.integrate import simpson, cumulative_trapezoid
from galpy.df import kingdf


def uniform_sampling_on_spherical(npoints, ndim=3, seed=42):
    """
    Sample on a sphere uniformly, using the normal dist.
    """
    np.random.seed(seed)
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T

def measure_spherical_density_profile(mass, radius, dN=100, cdf=False, sigma=False):
    i = 0
    if radius.ndim>1:
        radius = np.linalg.norm(radius, axis=-1)
    sort = np.argsort(radius)
    radius = radius[sort]
    mass = mass[sort]
    if cdf:
        return radius, np.cumsum(mass)

    r = []
    rho = []
    while i+dN<len(mass):
        if sigma:
            dM = np.std(mass[i:i+dN])
            r1 = radius[i+dN] 
            r0 = radius[i]
            dV = 1
        else:
            dM = np.sum(mass[i:i+dN])
            r1 = radius[i+dN] 
            r0 = radius[i]
            dV = (r1**3-r0**3)*np.pi*4/3
        rho_tmp = dM/dV
        r_tmp = (r0+r1)/2
        r.append(r_tmp)
        rho.append(rho_tmp)
        i += dN
    return np.array(r), np.array(rho)


def plummer_density_profile(r, M, a):
    return (3 * M) / (4 * np.pi * a**3) * (1 + (r/a)**2)**(-5/2)


def sigma_r_from_Jeans(x, density_profile, Menc_profile, rmin, rmax, G=1):
    x = min(max(x, rmin), rmax*0.99)
    r = np.linspace(x, rmax, 1000)
    rho_r = density_profile(r)
    Menc_r = Menc_profile(r)
    # print(x, Menc_r[0], rho_r[0])
    integral = simpson(y=rho_r*G*Menc_r/r**2, x=r)
    rho_x = density_profile(x)
    # print(x, rmin, rmax,  rho_x, integral)
    return np.sqrt(integral / rho_x)



def virial_radius_numerical(r, rho_r, M_enclosed):
    integral = simpson(y=rho_r * r * 4 * np.pi * M_enclosed,x=r)
    return 1 / 2 / integral

def construct_galpy_kingdf(M, virial_radius, w0):
    dfk= kingdf(W0=w0, M=1, rt=1) # unit mass and unit length

    r = np.linspace(dfk.r0/100., dfk.rt, 1000)
    rho_r = dfk.dens(r)
    M_enclosed = cumulative_trapezoid(4*np.pi*r**2*rho_r, r, initial=0)
    rv = virial_radius_numerical(r, rho_r, M_enclosed)

    radius_scale_factor = rv/virial_radius

    dfk= kingdf(W0=w0, M=M, rt=1/radius_scale_factor) # real mass and scale
    return dfk



def refine_in_xyz(x):
    x_ = np.copy(x)
    x_ /= 2
    x = np.concatenate(
            [
                x_
                + i * np.array([0.5, 0, 0])
                + j * np.array([0, 0.5, 0])
                + k * np.array([0, 0, 0.5])
                for i in range(2)
                for j in range(2)
                for k in range(2)
            ]
        )
    return x


def load_glass_coords(N, method='meshoid', glass_path='glass_256.npy'):
    """
    This must be 3d
    returns xy in [-1, 1], z in [-1, 1]
    """
    if method == 'meshoid':
        import meshoid
        x = meshoid.glass.particle_glass(N=N, )
    else:
        x = np.load(glass_path)
    while len(x)<N*3:
        print("Doing refinement")
        x = refine_in_xyz(x)
        
    print("Loaded %d (~%.0f^3) particles"%(len(x), len(x)**(1/3)))
    return x