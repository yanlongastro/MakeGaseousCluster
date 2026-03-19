from cosmic.sample import InitialCMCTable
import numpy as np
from scipy.integrate import simpson, cumulative_trapezoid
import yaml
from scipy import interpolate
import pytreegrav
import h5py

import utils
import fields
import importlib
importlib.reload(utils)
importlib.reload(fields)

import matplotlib.pyplot as plt

class MakeGaseousCluster:
    def __init__(self, param_file='params.yaml', debug=False):
        self.read_params(param_file)
        Singles = self.make_cmc_ic()
        self.eval_profile_interpolators(Singles, debug=debug)
        self.set_star_particles(Singles, debug=debug)
        if self.params['gas']['add_gas']:
            self.set_gas_particles(debug=debug)
            self.set_velocity_field(debug=debug)
            self.set_magnetic_field(debug=debug)
        self.add_bhs(debug=debug)
        self.check_bc(debug=debug)
        self.write_snapshot()

    def read_params(self, param_file):
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        params['cmc']['size'] = params['cluster']['nstar']
        self.params = params

        self.virial_radius = params['cluster']['virial_radius']
        self.w_0 = params['cmc']['w_0']
        self.M_gas = params['gas']['M_gas']
        self.a_gas = params['gas']['a_gas']
        if not self.params['gas']['add_gas']:
            assert self.M_gas==0

        # unit: Msun, pc, km/s
        Munit_in_Msun = params['snapshot']['Munit_in_Msun']
        lunit_in_au = 206265*params['snapshot']['lunit_in_pc']
        vunit_in_kms = params['snapshot']['vunit_in_kms']
        tunit_in_yr = lunit_in_au/vunit_in_kms*1.496e8/(86400*365)
        self.G = 4*np.pi**2 *(1/lunit_in_au)**3/(1/tunit_in_yr)**2/(1/Munit_in_Msun)
        self.Munit_in_Msun = Munit_in_Msun
        self.lunit_in_pc = params['snapshot']['lunit_in_pc']
        self.vunit_in_kms = vunit_in_kms
        self.Bunit_in_G = params['snapshot']['Bunit_in_G']
        print('G =', self.G)

    def make_cmc_ic(self):
        print("Making CMC IC")
        Singles, Binaries = InitialCMCTable.sampler('cmc', **(self.params['cmc']))
        Singles['r'] *= self.virial_radius
        assert np.max(Singles['r'])<self.params['box']['box_size']/2
        self.M_star = np.sum(Singles['m'])
        return Singles
    
    def eval_profile_interpolators(self, Singles, debug=False):
        print("Evaluating profile interpolators")
        M_star = self.M_star
        virial_radius = self.virial_radius
        w_0 = self.w_0
        M_gas = self.M_gas
        a_gas = self.a_gas

        dfk = utils.construct_galpy_kingdf(M_star, virial_radius, w_0)

        
        rmax = self.params['box']['box_size']*3
        rmin = self.params['box']['box_size']/1e4
        r =  np.linspace(0, rmax, 10000)
        
        Mencs = []
        for i in range(3):
            rho_cluster = dfk.dens(r) 
            rho_cluster[np.isnan(rho_cluster)] = np.nanmin(rho_cluster)
            rho_gas = utils.plummer_density_profile(r, M_gas, a_gas)

            if i==0:
                rho_r = rho_cluster
            if i==1:
                rho_r = rho_gas
            if i==2:
                rho_r = rho_cluster+rho_gas

            M_enclosed = cumulative_trapezoid(4*np.pi*r**2*rho_r, r, initial=0)
            Mencs.append(M_enclosed)

        self.Menc_func_star_only = interpolate.InterpolatedUnivariateSpline(r, Mencs[0])
        if self.params['gas']['add_gas']:
            self.Menc_func_gas_only = interpolate.InterpolatedUnivariateSpline(r, Mencs[1])
            self.r_M_func_gas_only = interpolate.InterpolatedUnivariateSpline(Mencs[1]/np.max(Mencs[1]), r)
            self.Menc_func_star_gas = interpolate.InterpolatedUnivariateSpline(r, Mencs[2])

        self.rho_func_star_only = interpolate.InterpolatedUnivariateSpline(r, rho_cluster)
        if self.params['gas']['add_gas']:
            self.rho_func_gas_only = interpolate.InterpolatedUnivariateSpline(r, rho_gas)
            self.rho_func_star_gas = interpolate.InterpolatedUnivariateSpline(r, rho_cluster+rho_gas)

        # note that below we redefine r, so it only covers the box but not rmax
        r =  np.linspace(0, self.params['box']['box_size'], 10000)

        tunit_in_yr = 206265*1.496e8/(86400*365)
        G = 4*np.pi**2 *(1/206265)**3/(1/tunit_in_yr)**2
        sigmas_star_only = np.array([utils.sigma_r_from_Jeans(x, self.rho_func_star_only, self.Menc_func_star_only, rmin, rmax, G=G) for x in r])
        sigmas_star_only[np.isnan(sigmas_star_only)] = np.nanmin(sigmas_star_only)
        if self.params['gas']['add_gas']:
            sigmas_star_composite = np.array([utils.sigma_r_from_Jeans(x, self.rho_func_star_only, self.Menc_func_star_gas, rmin, rmax, G=G) for x in r])
            sigmas_star_composite[np.isnan(sigmas_star_composite)] = np.nanmin(sigmas_star_composite)
            sigmas_gas_only = np.array([utils.sigma_r_from_Jeans(x, self.rho_func_gas_only, self.Menc_func_gas_only, rmin, rmax, G=G) for x in r])
            sigmas_gas_composite = np.array([utils.sigma_r_from_Jeans(x, self.rho_func_gas_only, self.Menc_func_star_gas, rmin, rmax, G=G) for x in r])

        self.star_only_sigma_r = interpolate.InterpolatedUnivariateSpline(r, sigmas_star_only)
        if self.params['gas']['add_gas']:
            self.star_vel_rescaler = interpolate.InterpolatedUnivariateSpline(r, sigmas_star_composite/sigmas_star_only)
            self.star_composite_sigma_r = interpolate.InterpolatedUnivariateSpline(r, sigmas_star_composite)
            self.gas_sigma_r_envelope = interpolate.InterpolatedUnivariateSpline(r, sigmas_gas_composite)
            self.gas_vel_envelope_normalized = interpolate.InterpolatedUnivariateSpline(r/self.params['box']['box_size'], np.sqrt(3)*sigmas_gas_composite)
            self.gas_B_envelope_normalized = interpolate.InterpolatedUnivariateSpline(r/self.params['box']['box_size'], 
                                                                                    self.params['gas']['B_at_center']*1e-6/self.Bunit_in_G* (rho_gas/rho_gas[0])**0.5 )
            self.gas_T_envelope = interpolate.InterpolatedUnivariateSpline(r, 
                                (rho_gas/rho_gas[0])*(self.params['gas']['T_at_center']-self.params['gas']['T_at_edge'])+self.params['gas']['T_at_edge'])

        

        if debug:
            plt.plot(r, sigmas_star_only/np.sqrt(G*M_star/virial_radius), 'k--', label='stars only')
            if self.params['gas']['add_gas']:
                plt.plot(r, sigmas_star_composite/np.sqrt(G*M_star/virial_radius), 'k:', label='star (in composite)')
                plt.plot(r, sigmas_gas_only/np.sqrt(G*M_star/virial_radius), 'r--', label='gas only')
                plt.plot(r, sigmas_gas_composite/np.sqrt(G*M_star/virial_radius), 'r:', label='gas (in composite)')

            r_, sigma_ = utils.measure_spherical_density_profile(Singles['vr'], Singles['r'], sigma=True, dN=1000)
            plt.plot(r_, sigma_, label='raw IC')
            plt.loglog()
            plt.xlabel('Radius [pc]')
            plt.ylabel('$\sigma_r$ [Henon unit]')
            plt.legend()
            plt.show()

    def set_star_particles(self, Singles, debug=False):
        print("Setting stars")
        mass = np.array(Singles['m'])/self.Munit_in_Msun
        radius = np.array(Singles['r'])/self.lunit_in_pc
        vt = np.array(Singles['vt'])
        vr = np.array(Singles['vr'])
        v = np.sqrt(vt**2+vr**2)
        nstar = len(mass)
        ids = np.array(Singles['id'], dtype=int)

        pos = utils.uniform_sampling_on_spherical(nstar, seed=42)*radius[:,np.newaxis]
        vel = utils.uniform_sampling_on_spherical(nstar, seed=45)*v[:,np.newaxis]
        vel -= np.mean(vel, axis=0)

        PE = 0.5*np.sum(mass*pytreegrav.Potential(pos, mass, G=self.G))
        KE = 0.5*np.sum(mass*np.linalg.norm(vel, axis=-1)**2)
        vel_fac = np.sqrt(-PE/(KE*2))
        vel *= vel_fac
        
        PE = 0.5*np.sum(mass*pytreegrav.Potential(pos, mass, G=self.G))
        KE = 0.5*np.sum(mass*np.linalg.norm(vel, axis=-1)**2)
        print(" (2T+U)/U = %g"%((PE+2*KE)/PE))

        if self.params['gas']['add_gas']:
            rescale = self.star_vel_rescaler(np.array(Singles['r']))
            vel *= rescale[:, np.newaxis]

        self.star_data = {}
        self.star_data['Masses'] = mass
        self.star_data['Coordinates'] = pos
        self.star_data['Velocities'] = vel
        ids = 1 + np.arange(nstar)
        self.star_data["ParticleIDs"] = ids
        self.star_data['ProtoStellarStage'] = np.repeat(5, nstar)

        if debug:
            for i in range(3):
                r_, sigma_ = utils.measure_spherical_density_profile(vel[:,i], pos, sigma=True, dN=1000)
                plt.plot(r_, sigma_,)
            plt.loglog()
            if self.params['gas']['add_gas']:
                plt.plot(r_, self.star_composite_sigma_r(r_*self.lunit_in_pc), 'k--')
            plt.plot(r_, self.star_only_sigma_r(r_*self.lunit_in_pc), 'k:')
            plt.xlabel('Radius')
            plt.ylabel('$\sigma_{1d}$')
            plt.show()


    def set_gas_particles(self, debug=False):
        print("Setting gas")
        N = int(self.params['gas']['M_gas']/self.params['gas']['res_gas'])

        # load a uniform unit sphere
        x = utils.load_glass_coords(N, method='')
        x = x*2-1
        r = np.linalg.norm(x, axis=-1)
        x = x[np.argsort(r)][:N]
        r = np.linalg.norm(x, axis=-1)
        x *= (1/2)**(1/3)/ ((r[N//2-1]+r[N//2])/2)

        # apply shape
        r = np.linalg.norm(x, axis=-1)
        m = (np.arange(1, len(x)+1))/len(x)
        r_new = self.r_M_func_gas_only(m)
        x *= (r_new/r)[:,np.newaxis]
        cut = np.linalg.norm(x, ord=np.inf, axis=-1)<self.params['box']['box_size']/2
        x = x[cut]
        r = np.linalg.norm(x, axis=-1)
        T = self.gas_T_envelope(r)
        U = 1.38e-16/1.67e-24*T/(self.params['gas']['gamma']-1) /self.params['gas']['mu']
        U /= (self.vunit_in_kms*1e5)**2

        m = np.ones(x.shape[0])*self.params['gas']['res_gas']

        # set gas data
        self.gas_data = {}
        self.gas_data['Masses'] = m/self.Munit_in_Msun
        self.gas_data['Coordinates'] = x/self.lunit_in_pc
        ids = np.max(self.star_data['ParticleIDs'])*10 + np.arange(len(m))
        self.gas_data["ParticleIDs"] = ids
        self.gas_data["InternalEnergy"] = U

        if debug:
            r_, rho_ = utils.measure_spherical_density_profile(m, x, sigma=False, dN=1000)
            plt.loglog(r_, rho_, label='raw IC')
            plt.loglog(r_, self.rho_func_gas_only(r_), 'k--')
            plt.xlabel("Radius")
            plt.ylabel("Gas density")
            plt.show()

    def power_spectrum_func(self, k):
        Nx = self.params['field']['Nx']
        kmin = Nx/20
        kmax = Nx/2
        alpha, beta, gamma = 6, -2, 3
        return k**beta/(1+(k/kmin)**(-alpha)) * np.exp(-(k/kmax)**gamma)

    def set_velocity_field(self, debug=False):
        print("Setting gas velocity")
        x = np.copy(self.gas_data['Coordinates'])*self.lunit_in_pc
        x += self.params['box']['box_size']/2
        x /= self.params['box']['box_size']
        assert x.min()>0 and x.max()<1

        envelope_func = self.gas_vel_envelope_normalized
        Nx = self.params['field']['Nx']
        shape = (Nx, Nx, Nx)
        power_spectrum_func = self.power_spectrum_func
        vx, vy, vz = fields.generate_random_v_with_envelope(shape, power_spectrum_func, envelope_func, sol_ratio=0.25, seed=43)
        vels = fields.interpolate_vector_field_to_particles(vx, vy, vz, x)

        
        if self.params['field']['compensate_sigma']:
            radius = np.linalg.norm(x-0.5, axis=-1)
            sort = np.argsort(radius)
            radius_sorted = radius[sort]
            dN = 1000
            N = len(radius)
            i = 0
            while i+dN<N:
                r1 = radius_sorted[i+dN] 
                r0 = radius_sorted[i]
                cut = (radius>=r0) & (radius<=r1)
                for j in range(3):
                    sigma = np.std(vels[cut][:,j])
                    sigma_target = envelope_func((r0+r1)/2) /np.sqrt(3) # now we use 1d velocity std
                    if sigma<sigma_target: 
                        dsigma = np.sqrt(sigma_target**2-sigma**2)
                        dv = np.random.normal(0, dsigma, int(np.sum(cut)))
                        vels[:,j][cut] += dv
                i += dN


        self.gas_data['Velocities'] = vels

        if debug:
            # plt.scatter(np.linalg.norm(self.gas_data['Coordinates'], axis=-1), np.linalg.norm(self.gas_data['Velocities'], axis=-1), s=.1)
            # plt.loglog()
            v = self.gas_data['Velocities']
            x = self.gas_data['Coordinates']
            for i in range(3):
                r_, sigma_ = utils.measure_spherical_density_profile(v[:,i], x*self.lunit_in_pc, sigma=True, dN=1000)
                plt.plot(r_, sigma_*np.sqrt(3), label='raw IC')
            plt.plot(r_, envelope_func(r_/self.params['box']['box_size']), 'k--')
            plt.loglog()
            plt.xlabel("Radius")
            plt.ylabel("3D velocity dispersion")
            plt.show()


    def set_magnetic_field(self, debug=False):
        print("Setting gas magnetic field")
        x = np.copy(self.gas_data['Coordinates'])*self.lunit_in_pc
        x += self.params['box']['box_size']/2
        x /= self.params['box']['box_size']
        assert x.min()>0 and x.max()<1

        envelope_func = self.gas_B_envelope_normalized
        Nx = self.params['field']['Nx']
        shape = (Nx, Nx, Nx)
        power_spectrum_func = self.power_spectrum_func
        Bx, By, Bz = fields.generate_periodic_B_with_envelope(shape, power_spectrum_func, envelope_func, seed=42)
        B = fields.interpolate_vector_field_to_particles(Bx, By, Bz, x)
        self.gas_data['MagneticField'] = B

        if debug:
            # plt.scatter(np.linalg.norm(self.gas_data['Coordinates'], axis=-1), np.linalg.norm(self.gas_data['Velocities'], axis=-1), s=.1)
            # plt.loglog()
            b = self.gas_data['MagneticField']
            x = self.gas_data['Coordinates']
            for i in range(3):
                r_, sigma_ = utils.measure_spherical_density_profile(b[:,i], x*self.lunit_in_pc, sigma=True, dN=1000)
                plt.plot(r_, sigma_*np.sqrt(3), label='raw IC')
            plt.plot(r_, envelope_func(r_/self.params['box']['box_size']), 'k--')
            plt.loglog()
            plt.xlabel("Radius")
            plt.ylabel("3D B dispersion")
            plt.show()

    def add_bhs(self, debug=False):
        if not self.params['bh']['add_bh']:
            return
        a = self.params['bh']['a']/self.lunit_in_pc
        M1 = self.params['bh']['M1']/self.Munit_in_Msun
        M2 = self.params['bh']['M2']/self.Munit_in_Msun
        T = np.sqrt((a)**3*np.pi**2*4/self.G/(M1+M2))
        v = np.sqrt(self.G*(M1+M2)/a)
        v1 = -M2/(M1+M2)*v
        v2 = M1/(M1+M2)*v
        x1 = -M2/(M1+M2)*a
        x2 = M1/(M1+M2)*a

        self.star_data['Masses'] = np.concatenate((self.star_data['Masses'], [M1, M2] ))
        self.star_data['Coordinates'] = np.concatenate((self.star_data['Coordinates'], [[x1, 0, 0], [x2, 0, 0]] ))
        self.star_data['Velocities'] = np.concatenate((self.star_data['Velocities'], [[0, v1, 0], [0, v2, 0]]))
        id_max = np.max(self.star_data["ParticleIDs"])
        self.star_data["ParticleIDs"] = np.concatenate((self.star_data['ParticleIDs'], [id_max+1, id_max+2]))
        self.star_data["ProtoStellarStage"] = np.concatenate((self.star_data['ProtoStellarStage'], [7, 7]))

    def check_bc(self, debug=False):
        if self.params['box']['periodic_box']:
            self.star_data['Coordinates'] += self.params['box']['box_size']/2 /self.lunit_in_pc
            if self.params['gas']['add_gas']:
                self.gas_data['Coordinates'] += self.params['box']['box_size']/2 /self.lunit_in_pc
        if debug:
            x = self.star_data['Coordinates']
            s = self.star_data["ProtoStellarStage"]
            plt.scatter(x[s==5][:,0], x[s==5][:,1], s=.1)
            plt.scatter(x[s==7][:,0], x[s==7][:,1], s=10)
            if self.params['gas']['add_gas']:
                x = self.gas_data['Coordinates']
                plt.scatter(x[:,0], x[:,1], s=.01, alpha=.5)
            plt.gca().set_aspect(1)
            plt.show()

    def write_snapshot(self):
        filename = self.params['snapshot']['file']
        with h5py.File(filename, "w") as F:
            F.create_group("Header")
            num_part = [0, 0, 0, 0, 0, 0]
            F["Header"].attrs["Time"] = 0.0

            if self.params['gas']['add_gas']:
                part_type = "PartType0"
                print(part_type)
                F.create_group(part_type)
                for k in self.gas_data.keys():
                    print(' ', k)
                    F[part_type].create_dataset(k, data=self.gas_data[k])
                num_part[0] = len(self.gas_data['Masses'])
            
            part_type = "PartType5"
            F.create_group(part_type)
            print(part_type)
            for k in self.star_data.keys():
                print(' ', k)
                F[part_type].create_dataset(k, data=self.star_data[k])
            num_part[5] = len(self.star_data['Masses'])

            # conclude
            F["Header"].attrs["NumPart_ThisFile"] = num_part
            F["Header"].attrs["NumPart_Total"] = num_part


if __name__ == '__main__':
    MakeGaseousCluster()