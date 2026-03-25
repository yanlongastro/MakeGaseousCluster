from cosmic.sample import InitialCMCTable
import numpy as np
from scipy.integrate import simpson, cumulative_trapezoid
import yaml
from scipy import interpolate
import pytreegrav
import h5py

import utils
import fields
import constants_units as cu
from MultiScaleVelocityGenerator import MultiScaleVelocityGenerator

import matplotlib.pyplot as plt

class MakeGaseousCluster:
    def __init__(self, param_file='params.yaml', debug=False):
        self.read_params(param_file, debug=debug)
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

    def read_params(self, param_file, debug=False):
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

        # units
        self.snapshot_units = cu.units(params['snapshot']['UnitMass_in_g'],
                                        params['snapshot']['UnitLength_in_cm'],
                                        params['snapshot']['UnitVelocity_in_cm_per_s'],
                                        params['snapshot']['UnitMagneticField_in_gauss']
                              )
        self.internal_units = cu.units(cu.Msun_cgs,
                                       cu.pc_cgs,
                                       1e5,
                                       1
                              )
        if debug:
            print('Snapshot G =', self.snapshot_units.G)
            print('Internal G =', self.internal_units.G)

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

        G = self.internal_units.G
        sigmas_star_only = np.array([utils.sigma_r_from_Jeans(x, self.rho_func_star_only, self.Menc_func_star_only, rmin, rmax, G=G) for x in r])
        sigmas_star_only[np.isnan(sigmas_star_only)] = np.nanmin(sigmas_star_only)
        if self.params['gas']['add_gas']:
            sigmas_star_composite = np.array([utils.sigma_r_from_Jeans(x, self.rho_func_star_only, self.Menc_func_star_gas, rmin, rmax, G=G) for x in r])
            sigmas_star_composite[np.isnan(sigmas_star_composite)] = np.nanmin(sigmas_star_composite)
            sigmas_gas_only = np.array([utils.sigma_r_from_Jeans(x, self.rho_func_gas_only, self.Menc_func_gas_only, rmin, rmax, G=G) for x in r])
            sigmas_gas_eff_composite = np.array([utils.sigma_r_from_Jeans(x, self.rho_func_gas_only, self.Menc_func_star_gas, rmin, rmax, G=G) for x in r])
            if self.params['field']['turbulent_B_field']:
                B_profile = self.params['gas']['B_at_center']*1e-6 * (rho_gas/rho_gas[0])**0.5 # gauss
                vA_profile = B_profile/np.sqrt(4*np.pi*rho_gas*self.internal_units.UnitDensity_in_cgs) /self.internal_units.UnitVelocity_in_cm_per_s
            else:
                B_profile = np.repeat(self.params['gas']['B_at_center']*1e-6, len(r))
                vA_profile = np.zeros_like(r) # !!!! this is purely mannual
            T_profile = np.exp(-(r/self.params['gas']['a_T'])**2) \
                *(self.params['gas']['T_at_center']-self.params['gas']['T_at_edge'])+self.params['gas']['T_at_edge']
            cs_profile = np.sqrt(self.params['gas']['gamma']*cu.kB_cgs*T_profile/(cu.mp_cgs*self.params['gas']['mu'])) /self.internal_units.UnitVelocity_in_cm_per_s
            
            # now, note that sigma_eff^2 = sigma_r^2+cs^2+vA^2
            sigmas_gas_composite = np.sqrt(sigmas_gas_eff_composite**2 -vA_profile**2 -cs_profile**2)
            # sigmas_gas_composite = sigmas_gas_eff_composite # !!!!!
            sigmas_gas_composite[np.isnan(sigmas_gas_composite)] = 0.1 # km/s, to avoid nans

        self.star_only_sigma_r = interpolate.InterpolatedUnivariateSpline(r, sigmas_star_only)
        if self.params['gas']['add_gas']:
            self.star_vel_rescaler = interpolate.InterpolatedUnivariateSpline(r, sigmas_star_composite/sigmas_star_only)
            self.star_composite_sigma_r = interpolate.InterpolatedUnivariateSpline(r, sigmas_star_composite)
            self.gas_sigma_r_envelope = interpolate.InterpolatedUnivariateSpline(r, sigmas_gas_composite)
            self.gas_vel_envelope_normalized = interpolate.InterpolatedUnivariateSpline(r/self.params['box']['box_size'], np.sqrt(3)*sigmas_gas_composite)
            self.gas_B_envelope = interpolate.InterpolatedUnivariateSpline(r, B_profile)
            self.gas_B_envelope_normalized = interpolate.InterpolatedUnivariateSpline(r/self.params['box']['box_size'], B_profile)
            self.gas_T_envelope = interpolate.InterpolatedUnivariateSpline(r, T_profile)

        

        if debug:
            plt.plot(r, sigmas_star_only, 'k--', label=r'star $\sigma_r$ (standalone)')
            if self.params['gas']['add_gas']:
                plt.plot(r, sigmas_star_composite, 'k:', label=r'star $\sigma_r$ (in composite)')
                plt.plot(r, sigmas_gas_only, 'r--', label=r'gas $\sigma_r$ (standalone)')
                plt.plot(r, sigmas_gas_eff_composite, 'g-', label=r'gas $\sigma_{\rm eff}$ (in composite)')
                plt.plot(r, sigmas_gas_composite, 'g--', label=r'gas $\sigma_r$ (in composite)')
                plt.plot(r, vA_profile, 'g:', label=r'gas $v_{\rm A}$ (in composite)')
                plt.plot(r, cs_profile, 'g-.', label=r'gas $c_{\rm s}$ (in composite)')

            r_, sigma_ = utils.measure_spherical_profile(Singles['vr'], Singles['r'], method='std', dN=1000)
            sigma_ *= np.sqrt(G*M_star/virial_radius) # Henon units -> internal units
            plt.plot(r_, sigma_, label='raw IC')
            plt.loglog()
            plt.xlabel('Radius [pc]')
            plt.ylabel('$\sigma_r$ [km/s]')
            plt.legend()
            plt.show()

    def set_star_particles(self, Singles, debug=False):
        print("Setting stars")
        mass = np.array(Singles['m'])
        radius = np.array(Singles['r'])
        vt = np.array(Singles['vt'])
        vr = np.array(Singles['vr'])
        v = np.sqrt(vt**2+vr**2)
        nstar = len(mass)
        ids = np.array(Singles['id'], dtype=int)

        pos = utils.uniform_sampling_on_spherical(nstar, seed=42)*radius[:,np.newaxis]
        vel = utils.uniform_sampling_on_spherical(nstar, seed=45)*v[:,np.newaxis]
        vel -= np.mean(vel, axis=0)

        PE = 0.5*np.sum(mass*pytreegrav.Potential(pos, mass, G=self.internal_units.G))
        KE = 0.5*np.sum(mass*np.linalg.norm(vel, axis=-1)**2)
        vel_fac = np.sqrt(-PE/(KE*2))
        vel *= vel_fac
        
        PE = 0.5*np.sum(mass*pytreegrav.Potential(pos, mass, G=self.internal_units.G))
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
                r_, sigma_ = utils.measure_spherical_profile(vel[:,i], pos, method='std', dN=1000)
                plt.semilogx(r_, sigma_, label='$\sigma_{%s}$'%('xyz'[i]))
            if self.params['gas']['add_gas']:
                plt.plot(r_, self.star_composite_sigma_r(r_), 'k--', label='$\sigma_r$ (in composite)')
            plt.plot(r_, self.star_only_sigma_r(r_), 'k:', label='$\sigma_r$ (standalone)')
            plt.xlabel('Radius [pc]')
            plt.ylabel('$\sigma_{1d}$ [km/s]')
            plt.legend()
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
        U = cu.kB_cgs*T/(cu.mp_cgs*self.params['gas']['mu'])/(self.params['gas']['gamma']-1) # cgs
        U /= (self.internal_units.UnitVelocity_in_cm_per_s)**2

        m = np.ones(x.shape[0])*self.params['gas']['res_gas']

        # set gas data
        self.gas_data = {}
        self.gas_data['Masses'] = m
        self.gas_data['Coordinates'] = x
        ids = np.max(self.star_data['ParticleIDs'])*10 + np.arange(len(m))
        self.gas_data["ParticleIDs"] = ids
        self.gas_data["InternalEnergy"] = U

        if debug:
            r_, rho_ = utils.measure_spherical_profile(m, x, method='density', dN=1000)
            plt.loglog(r_, rho_, label='raw IC')
            plt.loglog(r_, self.rho_func_gas_only(r_), 'k--', label='model')
            plt.legend()
            plt.xlabel("Radius [pc]")
            plt.ylabel(r"Gas density [$\rm M_\odot/pc^3$]")
            plt.show()

            r_, rho_ = utils.measure_spherical_profile(U, x, method='rms', dN=1000)
            plt.semilogx(r_, np.sqrt(rho_), label='raw IC')
            plt.legend()
            plt.xlabel("Radius [pc]")
            plt.ylabel(r"[$\sqrt{e}$ [km/s]")
            plt.show()

    def power_spectrum_func_unit_box(self, k):
        Nx = self.params['field']['Nx']
        kmin = Nx/20
        kmax = Nx/2
        alpha, beta, gamma = 6, -2, 3
        return k**beta/(1+(k/kmin)**(-alpha)) * np.exp(-(k/kmax)**gamma)
    
    def power_spectrum_func(self, k):
        kmin = 10 # 1/pc
        kmax = 1000 # 1/pc
        alpha, beta, gamma = 6, -2, 3
        return k**beta/(1+(k/kmin)**(-alpha)) * np.exp(-(k/kmax)**gamma)

    def set_config_for_msvg(self):
        L = self.params['box']['box_size']
        Ls = self.r_M_func_gas_only([0.2, 0.5, 1])
        Ls = np.sort(Ls)[::-1]*2
        Ls[0] = L

        config = {
            "particles":          None,
            "boxes": [
                {"size": Lx, "resolution": 256, "center": [L/2, L/2, L/2]} for Lx in Ls
            ],
            "power_spectrum":     self.power_spectrum_func,
            "sigma_r_target":     None,
            "solenoidal_ratio":   None,
            "transition_width":   0.3,
            "max_iterations":     10,
            "tolerance":          0.02,
            "relaxation_omega":   0.9,
            "n_sigma_bins":       50,
            "seed":               None,
            "enable_correction":  True,
        }
        return config


    def set_velocity_field(self, debug=False, legacy_method=False):
        print("Setting gas velocity")
        x = np.copy(self.gas_data['Coordinates'])

        if legacy_method:
            x += self.params['box']['box_size']/2
            x /= self.params['box']['box_size']
            assert x.min()>0 and x.max()<1

            envelope_func = self.gas_vel_envelope_normalized
            Nx = self.params['field']['Nx']
            shape = (Nx, Nx, Nx)
            power_spectrum_func = self.power_spectrum_func_unit_box
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

        else:
            config = self.set_config_for_msvg()
            config["particles"] = x + self.params['box']['box_size']/2
            config["sigma_r_target"] = self.gas_sigma_r_envelope
            config["solenoidal_ratio"] = lambda k: 0.25
            config["seed"] = 34
            gen = MultiScaleVelocityGenerator(config)
            vels   = gen.generate()
            results = gen.validate()
            

        # vels -= np.mean(vels, axis=0)
        self.gas_data['Velocities'] = vels

        if debug:
            # plt.scatter(np.linalg.norm(self.gas_data['Coordinates'], axis=-1), np.linalg.norm(self.gas_data['Velocities'], axis=-1), s=.1)
            # plt.loglog()
            v = self.gas_data['Velocities']
            x = self.gas_data['Coordinates']
            for i in range(3):
                r_, sigma_ = utils.measure_spherical_profile(v[:,i], x, method='std', dN=1000)
                plt.plot(r_, sigma_, label='$\sigma_{%s}$'%('xyz'[i]))
            if legacy_method:
                plt.semilogx(r_, envelope_func(r_/self.params['box']['box_size'])/np.sqrt(3), 'k--', label='model')
            else:
                plt.semilogx(r_, self.gas_sigma_r_envelope(r_), 'k--', label='model')
            plt.legend()
            plt.xlabel("Radius [pc]")
            plt.ylabel("1D velocity dispersion [km/s]")
            plt.show()


    def set_magnetic_field(self, debug=False):
        print("Setting gas magnetic field")
        envelope_func = self.gas_B_envelope_normalized

        if self.params['field']['turbulent_B_field']:
            print(" set a turbulent B field")
            x = np.copy(self.gas_data['Coordinates'])
            x += self.params['box']['box_size']/2
            x /= self.params['box']['box_size']
            assert x.min()>0 and x.max()<1

            Nx = self.params['field']['Nx']
            shape = (Nx, Nx, Nx)
            power_spectrum_func = self.power_spectrum_func_unit_box
            Bx, By, Bz = fields.generate_periodic_B_with_envelope(shape, power_spectrum_func, envelope_func, seed=42)
            B = fields.interpolate_vector_field_to_particles(Bx, By, Bz, x)
        else:
            print(" set a uniform B field")
            B = np.array([0, 0, self.params['gas']['B_at_center']*1e-6]) * np.ones_like(self.gas_data['Coordinates'])[:,np.newaxis]
        self.gas_data['MagneticField'] = B

        if debug:
            # plt.scatter(np.linalg.norm(self.gas_data['Coordinates'], axis=-1), np.linalg.norm(self.gas_data['Velocities'], axis=-1), s=.1)
            # plt.loglog()
            b = self.gas_data['MagneticField']
            x = self.gas_data['Coordinates']
            bmag = np.linalg.norm(b, axis=-1)
            r_, sigma_ = utils.measure_spherical_profile(bmag, x, method='rms', dN=1000)
            plt.plot(r_, sigma_, label=r'$B_{\rm rms}$')
            plt.plot(r_, envelope_func(r_/self.params['box']['box_size']), 'k--', label='model')
            plt.loglog()
            plt.xlabel("Radius [pc]")
            plt.ylabel("1D B dispersion [G]")
            plt.show()

    def add_bhs(self, debug=False):
        if not self.params['bh']['add_bh']:
            return
        a = self.params['bh']['a']
        M1 = self.params['bh']['M1']
        M2 = self.params['bh']['M2']
        v = np.sqrt(self.internal_units.G*(M1+M2)/a)
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
            self.star_data['Coordinates'] += self.params['box']['box_size']/2
            if self.params['gas']['add_gas']:
                self.gas_data['Coordinates'] += self.params['box']['box_size']/2
        if debug:
            x = self.star_data['Coordinates']
            s = self.star_data["ProtoStellarStage"]
            plt.scatter(x[s==5][:,0], x[s==5][:,1], s=.1)
            plt.scatter(x[s==7][:,0], x[s==7][:,1], s=10)
            if self.params['gas']['add_gas']:
                x = self.gas_data['Coordinates']
                plt.scatter(x[:,0], x[:,1], s=.01, alpha=.5)
            plt.gca().set_aspect(1)
            plt.xlabel(r'$x$ [pc]')
            plt.ylabel(r'$y$ [pc]')
            plt.show()

    def save_dict_to_hdf5(self, F, dict_dat, part_type, num_part):
        group_name = "PartType%d"%part_type
        if part_type==0:
            assert 'InternalEnergy' in dict_dat.keys()
        print(group_name)
        F.create_group(group_name)
        for k in dict_dat.keys():
            print(' ', k)
            data = dict_dat[k]

            # do some unit conversions
            if k in ["Masses"]:
                data *= self.internal_units.UnitMass_in_g / self.snapshot_units.UnitMass_in_g
            if k in ["Coordinates"]:
                data *= self.internal_units.UnitLength_in_cm / self.snapshot_units.UnitLength_in_cm
            if k in ["Velocities"]:
                data *= self.internal_units.UnitVelocity_in_cm_per_s / self.snapshot_units.UnitVelocity_in_cm_per_s
            if k in ["MagneticField"]:
                data *= self.internal_units.UnitMagneticField_in_gauss / self.snapshot_units.UnitMagneticField_in_gauss
            if k in ['InternalEnergy']:
                data *= (self.internal_units.UnitVelocity_in_cm_per_s / self.snapshot_units.UnitVelocity_in_cm_per_s)**2
            
            F[group_name].create_dataset(k, data=data)
        num_part[part_type] = len(dict_dat['Masses'])
        return num_part


    def write_snapshot(self):
        filename = self.params['snapshot']['file']
        with h5py.File(filename, "w") as F:
            F.create_group("Header")
            num_part = [0, 0, 0, 0, 0, 0]
            F["Header"].attrs["Time"] = 0.0

            if self.params['gas']['add_gas']:
                num_part = self.save_dict_to_hdf5(F, self.gas_data, 0, num_part)
            num_part = self.save_dict_to_hdf5(F, self.star_data, 5, num_part)

            # conclude
            F["Header"].attrs["NumPart_ThisFile"] = num_part
            F["Header"].attrs["NumPart_Total"] = num_part

if __name__ == '__main__':
    MakeGaseousCluster()