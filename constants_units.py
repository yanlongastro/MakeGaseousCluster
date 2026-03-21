import numpy as np


G_cgs = 6.6743e-8
yr_cgs = 365.2422*24*3600
c_cgs = 299792458*100
pc_cgs = 3.08568e18
kB_cgs = 1.380649e-16
mp_cgs = 1.6726219e-24
Msun_cgs = 1.989e33
Rsun_cgs = 6.957e10

class units:
    def __init__(self, UnitMass_in_g=1.989e+43, UnitLength_in_cm=3.08568e+21, UnitVelocity_in_cm_per_s=1e5, UnitMagneticField_in_gauss=1,
                 h=1):
        
        self.UnitMass_in_g = UnitMass_in_g/h
        self.UnitLength_in_cm = UnitLength_in_cm
        self.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s
        self.UnitMagneticField_in_gauss = UnitMagneticField_in_gauss
        self.h = h
        
        self.derive_units()
        self.derive_constants()
        
    def derive_units(self):
        self.UnitTime_in_s = self.UnitLength_in_cm/self.UnitVelocity_in_cm_per_s
        self.UnitTime_in_yr = self.UnitTime_in_s/yr_cgs
        self.UnitTime_in_Myr = self.UnitTime_in_yr/1e6
        self.UnitTime_in_Gyr = self.UnitTime_in_yr/1e9
        self.UnitMass_in_solar = self.UnitMass_in_g/Msun_cgs
        self.UnitLength_in_pc = self.UnitLength_in_cm/pc_cgs
        self.UnitVelocity_in_m_per_s = self.UnitVelocity_in_cm_per_s/100
        self.UnitDensity_in_cgs = self.UnitMass_in_g/self.UnitLength_in_cm**3
        self.UnitPressure_in_cgs = self.UnitDensity_in_cgs*self.UnitVelocity_in_cm_per_s**2
    
    def derive_constants(self):
        self.G = G_cgs*self.UnitMass_in_g**(1) *self.UnitLength_in_cm**(-3) *self.UnitTime_in_s**(2)
        self.c = c_cgs/self.UnitVelocity_in_cm_per_s

    def dynamical_time(self, M, R, output_unit=None):
        tdyn =  np.sqrt(R**3/self.G/M)
        if output_unit is not None:
            tdyn *= getattr(self, 'UnitTime_in_'+output_unit)
        return tdyn

    def free_fall_time(self, M, R, output_unit=None):
        tff =  np.pi*np.sqrt(1/8)*self.dynamical_time(M, R, output_unit)
    
    def circular_period(self, M, R, output_unit=None):
        return 2*np.pi*self.dynamical_time(M, R, output_unit)

cgs = units(1, 1, 1, 1)
SI = units(1000, 100, 100, 1e4)
FIRE = units(h=0.68)
FIRE_noncosmological = units()
STARFORGE = units(Msun_cgs, pc_cgs, 1e2, 1e4)
star = units(Msun_cgs, Rsun_cgs, 1e5, 1)