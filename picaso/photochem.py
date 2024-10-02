"""
This module contains wrappers for the "Photochem" photochemical
model (https://github.com/Nicholaswogan/photochem). These wrappers are 
called in "justdoit.py" during climate simulations if photochemistry
is turned on. The only function useful for general users is 
`generate_photochem_rx_and_thermo_file`, which can generate reaction and 
thermodynamic files used for initializing Photochem
"""

import numpy as np
import numba as nb
from numba import types
from scipy import constants as const
from scipy import integrate
from tempfile import NamedTemporaryFile
import copy

import pkg_resources
import warnings
if pkg_resources.get_distribution("photochem").version != '0.5.6':
    warnings.warn('You have photochem version '+pkg_resources.get_distribution("photochem").version
                  +' installed, but version 0.5.6 is recommended.')
from photochem import EvoAtmosphere, PhotoException, zahnle_earth
from photochem import equilibrate
from photochem.utils._format import yaml, FormatSettings_main, MyDumper, FormatReactions_main

# Turn off Panda's performance warnings
import pandas as pd
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    
@nb.cfunc(nb.double(nb.double, nb.double, nb.double))
def custom_binary_diffusion_fcn(mu_i, mubar, T):
    # Equation 6 in Gladstone et al. (1996)
    b = 3.64e-5*T**(1.75-1.0)*7.3439e21*np.sqrt(2.01594/mu_i)
    return b

###
### Extension of EvoAtmosphere class for gas giants
###

class EvoAtmosphereGasGiant(EvoAtmosphere):

    def __init__(self, mechanism_file, stellar_flux_file, planet_mass, planet_radius, 
                 nz=100, photon_scale_factor=1.0, P_ref=1.0e6, thermo_file=None):
        """Initializes the code

        Parameters
        ----------
        mechanism_file : str
            Path to the file describing the reaction mechanism
        stellar_flux_file : str
            Path to the file describing the stellar UV flux.
        planet_mass : float
            Planet mass in grams
        planet_radius : float
            Planet radius in cm
        nz : int, optional
            The number of layers in the photochemical model, by default 100
        P_ref : float, optional
            Pressure level corresponding to the planet_radius, by default 1e6 dynes/cm^2
        thermo_file : str, optional
            Optionally include a dedicated thermodynamic file.
        """        
        
        # First, initialize photochemical model with dummy inputs
        sol = yaml.safe_load(SETTINGS_TEMPLATE)
        sol['atmosphere-grid']['number-of-layers'] = int(nz)
        sol['planet']['planet-mass'] = float(planet_mass)
        sol['planet']['planet-radius'] = float(planet_radius)
        sol['planet']['photon-scale-factor'] = float(photon_scale_factor)
        sol = FormatSettings_main(sol)
        with NamedTemporaryFile('w',suffix='.txt') as ff:
            ff.write(ATMOSPHERE_INIT)
            ff.flush()
            with NamedTemporaryFile('w',suffix='.yaml') as f:
                yaml.dump(sol,f,Dumper=MyDumper)
                super().__init__(
                    mechanism_file,
                    f.name,
                    stellar_flux_file,
                    ff.name
                )

        # Save inputs that matter
        self.planet_radius = planet_radius
        self.planet_mass = planet_mass
        self.P_ref = P_ref

        # Parameters using during initialization
        # The factor of pressure the atmosphere extends
        # compared to predicted quench points of gases
        self.BOA_pressure_factor = 1.0
        # If True, then the guessed initial condition will used
        # quenching relations as an initial guess
        self.initial_cond_with_quenching = True
        # For computing chemical equilibrium at a metallicity.
        if thermo_file is None:
            thermo_file = mechanism_file
        self.m = Metallicity(thermo_file)

        # Parameters for determining steady state
        self.TOA_pressure_avg = 1.0e-7*1e6 # mean TOA pressure (dynes/cm^2)
        self.max_dT_tol = 5 # The permitted difference between T in photochem and desired T
        self.max_dlog10edd_tol = 0.2 # The permitted difference between Kzz in photochem and desired Kzz
        self.freq_update_PTKzz = 1000 # step frequency to update PTKzz profile.
        self.max_total_step = 100_000 # Maximum total allowed steps before giving up
        self.min_step_conv = 300 # Min internal steps considered before convergence is allowed
        self.verbose = True # print information or not?
        self.freq_print = 100 # Frequency in which to print

        # Values in photochem to adjust
        self.var.verbose = 0
        self.var.upwind_molec_diff = True
        self.var.autodiff = True # Turn on autodiff
        self.var.atol = 1.0e-18
        self.var.conv_min_mix = 1e-10 # Min mix to consider during convergence check
        self.var.conv_longdy = 0.01 # threshold relative change that determines convergence
        self.var.custom_binary_diffusion_fcn = custom_binary_diffusion_fcn

        # Values that will be needed later. All of these set
        # in `initialize_to_climate_equilibrium_PT`
        self.P_clima_grid = None # The climate grid
        self.metallicity = None
        self.CtoO = None
        # Below for interpolation
        self.log10P_interp = None
        self.T_interp = None
        self.log10edd_interp = None
        self.P_desired = None
        self.T_desired = None
        self.Kzz_desired = None
        # Index of climate grid that is bottom of photochemical grid
        self.ind_b = None
        # information needed during robust stepping
        self.total_step_counter = None
        self.nerrors = None
        self.robust_stepper_initialized = None

    def initialize_to_climate_equilibrium_PT(self, P_in, T_in, Kzz_in, metallicity, CtoO, rainout_condensed_atoms=True):
        """Initialized the photochemical model to a climate model result that assumes chemical equilibrium
        at some metallicity and C/O ratio.

        Parameters
        ----------
        P_in : ndarray[dim=1,double]
            The pressures in the climate grid (dynes/cm^2). P_in[0] is pressure at
            the deepest layer of the atmosphere
        T_in : ndarray[dim1,double]
            The temperatures in the climate grid corresponding to P_in (K)
        Kzz_in : ndarray[dim1,double]
            The eddy diffusion at each pressure P_in (cm^2/s)
        metallicity : float
            Metallicity relative to solar.
        CtoO : float
            C/O ratio relative to solar. So CtoO = 1 is solar C/O ratio.
            CtoO = 2 is twice the solar C/O ratio.
        """

        if P_in.shape[0] != T_in.shape[0]:
            raise Exception('Input P and T must have same shape')
        if P_in.shape[0] != Kzz_in.shape[0]:
            raise Exception('Input P and Kzz must have same shape')

        # Save inputs
        self.P_clima_grid = P_in
        self.metallicity = metallicity
        self.CtoO = CtoO

        # Compute chemical equilibrium along the whole P-T profile
        mix, mubar = self.m.composition(T_in, P_in, CtoO, metallicity, rainout_condensed_atoms)

        if self.TOA_pressure_avg*3 > P_in[-1]:
            raise Exception('The photochemical grid needs to extend above the climate grid')

        # Altitude of P-T grid
        P1, T1, mubar1, z1 = compute_altitude_of_PT(P_in, self.P_ref, T_in, mubar, self.planet_radius, self.planet_mass, self.TOA_pressure_avg)
        # If needed, extrapolate Kzz and mixing ratios
        if P1.shape[0] != Kzz_in.shape[0]:
            Kzz1 = np.append(Kzz_in,Kzz_in[-1])
            mix1 = {}
            for sp in mix:
                mix1[sp] = np.append(mix[sp],mix[sp][-1])
        else:
            Kzz1 = Kzz_in.copy()
            mix1 = mix

        # The gravity
        grav1 = gravity(self.planet_radius, self.planet_mass, z1)

        # Next, we compute the quench levels
        quench_levels = determine_quench_levels(T1, P1, Kzz1, mubar1, grav1)
        ind = np.min(quench_levels) # the deepest quench level

        # If desired, this bit applies quenched initial conditions, and recomputes
        # the altitude profile for this new mubar.
        if self.initial_cond_with_quenching:

            # Apply quenching to mixing ratios
            mix1['CH4'][quench_levels[0]:] = mix1['CH4'][quench_levels[0]]
            mix1['CO'][quench_levels[0]:] = mix1['CO'][quench_levels[0]]
            mix1['CO2'][quench_levels[1]:] = mix1['CO2'][quench_levels[1]]
            mix1['NH3'][quench_levels[2]:] = mix1['NH3'][quench_levels[2]]
            mix1['HCN'][quench_levels[3]:] = mix1['HCN'][quench_levels[3]]

            # Quenching out H2 at the CH4 level seems to work well
            mix1['H2'][quench_levels[0]:] = mix1['H2'][quench_levels[0]]

            # Normalize mixing ratios
            mix_tot = np.zeros(mix1['CH4'].shape[0])
            for key in mix1:
                mix_tot += mix1[key]
            for key in mix1:
                mix1[key] = mix1[key]/mix_tot

            # Compute mubar again
            mubar1[:] = 0.0
            for i,sp in enumerate(self.dat.species_names[:-2]):
                if sp in mix1:
                    for j in range(P1.shape[0]):
                        mubar1[j] += mix1[sp][j]*self.dat.species_mass[i]

            # Update z1 to get a new altitude profile
            P1, T1, mubar1, z1 = compute_altitude_of_PT(P1, self.P_ref, T1, mubar1, self.planet_radius, self.planet_mass, self.TOA_pressure_avg)

        # Save P-T-Kzz for later interpolation and corrections
        self.log10P_interp = np.log10(P1.copy()[::-1])
        self.T_interp = T1.copy()[::-1]
        self.log10edd_interp = np.log10(Kzz1.copy()[::-1])
        self.P_desired = P1.copy()
        self.T_desired = T1.copy()
        self.Kzz_desired = Kzz1.copy()

        # Bottom of photochemical model will be at a pressure a factor
        # larger than the predicted quench pressure.
        if P1[ind]*self.BOA_pressure_factor > P1[0]:
            raise Exception('BOA in photochemical model wants to be deeper than BOA of climate model.')
        self.ind_b = np.argmin(np.abs(P1 - P1[ind]*self.BOA_pressure_factor))
        
        self._initialize_atmosphere(P1, T1, Kzz1, z1, mix1)

    def reinitialize_to_new_climate_PT(self, P_in, T_in, Kzz_in, mix):
        """Reinitializes the photochemical model to the input P, T, Kzz, and mixing ratios
        from the climate model.

        Parameters
        ----------
        P_in : ndarray[ndim=1,double]
            Pressure grid in climate model (dynes/cm^2).
        T_in : ndarray[ndim=1,double]
            Temperatures corresponding to P_in (K)
        Kzz_in : ndarray[ndim,double]
            Eddy diffusion coefficients at each pressure level (cm^2/s)
        mix : dict
            Mixing ratios of all species in the atmosphere

        """        

        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')
        if not np.all(np.isclose(self.P_clima_grid,P_in)):
            raise Exception('Input pressure grid does not match saved pressure grid')
        if P_in.shape[0] != T_in.shape[0]:
            raise Exception('Input P and T must have same shape')
        if P_in.shape[0] != Kzz_in.shape[0]:
            raise Exception('Input P and Kzz must have same shape')
        for key in mix:
            if P_in.shape[0] != mix[key].shape[0]:
                raise Exception('Input P and mix must have same shape')
        # Require all gases be specified. Particles can be ignored.
        if set(list(mix.keys())) != set(self.dat.species_names[self.dat.np:(-2-self.dat.nsl)]):
            raise Exception('Some species are missing from input mix') 
        
        # Compute mubar
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        mubar = np.zeros(T_in.shape[0])
        species_mass = self.dat.species_mass
        particle_names = self.dat.species_names[:self.dat.np]
        for sp in mix:
            if sp not in particle_names:
                ind = species_names.index(sp)
                mubar = mubar + mix[sp]*species_mass[ind]

        # Compute altitude of P-T grid
        P1, T1, mubar1, z1 = compute_altitude_of_PT(P_in, self.P_ref, T_in, mubar, self.planet_radius, self.planet_mass, self.TOA_pressure_avg)
        # If needed, extrapolte Kzz and mixing ratios
        if P1.shape[0] != Kzz_in.shape[0]:
            Kzz1 = np.append(Kzz_in,Kzz_in[-1])
            mix1 = {}
            for sp in mix:
                mix1[sp] = np.append(mix[sp],mix[sp][-1])
        else:
            Kzz1 = Kzz_in.copy()
            mix1 = mix

        # Save P-T-Kzz for later interpolation and corrections
        self.log10P_interp = np.log10(P1.copy()[::-1])
        self.T_interp = T1.copy()[::-1]
        self.log10edd_interp = np.log10(Kzz1.copy()[::-1])
        self.P_desired = P1.copy()
        self.T_desired = T1.copy()
        self.Kzz_desired = Kzz1.copy()

        self._initialize_atmosphere(P1, T1, Kzz1, z1, mix1)

    def _initialize_atmosphere(self, P1, T1, Kzz1, z1, mix1):
        "Little helper function preventing code duplication."

        # Compute TOA index
        ind_t = np.argmin(np.abs(P1 - self.TOA_pressure_avg))

        # Shift z profile so that zero is at photochem BOA
        z1_p = z1 - z1[self.ind_b]

        # Calculate the photochemical grid
        z_top = z1_p[ind_t]
        z_bottom = 0.0
        dz = (z_top - z_bottom)/self.var.nz
        z_p = np.empty(self.var.nz)
        z_p[0] = dz/2.0
        for i in range(1,self.var.nz):
            z_p[i] = z_p[i-1] + dz

        # Now, we interpolate all values to the photochemical grid
        P_p = 10.0**np.interp(z_p, z1_p, np.log10(P1))
        T_p = np.interp(z_p, z1_p, T1)
        Kzz_p = 10.0**np.interp(z_p, z1_p, np.log10(Kzz1))
        mix_p = {}
        for sp in mix1:
            mix_p[sp] = 10.0**np.interp(z_p, z1_p, np.log10(mix1[sp]))
        k_boltz = const.k*1e7
        den_p = P_p/(k_boltz*T_p)

        # Compute new planet radius
        planet_radius_new = self.planet_radius + z1[self.ind_b]

        # Update photochemical model grid
        self.dat.planet_radius = planet_radius_new
        self.update_vertical_grid(TOA_alt=z_top) # this will update gravity for new planet radius
        self.set_temperature(T_p)
        self.var.edd = Kzz_p
        usol = np.ones(self.wrk.usol.shape)*1e-40
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for sp in mix_p:
            if sp in species_names:
                ind = species_names.index(sp)
                usol[ind,:] = mix_p[sp]*den_p
        self.wrk.usol = usol

        # Now set boundary conditions
        for i,sp in enumerate(species_names):
            if i >= self.dat.np:
                self.set_lower_bc(sp, bc_type='Moses') # gas
            else:
                self.set_lower_bc(sp, bc_type='vdep', vdep=0.0) # particle
        particle_names = self.dat.species_names[:self.dat.np]
        for sp in mix_p:
            if sp not in particle_names:
                Pi = P_p[0]*mix_p[sp][0]
                self.set_lower_bc(sp, bc_type='press', press=Pi)

        self.prep_atmosphere(self.wrk.usol)

    def return_atmosphere_climate_grid(self):
        """Returns a dictionary with temperature, Kzz and mixing ratios
        on the climate model grid.

        Returns
        -------
        dict
            Contains temperature, Kzz, and mixing ratios.
        """ 
        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')

        # return full atmosphere
        out = self.return_atmosphere()

        # Interpolate full atmosphere to clima grid
        sol = {}
        sol['pressure'] = self.P_clima_grid.copy()
        log10Pclima = np.log10(self.P_clima_grid[::-1]).copy()
        log10P = np.log10(out['pressure'][::-1]).copy()

        T = np.interp(log10Pclima, log10P, out['temperature'][::-1].copy())
        sol['temperature'] = T[::-1].copy()

        Kzz = np.interp(log10Pclima, log10P, np.log10(out['Kzz'][::-1].copy()))
        sol['Kzz'] = 10.0**Kzz[::-1].copy()

        for key in out:
            if key not in ['pressure','temperature','Kzz']:
                tmp = np.log10(np.clip(out[key][::-1].copy(),a_min=1e-100,a_max=np.inf))
                mix = np.interp(log10Pclima, log10P, tmp)
                sol[key] = 10.0**mix[::-1].copy()

        return sol

    def return_atmosphere(self, include_deep_atmosphere = True, equilibrium = False, rainout_condensed_atoms = True):
        """Returns a dictionary with temperature, Kzz and mixing ratios
        on the photochemical grid.

        Parameters
        ----------
        include_deep_atmosphere : bool, optional
            If True, then results will include portions of the deep
            atomsphere that are not part of the photochemical grid, by default True

        Returns
        -------
        dict
            Contains temperature, Kzz, and mixing ratios.
        """        

        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')

        out = {}
        out['pressure'] = self.wrk.pressure_hydro
        out['temperature'] = self.var.temperature
        out['Kzz'] = self.var.edd
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        if equilibrium:
            mix, mubar = self.m.composition(out['temperature'], out['pressure'], self.CtoO, self.metallicity, rainout_condensed_atoms)
            for key in mix:
                out[key] = mix[key]
            for key in species_names[:self.dat.np]:
                out[key] = np.zeros(mix['H2'].shape[0])
        else:
            for i,sp in enumerate(species_names):
                mix = self.wrk.usol[i,:]/self.wrk.density
                out[sp] = mix

        if not include_deep_atmosphere:
            return out

        # Prepend the deeper atmosphere, which we will assume is at Equilibrium
        inds = np.where(self.P_desired > self.wrk.pressure_hydro[0])
        out1 = {}
        out1['pressure'] = self.P_desired[inds]
        out1['temperature'] = self.T_desired[inds]
        out1['Kzz'] = self.Kzz_desired[inds]
        mix, mubar = self.m.composition(out1['temperature'], out1['pressure'], self.CtoO, self.metallicity, rainout_condensed_atoms)
        
        out['pressure'] = np.append(out1['pressure'],out['pressure'])
        out['temperature'] = np.append(out1['temperature'],out['temperature'])
        out['Kzz'] = np.append(out1['Kzz'],out['Kzz'])
        for i,sp in enumerate(species_names):
            if sp in mix:
                out[sp] = np.append(mix[sp],out[sp])
            else:
                out[sp] = np.append(np.zeros(mix['H2'].shape[0]),out[sp])

        return out
    
    def initialize_robust_stepper(self, usol):
        """Initialized a robust integrator.

        Parameters
        ----------
        usol : ndarray[double,dim=2]
            Input number densities
        """        
        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')
        
        self.total_step_counter = 0
        self.nerrors = 0
        self.initialize_stepper(usol)
        self.robust_stepper_initialized = True

    def robust_step(self):
        """Takes a single robust integrator step

        Returns
        -------
        tuple
            The tuple contains two bools `give_up, reached_steady_state`. If give_up is True
            then the algorithm things it is time to give up on reaching a steady state. If
            reached_steady_state then the algorithm has reached a steady state within
            tolerance.
        """        
        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')

        if not self.robust_stepper_initialized:
            raise Exception('This routine can only be called after `initialize_robust_stepper`')

        give_up = False
        reached_steady_state = False

        for i in range(1):
            try:
                self.step()
                self.total_step_counter += 1
            except PhotoException as e:
                # If there is an error, lets reinitialize, but get rid of any
                # negative numbers
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                self.nerrors += 1

                if self.nerrors > 10:
                    give_up = True
                    break

            # convergence checking
            converged = self.check_for_convergence()

            # Compute the max difference between the P-T profile in photochemical model
            # and the desired P-T profile
            T_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), self.log10P_interp, self.T_interp)
            T_p = T_p.copy()[::-1]
            max_dT = np.max(np.abs(T_p - self.var.temperature))

            # Compute the max difference between the P-edd profile in photochemical model
            # and the desired P-edd profile
            log10edd_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), self.log10P_interp, self.log10edd_interp)
            log10edd_p = log10edd_p.copy()[::-1]
            max_dlog10edd = np.max(np.abs(log10edd_p - np.log10(self.var.edd)))

            # TOA pressure
            TOA_pressure = self.wrk.pressure_hydro[-1]

            condition1 = converged and self.wrk.nsteps > self.min_step_conv or self.wrk.tn > self.var.equilibrium_time
            condition2 = max_dT < self.max_dT_tol and max_dlog10edd < self.max_dlog10edd_tol and self.TOA_pressure_avg/3 < TOA_pressure < self.TOA_pressure_avg*3

            if condition1 and condition2:
                if self.verbose:
                    print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                        (self.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                # success!
                reached_steady_state = True
                break

            if not (self.wrk.nsteps % self.freq_update_PTKzz) or (condition1 and not condition2):
                # After ~1000 steps, lets update P,T, edd and vertical grid, if possible.
                try:
                    self.set_press_temp_edd(self.P_desired,self.T_desired,self.Kzz_desired,hydro_pressure=True)
                except PhotoException:
                    pass
                try:
                    self.update_vertical_grid(TOA_pressure=self.TOA_pressure_avg)
                except PhotoException:
                    pass
                self.initialize_stepper(self.wrk.usol)

            if self.total_step_counter > self.max_total_step:
                give_up = True
                break

            if not (self.wrk.nsteps % self.freq_print) and self.verbose:
                print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                    (self.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                
        return give_up, reached_steady_state
    
    def find_steady_state(self):
        """Attempts to find a photochemical steady state.

        Returns
        -------
        bool
            If True, then the routine was successful.
        """    

        self.initialize_robust_stepper(self.wrk.usol)
        success = True
        while True:
            give_up, reached_steady_state = self.robust_step()
            if reached_steady_state:
                break
            if give_up:
                success = False
                break
        return success
    
    def model_state_to_dict(self):
        """Returns a dictionary containing all information needed to reinitialize the atmospheric
        state. This dictionary can be used as an input to "initialize_from_dict".
        """

        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')

        out = {}
        out['P_clima_grid'] = self.P_clima_grid
        out['metallicity'] = self.metallicity
        out['CtoO'] = self.CtoO
        out['log10P_interp'] = self.log10P_interp
        out['T_interp'] = self.T_interp
        out['log10edd_interp'] = self.log10edd_interp
        out['P_desired'] = self.P_desired
        out['T_desired'] = self.T_desired
        out['Kzz_desired'] = self.Kzz_desired
        out['ind_b'] = self.ind_b
        out['planet_radius_new'] = self.dat.planet_radius
        out['top_atmos'] = self.var.top_atmos
        out['temperature'] = self.var.temperature
        out['edd'] = self.var.edd
        out['usol'] = self.wrk.usol
        out['P_i_surf'] = (self.wrk.usol[self.dat.np:,0]/self.wrk.density[0])*self.wrk.pressure[0]

        return out

    def initialize_from_dict(self, out):
        """Initializes the model from a dictionary created by the "model_state_to_dict" routine.
        """

        self.P_clima_grid = out['P_clima_grid']
        self.metallicity = out['metallicity']
        self.CtoO = out['CtoO']
        self.log10P_interp = out['log10P_interp']
        self.T_interp = out['T_interp']
        self.log10edd_interp = out['log10edd_interp']
        self.P_desired = out['P_desired']
        self.T_desired = out['T_desired']
        self.Kzz_desired = out['Kzz_desired']
        self.ind_b = out['ind_b']
        self.dat.planet_radius = out['planet_radius_new']
        self.update_vertical_grid(TOA_alt=out['top_atmos'])
        self.set_temperature(out['temperature'])
        self.var.edd = out['edd']
        self.wrk.usol = out['usol']

        # Now set boundary conditions
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for i,sp in enumerate(species_names):
            if i >= self.dat.np:
                self.set_lower_bc(sp, bc_type='Moses') # gas
            else:
                self.set_lower_bc(sp, bc_type='vdep', vdep=0.0) # particle
        species_names = self.dat.species_names[self.dat.np:(-2-self.dat.nsl)]
        for i,sp in enumerate(species_names):
            self.set_lower_bc(sp, bc_type='press', press=out['P_i_surf'][i])

        self.prep_atmosphere(self.wrk.usol)

###
### Some PICASO specific methods for the class
###

    def add_concentrations_to_picaso_df(self, df):
        """Adds photochem concentrations to a PICASO "profile" DataFrame

        Parameters
        ----------
        df : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar), 
            temperature (K), and gas concentrations in volume mixing ratios.

        Returns
        -------
        DataFrame
            Pandas DataFrame with Photochem result added. Mixing ratios are normalized so that
            they sum to 1.
        """        

        # Get the photochem results
        sol = self.return_atmosphere_climate_grid()

        # Check to make sure P in df match what is in photochem
        if not np.all(np.isclose(df['pressure'].to_numpy()[::-1].copy()*1e6, self.P_clima_grid)):
            raise Exception('The pressures in `df` does not match the climate grid in photochem')

        # Add mixing ratios to df. Make sure to exclude particles.
        species_names = self.dat.species_names[self.dat.np:(-2-self.dat.nsl)]
        for key in species_names:
            if key not in ['pressure','temperature','Kzz']:
                df[key] = sol[key][::-1].copy()

        # Renormalized so that mixing ratios sum to 1
        mix_tot = np.zeros(len(df['pressure']))
        for key in df:
            if key not in ['pressure', 'temperature', 'kz']:
                mix_tot += df[key].to_numpy()
        for key in df:
            if key not in ['pressure', 'temperature', 'kz']:
                df[key] = df[key]/mix_tot

        return df

    def initialize_to_climate_equilibrium_PT_picaso(self, df, Kzz_in, metallicity, CtoO, rainout_condensed_atoms=True):
        """Wrapper to `initialize_to_climate_equilibrium_PT`, which accepts a Pandas DataFrame
        containing the input pressure (bar) and temperature (K). The order of all input arrays 
        flipped (i.e., first element is TOA) following the PICASO convention.

        Parameters
        ----------
        df : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar), 
            temperature (K). 
        Kzz_in : ndarray[dim=1,float64]
            Eddy diffusion (cm^2/s) array corresponding to each pressure level in df['pressure']
        """
        P_in = df['pressure'].to_numpy()
        T_in = df['temperature'].to_numpy()

        self.initialize_to_climate_equilibrium_PT(P_in[::-1].copy()*1e6, T_in[::-1].copy(), Kzz_in[::-1].copy(), 
                                                  metallicity, CtoO, rainout_condensed_atoms)
        
    def reinitialize_to_new_climate_PT_picaso(self, df, Kzz_in):
        """Wrapper to `reinitialize_to_new_climate_PT`, which accepts a Pandas DataFrame which contains
        `pressure` in bar, `temperature` in K, and mixing ratios. The order of input arrays are flipped 
        (i.e., first element is TOA) following the PICASO convention. 

        Parameters
        ----------
        df : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar), 
            temperature (K), and gas concentrations in volume mixing ratios. 
        Kzz_in : ndarray[dim=1,float64]
            Eddy diffusion (cm^2/s) array corresponding to each pressure level in df['pressure']
        """

        P_in = df['pressure'].to_numpy()[::-1].copy()*1e6
        T_in = df['temperature'].to_numpy()[::-1].copy()
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        mix = {}
        for key in df:
            if key in species_names:
                mix[key] = df[key].to_numpy()[::-1].copy()

        # normalize
        mix_tot = np.zeros(P_in.shape[0])
        for key in mix:
            mix_tot += mix[key]
        for key in mix:
            mix[key] = mix[key]/mix_tot

        self.reinitialize_to_new_climate_PT(P_in, T_in, Kzz_in[::-1].copy(), mix)

    def run_for_picaso(self, df, log10metallicity, CtoO, Kzz, first_run, rainout_condensed_atoms=True):
        """Runs the Photochemical model to steady-state using inputs from the PICASO climate model.

        Parameters
        ----------
        df : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar), 
            temperature (K), and gas concentrations in volume mixing ratios. The first element
            of each array is the top of the atomsphere (i.e. order is flipped).
        log10metallicity : float
            log10 metallicity relative to solar.
        CtoO : float
            The C/O ratio relative to solar.
        Kzz : ndarray[dim=1,float64]
            Eddy diffusion (cm^2/s) corresponding to each pressure in df['pressure'].
        first_run : bool
            If this is the first photochem call, then this should be True
        rainout_condensed_atoms : bool, optional
            If True and `first_run` is True, then the code rains out condensed
            atoms when guesing the initial solution, by default True.

        Returns
        -------
        DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame similar to the input, except
            steady-state photochemistry gas concentrations are loaded in.
        """        

        # Initialize Photochem to `df`
        if first_run:
            self.initialize_to_climate_equilibrium_PT_picaso(df, Kzz, 10.0**log10metallicity, CtoO, rainout_condensed_atoms)
        else:
            self.reinitialize_to_new_climate_PT_picaso(df, Kzz)
            if not np.isclose(self.metallicity, 10.0**log10metallicity) or not np.isclose(self.CtoO, CtoO):
                raise Exception('`metallicity` or `CtoO` does not match.')

        # Compute steady state 
        success = self.find_steady_state()
        assert success

        # Return a DataFrame with the Photochem chemistry
        return self.add_concentrations_to_picaso_df(df)

###
### Helper functions for the EvoAtmosphereGasGiant class
###

@nb.njit()
def CH4_CO_quench_timescale(T, P):
    "T in K, P in dynes/cm^2, tq in s. Equation 11."
    P_bars = P/1.0e6
    tq = 3.0e-6*P_bars**-1*np.exp(42_000.0/T)
    return tq

@nb.njit()
def NH3_quench_timescale(T, P):
    "T in K, P in dynes/cm^2, tq in s. Equation 32."
    P_bars = P/1.0e6
    tq = 1.0e-7*P_bars**-1*np.exp(52_000.0/T)
    return tq

@nb.njit()
def HCN_quench_timescale(T, P):
    "T in K, P in dynes/cm^2, tq in s. From PICASO."
    P_bars = P/1.0e6
    tq = (1.5e-4/(P_bars*(3.0**0.7)))*np.exp(36_000.0/T)
    return tq

@nb.njit()
def CO2_quench_timescale(T, P):
    "T in K, P in dynes/cm^2, tq in s. Equation 44."
    P_bars = P/1.0e6
    tq = 1.0e-10*P_bars**-0.5*np.exp(38_000.0/T)
    return tq

@nb.njit()
def scale_height(T, mubar, grav):
    "All inputs are CGS."
    k_boltz = const.k*1e7
    H = (const.Avogadro*k_boltz*T)/(mubar*grav)
    return H

@nb.njit()
def determine_quench_levels(T, P, Kzz, mubar, grav):

    # Mixing timescale
    tau_mix = scale_height(T, mubar, grav)**2/Kzz

    # Quenching timescales
    tau_CH4 = CH4_CO_quench_timescale(T, P)
    tau_CO2 = CO2_quench_timescale(T, P)
    tau_NH3 = NH3_quench_timescale(T, P)
    tau_HCN = HCN_quench_timescale(T, P)

    # Quench level is when the chemistry timescale
    # exceeds the mixing timescale.
    quench_levels = np.zeros(4, dtype=np.int32)
    
    for i in range(P.shape[0]):
        quench_levels[0] = i
        if tau_CH4[i] > tau_mix[i]:
            break

    for i in range(P.shape[0]):
        quench_levels[1] = i
        if tau_CO2[i] > tau_mix[i]:
            break

    for i in range(P.shape[0]):
        quench_levels[2] = i
        if tau_NH3[i] > tau_mix[i]:
            break

    for i in range(P.shape[0]):
        quench_levels[3] = i
        if tau_HCN[i] > tau_mix[i]:
            break

    return quench_levels

@nb.njit()
def deepest_quench_level(T, P, Kzz, mubar, grav):
    quench_levels = determine_quench_levels(T, P, Kzz, mubar, grav)
    return np.min(quench_levels)
    
@nb.experimental.jitclass()
class TempPressMubar:

    log10P : types.double[:]
    T : types.double[:]
    mubar : types.double[:]

    def __init__(self, P, T, mubar):
        self.log10P = np.log10(P)[::-1].copy()
        self.T = T[::-1].copy()
        self.mubar = mubar[::-1].copy()

    def temperature_mubar(self, P):
        T = np.interp(np.log10(P), self.log10P, self.T)
        mubar = np.interp(np.log10(P), self.log10P, self.mubar)
        return T, mubar

@nb.njit()
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

@nb.njit()
def hydrostatic_equation(P, u, planet_radius, planet_mass, ptm):
    z = u[0]
    grav = gravity(planet_radius, planet_mass, z)
    T, mubar = ptm.temperature_mubar(P)
    k_boltz = const.Boltzmann*1e7
    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)
    return np.array([dz_dP])

def compute_altitude_of_PT(P, P_ref, T, mubar, planet_radius, planet_mass, P_top):
    ptm = TempPressMubar(P, T, mubar)
    args = (planet_radius, planet_mass, ptm)

    if P_top < P[-1]:
        # If P_top is lower P than P grid, then we extend it
        P_top_ = P_top
        P_ = np.append(P,P_top_)
        T_ = np.append(T,T[-1])
        mubar_ = np.append(mubar,mubar[-1])
    else:
        P_top_ = P[-1]
        P_ = P.copy()
        T_ = T.copy()
        mubar_ = mubar.copy()

    # Make sure P_ref is in the P grid
    if P_ref > P_[0] or P_ref < P_[-1]:
        raise Exception('Reference pressure must be within P grid.')
    
    # Find first index with lower pressure than P_ref
    ind = 0
    for i in range(P_.shape[0]):
        if P_[i] < P_ref:
            ind = i
            break

    # Integrate from P_ref to TOA
    out2 = integrate.solve_ivp(hydrostatic_equation, [P_ref, P_[-1]], np.array([0.0]), t_eval=P_[ind:], args=args, rtol=1e-6)
    assert out2.success
    # Integrate from P_ref to BOA
    out1 = integrate.solve_ivp(hydrostatic_equation, [P_ref, P_[0]], np.array([0.0]), t_eval=P_[:ind][::-1], args=args, rtol=1e-6)
    assert out1.success

    # Stitch together
    z_ = np.append(out1.y[0][::-1],out2.y[0])

    return P_, T_, mubar_, z_

###
### A simple metallicity calculator
###

class Metallicity():

    def __init__(self, filename):
        """A simple Metallicity calculator.

        Parameters
        ----------
        filename : str
            Path to a thermodynamic file
        """
        self.gas = equilibrate.ChemEquiAnalysis(filename)

    def composition(self, T, P, CtoO, metal, rainout_condensed_atoms = True):
        """Given a T-P profile, C/O ratio and metallicity, the code
        computes chemical equilibrium composition.

        Parameters
        ----------
        T : ndarray[dim=1,float64]
            Temperature in K
        P : ndarray[dim=1,float64]
            Pressure in dynes/cm^2
        CtoO : float
            The C / O ratio relative to solar. CtoO = 1 would be the same
            composition as solar.
        metal : float
            Metallicity relative to solar.
        rainout_condensed_atoms : bool, optional
            If True, then the code will rainout atoms that condense.

        Returns
        -------
        dict
            Composition at chemical equilibrium.
        """

        # Check T and P
        if isinstance(T, float) or isinstance(T, int):
            T = np.array([T],np.float64)
        if isinstance(P, float) or isinstance(P, int):
            P = np.array([P],np.float64)
        if not isinstance(P, np.ndarray):
            raise ValueError('"P" must by an np.ndarray')
        if not isinstance(T, np.ndarray):
            raise ValueError('"P" must by an np.ndarray')
        if T.ndim != 1:
            raise ValueError('"T" must have one dimension')
        if P.ndim != 1:
            raise ValueError('"P" must have one dimension')
        if T.shape[0] != P.shape[0]:
            raise ValueError('"P" and "T" must be the same length')
        # Check CtoO and metal
        if CtoO <= 0:
            raise ValueError('"CtoO" must be greater than 0')
        if metal <= 0:
            raise ValueError('"metal" must be greater than 0')

        # For output
        out = {}
        for sp in self.gas.gas_names:
            out[sp] = np.empty(P.shape[0])
        mubar = np.empty(P.shape[0])
        
        molfracs_atoms = self.gas.molfracs_atoms_sun
        for i,sp in enumerate(self.gas.atoms_names):
            if sp != 'H' and sp != 'He':
                molfracs_atoms[i] = self.gas.molfracs_atoms_sun[i]*metal
        molfracs_atoms = molfracs_atoms/np.sum(molfracs_atoms)

        # Adjust C and O to get desired C/O ratio. CtoO is relative to solar
        indC = self.gas.atoms_names.index('C')
        indO = self.gas.atoms_names.index('O')
        x = CtoO*(molfracs_atoms[indC]/molfracs_atoms[indO])
        a = (x*molfracs_atoms[indO] - molfracs_atoms[indC])/(1+x)
        molfracs_atoms[indC] = molfracs_atoms[indC] + a
        molfracs_atoms[indO] = molfracs_atoms[indO] - a

        # Compute chemical equilibrium at all altitudes
        for i in range(P.shape[0]):
            self.gas.solve(P[i], T[i], molfracs_atoms=molfracs_atoms)
            for j,sp in enumerate(self.gas.gas_names):
                out[sp][i] = self.gas.molfracs_species_gas[j]
            mubar[i] = self.gas.mubar
            if rainout_condensed_atoms:
                molfracs_atoms = self.gas.molfracs_atoms_gas

        return out, mubar

###
### Template input files for Photochem
###

ATMOSPHERE_INIT = \
"""alt      den        temp       eddy                       
0.0      1          1000       1e6              
1.0e3    1          1000       1e6         
"""

SETTINGS_TEMPLATE = \
"""
atmosphere-grid:
  bottom: 0.0
  top: atmospherefile
  number-of-layers: NULL

photolysis-grid:
  regular-grid: true
  lower-wavelength: 92.5
  upper-wavelength: 855.0
  number-of-bins: 200

planet:
  planet-mass: NULL
  planet-radius: NULL
  surface-albedo: 0.0
  solar-zenith-angle: 60.0
  hydrogen-escape:
    type: none
  default-gas-lower-boundary: Moses
  water:
    fix-water-in-troposphere: false
    gas-rainout: false
    water-condensation: false

boundary-conditions:
- name: He
  lower-boundary: {type: Moses}
  upper-boundary: {type: veff, veff: 0}
"""

###
### A series of functions for generating reactions and thermo files.
###

def mechanism_with_atoms(dat, atoms_names):

    atoms = []
    exclude_atoms = []
    for i,at in enumerate(dat['atoms']):
        if at['name'] in atoms_names:
            atoms.append(at)
        else:
            exclude_atoms.append(at['name'])

    species = []
    comp = {}
    comp['hv'] = []
    comp['M'] = []
    for i,sp in enumerate(dat['species']):
        comp[sp['name']] = [key for key in sp['composition'] if sp['composition'][key] > 0]

        exclude = False
        for tmp in comp[sp['name']]:
            if tmp in exclude_atoms:
                exclude = True
                break
        if not exclude:
            species.append(sp)

    if "particles" in dat:
        particles = []
        for i,sp in enumerate(dat['particles']):
            comp_tmp = [key for key in sp['composition'] if sp['composition'][key] > 0]

            exclude = False
            for tmp in comp_tmp:
                if tmp in exclude_atoms:
                    exclude = True
                    break
            if not exclude:
                particles.append(sp)

    if "reactions" in dat:
        reactions = []
        for i,rx in enumerate(dat['reactions']):
            eq = rx['equation']
            eq = eq.replace('(','').replace(')','')
            if '<=>' in eq:
                split_str = '<=>'
            else:
                split_str = '=>'
    
            a,b = eq.split(split_str)
            a = a.split('+')
            b = b.split('+')
            a = [a1.strip() for a1 in a]
            b = [b1.strip() for b1 in b]
            sp = a + b
    
            exclude = False
            for s in sp:
                for tmp in comp[s]:
                    if tmp in exclude_atoms:
                        exclude = True
                        break
                if exclude:
                    break
            if not exclude:
                reactions.append(rx)
                
    out = dat
    out['atoms'] = atoms
    out['species'] = species
    if 'particles' in dat:
        out['particles'] = particles
    if 'reactions' in dat:
        out['reactions'] = reactions

    return out

def remove_reaction_particles(dat):
    if "particles" in dat:
        particles = []
        for i, particle in enumerate(dat['particles']):
            if particle['formation'] != "reaction":
                particles.append(particle)
        dat['particles'] = particles

    return dat

def generate_zahnle_reaction_thermo_file(atoms_names):

    if 'H' not in atoms_names or 'He' not in atoms_names:
        raise Exception('H and He must be in atoms_names')

    with open(zahnle_earth,'r') as f:
        rxns = yaml.load(f,Loader=yaml.Loader)
    # Make a deep copy for later
    rxns_copy = copy.deepcopy(rxns)

    out_rxns = mechanism_with_atoms(rxns, atoms_names)
    out_rxns = remove_reaction_particles(out_rxns)

    with open(zahnle_earth.replace('zahnle_earth.yaml','condensate_thermo.yaml'),'r') as f:
        thermo = yaml.load(f, Loader=yaml.Loader)

    # Delete information that is not needed
    for i,atom in enumerate(rxns_copy['atoms']):
        del rxns_copy['atoms'][i]['redox'] 
    if 'particles' in rxns_copy:
        del rxns_copy['particles']
    del rxns_copy['reactions']

    # Add condensates
    for i,sp in enumerate(thermo['species']):
        rxns_copy['species'].append(sp)

    out_thermo = mechanism_with_atoms(rxns_copy, atoms_names)

    return out_rxns, out_thermo

def generate_photochem_rx_and_thermo_files(atoms_names=['H','He','N','O','C','S'], 
                                           rxns_filename='photochem_rxns.yaml', thermo_filename='photochem_thermo.yaml'):
    """Generates input reactions and thermodynamic files for photochem.

    Parameters
    ----------
    atoms_names : list, optional
        Atoms to include in the thermodynamics, by default ['H','He','N','O','C','S']
    rxns_filename : str, optional
        Name of output reactions file, by default 'photochem_rxns.yaml'
    thermo_filename : str, optional
        Name of output thermodynamic file, by default 'photochem_thermo.yaml'
    """    
    rxns, thermo = generate_zahnle_reaction_thermo_file(atoms_names)
    rxns = FormatReactions_main(rxns)
    with open(rxns_filename,'w') as f:
        yaml.dump(rxns,f,Dumper=MyDumper,sort_keys=False,width=70)
    thermo = FormatReactions_main(thermo)
    with open(thermo_filename,'w') as f:
        yaml.dump(thermo,f,Dumper=MyDumper,sort_keys=False,width=70)

def set_equilibrium_composition_to_picaso_df(pc, mechanism_file, df, target_temp_abund = 1200):
    """Sets elemetal composition in photochem equilibrium solver to the bottom of the atmosphere
    composition in a picaso df.

    Parameters
    ----------
    pc : EvoAtomsphereGasGiant
        Photochem object
    mechanism_file : str
        Path to the reactions file
    df : DataFrame
        Pandas DataFrame describing the atmosphere
    """    
    
    # Read the mechanism file
    with open(mechanism_file,'r') as f:
        data = yaml.load(f,Loader=yaml.Loader)
    species_composition = {}
    for i,sp in enumerate(data['species']):
        species_composition[sp['name']] = sp['composition']
    try:
        for i,sp in enumerate(data['particles']):
            species_composition[sp['name']] = sp['composition']
    except KeyError:
        pass
    
    # Build a composition dictionary
    comp = {}
    for i,atom in enumerate(data['atoms']):
        comp[atom['name']] = 0.0

    #find the closest layer to 1200 K to grab abundances from
    try:
        closest_layer_idx = np.argmin(np.abs(df['temperature'].values - target_temp_abund))
        # Compute the composition of the deepest layer in PICASO df
        for i,sp in enumerate(df):
            if sp in species_composition:
                for atom in species_composition[sp]:
                    comp[atom] += species_composition[sp][atom]*df[sp].to_numpy()[closest_layer_idx]

    except AttributeError:
        closest_layer_idx = np.argmin(np.abs(df['temperature'] - target_temp_abund))
        # Compute the composition of the deepest layer in PICASO df
        for i,sp in enumerate(df['ptchem_df']):
            if sp in species_composition:
                for atom in species_composition[sp]:
                    comp[atom] += species_composition[sp][atom]*df['ptchem_df'][sp].to_numpy()[closest_layer_idx]
    
    # Renormalize the composition
    tot = 0.0
    for key in comp:
        tot += comp[key]
    for key in comp:
        comp[key] = comp[key]/tot
    
    # Convert composition to array
    molfracs_atoms = np.empty(len(pc.m.gas.atoms_names))
    for i,atom in enumerate(pc.m.gas.atoms_names):
        molfracs_atoms[i] = comp[atom]

    # Set composition in equilibrium solver
    pc.m.gas.molfracs_atoms_sun = molfracs_atoms