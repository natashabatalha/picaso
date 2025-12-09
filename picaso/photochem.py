"""
This module contains wrappers for the "Photochem" photochemical
model (https://github.com/Nicholaswogan/photochem). These wrappers are 
called in "justdoit.py" during climate simulations if photochemistry
is turned on.
"""

import numpy as np
import warnings
from photochem.extensions import EvoAtmosphereGasGiant
from photochem.utils import zahnle_rx_and_thermo_files
import pickle
import os

# Turn off Panda's performance warnings
import pandas as pd
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    
###
### Extension of EvoAtmosphereGasGiant for Picaso
###

class EvoAtmosphereGasGiantPicaso(EvoAtmosphereGasGiant):
    """PICASO-friendly wrapper around `EvoAtmosphereGasGiant` that swaps array
    orientations, normalizes outputs, and optionally stores intermediate model
    states during coupled climate/photochemistry runs."""

    def __init__(self, *args, **kwargs):
        """Initialize the photochemistry model using the base class settings.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to `photochem.extensions.EvoAtmosphereGasGiant`.
        """
        super().__init__(*args, **kwargs)

        # Path to a pickle file that will save atmospheric
        # states during a couple climate/photochemistry simulation.
        self.save_file = None

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
        if not np.all(np.isclose(df['pressure'].to_numpy()[::-1].copy()*1e6, self.gdat.P_clima_grid)):
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
        
    def reinitialize_to_new_climate_PT_picaso(self, df_temp, df_comp_guess, Kzz_in):
        """Wrapper to `reinitialize_to_new_climate_PT`, which accepts a Pandas DataFrame which contains
        `pressure` in bar, `temperature` in K, and mixing ratios. The order of input arrays are flipped 
        (i.e., first element is TOA) following the PICASO convention. 

        Parameters
        ----------
        df_temp : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing the current 
            pressure (bar) and temperature (K) to intialize to.
        df_comp_guess : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar) and 
            gas concentrations in volume mixing ratios to initalize to.
        Kzz_in : ndarray[dim=1,float64]
            Eddy diffusion (cm^2/s) array corresponding to each pressure level in df['pressure']
        """

        if not np.allclose(df_temp['pressure'].to_numpy(), df_comp_guess['pressure'].to_numpy()):
            raise Exception('`df` and `df_comp_guess` should have the same pressure levels.')

        P_in = df_temp['pressure'].to_numpy()[::-1].copy()*1e6
        T_in = df_temp['temperature'].to_numpy()[::-1].copy()
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        mix = {}
        for key in df_comp_guess:
            if key in species_names:
                mix[key] = df_comp_guess[key].to_numpy()[::-1].copy()

        # normalize
        mix_tot = np.zeros(P_in.shape[0])
        for key in mix:
            mix_tot += mix[key]
        for key in mix:
            mix[key] = mix[key]/mix_tot

        self.reinitialize_to_new_climate_PT(P_in, T_in, Kzz_in[::-1].copy(), mix)

    def run_for_picaso(self, df, log10metallicity, CtoO, Kzz, df_comp_guess=None, rainout_condensed_atoms=True):
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
        df_comp_guess : DataFrame
            DataFrame containing the photochem composition to initialize from.
        rainout_condensed_atoms : bool, optional
            If True and `df_comp_guess` is not None, then the code rains out condensed
            atoms when guesing the initial solution, by default True.

        Returns
        -------
        DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame similar to the input, except
            steady-state photochemistry gas concentrations are loaded in.
        """        

        # Initialize Photochem to `df`
        if df_comp_guess is None:
            self.initialize_to_climate_equilibrium_PT_picaso(df, Kzz, 10.0**log10metallicity, CtoO, rainout_condensed_atoms)
        else:
            self.reinitialize_to_new_climate_PT_picaso(df, df_comp_guess, Kzz)
            if not np.isclose(self.gdat.metallicity, 10.0**log10metallicity) or not np.isclose(self.gdat.CtoO, CtoO):
                raise Exception('`metallicity` or `CtoO` does not match.')

        # Compute steady state 
        success = self.find_steady_state()
        assert success

        # Save the output to a pickle file, if specified
        if self.save_file is not None:
            sol = self.return_atmosphere_climate_grid()
            model = self.model_state_to_dict()
            if not os.path.isfile(self.save_file):
                with open(self.save_file, 'wb') as f:
                    pass
            with open(self.save_file,'ab') as f:
                pickle.dump((sol,model,),f)

        # Return a DataFrame with the Photochem chemistry
        return self.add_concentrations_to_picaso_df(df)
    
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
    zahnle_rx_and_thermo_files(
        atoms_names=atoms_names,
        rxns_filename=rxns_filename,
        thermo_filename=thermo_filename,
        exclude_species=[],
        remove_particles=False,
        remove_reaction_particles=True
    )

# ~~~ Equilibrium Chemistry ~~~

from photochem.equilibrate import ChemEquiAnalysis
from tempfile import NamedTemporaryFile

class EquilibriumChemistry(ChemEquiAnalysis):
    """Compute equilibrium gas and condensate abundances for a given P–T
    profile, optionally with custom elemental ratios."""

    def __init__(self, thermofile=None, atoms=None, species=None, atom_to_H_ratios=None):
        """Create an equilibrium chemistry solver.

        Parameters
        ----------
        thermofile : str, optional
            Path to a YAML thermodynamics file. If None, uses the bundled
            `THERMODYNAMICS` string.
        atoms : list[str], optional
            Subset of atom symbols to include; defaults to all in the thermo
            file.
        species : list[str], optional
            Species names to load; defaults to the bundled list when
            `thermofile` is None.
        atom_to_H_ratios : dict, optional
            Mapping of element symbol to X/H ratio to override the solar
            composition before solving.
        """

        # Thermodynamics
        if thermofile is None:
            thermostr = THERMODYNAMICS
        else:
            with open(thermofile, 'r') as f:
                thermostr = f.read()

        # Species for THERMODYNAMICS
        if species is None and thermofile is None:
            species = [
                'H', 'H2', 'He', 'O', 'C', 'N', 'Mg', 'Si', 
                'Fe', 'S', 'AL', 'Ca', 'Na', 'Ni', 'P', 'K', 
                'Ti', 'CO', 'OH', 'SH', 'N2', 'O2', 'SiO', 
                'TiO', 'SiS', 'H2O', 'C2', 'CH', 'CN', 'CS', 
                'SiC', 'NH', 'SiH', 'NO', 'SN', 'SiN', 'SO', 
                'S2', 'C2H', 'HCN', 'C2H2', 'CH4', 
                'ALH', 'ALOH', 'AL2O', 'CaOH', 'MgH', 'MgOH', 
                'PH3', 'CO2', 'TiO2', 'Si2C', 'SiO2', 'FeO', 
                'NH2', 'NH3', 'CH2', 'CH3', 'H2S', 'VO', 'VO2', 
                'NaCL', 'KCL', 'e-', 'H+', 'H-', 'Na+', 'K+', 
                'PH2', 'P2', 'PS', 'PO', 'P4O6', 'PH', 'V', 
                'VO(c)', 'VO(L)', 'MgSiO3(c)', 'SiC(c)', 'Fe(c)', 
                'AL2O3(c)', 'Na2S(c)', 'KCL(c)', 'Fe(L)', 
                'SiC(L)', 'MgSiO3(L)', 'H2O(L)', 'H2O(c)', 
                'TiO(c)', 'TiO(L)', 'TiO2(c)', 'TiO2(L)', 
                'H3PO4(c)', 'H3PO4(L)'
            ]
            
        # Initialize
        with NamedTemporaryFile('w',suffix='.yaml') as f:
            f.write(thermostr)
            f.flush()
            super().__init__(
                f.name,
                atoms,
                species 
            )

        self._molfracs_atoms_sun_save = self.molfracs_atoms_sun.copy()
        if atom_to_H_ratios is not None:
            self.set_atom_to_H_ratios(atom_to_H_ratios)

    def set_atom_to_H_ratios(self, atom_to_H_ratios, reset=False):
        """Override or reset the elemental X/H ratios used by the solver.

        Parameters
        ----------
        atom_to_H_ratios : dict
            Mapping of atom symbol to desired abundance relative to hydrogen.
        reset : bool, optional
            If True, ignore overrides and restore the saved solar composition.
        """

        if reset:
            self.molfracs_atoms_sun = self._molfracs_atoms_sun_save.copy()
            return
            
        # Get X/H ratios of the Sun
        ind_H = self.atoms_names.index('H')
        comp_sun = {'H': 1.0}
        for i,atom in enumerate(self.atoms_names):
            if atom != 'H':
                comp_sun[atom] = self._molfracs_atoms_sun_save[i]/self._molfracs_atoms_sun_save[ind_H]

        # Replace comp_sun with values in atom_to_H_ratios, if they exist
        for key in atom_to_H_ratios:
            if key in comp_sun:
                comp_sun[key] = atom_to_H_ratios[key]
        
        # Build molfracs_atoms_sun
        molfracs_atoms_sun = self._molfracs_atoms_sun_save.copy()
        for i,atom in enumerate(self.atoms_names):
            molfracs_atoms_sun[i] = comp_sun[atom]

        # Normalize
        molfracs_atoms_sun /= np.sum(molfracs_atoms_sun)

        # Set the composition
        self.molfracs_atoms_sun = molfracs_atoms_sun

    def equilibrate_atmosphere(self, P, T, log10mh, CtoO_absolute):
        """Solve for equilibrium chemistry across a vertical profile.

        Parameters
        ----------
        P : ndarray
            Pressures in bar.
        T : ndarray
            Temperatures in Kelvin aligned with `P`.
        log10mh : float
            log10 metallicity relative to solar.
        CtoO_absolute : float
            Absolute C/O ratio to target.

        Returns
        -------
        tuple(dict, dict)
            Gas and condensate mole fraction dictionaries keyed by species
            name.
        """

        # Check inputs
        if not isinstance(P, np.ndarray):
            raise ValueError('`P` should be a numpy array.')
        if not isinstance(P, np.ndarray):
            raise ValueError('`T` should be a numpy array.')

        # Some conversions
        P_cgs = P*1e6
        metallicity = 10.0**log10mh
        indC = self.atoms_names.index('C')
        indO = self.atoms_names.index('O')
        CtoO_solar = self.molfracs_atoms_sun[indC]/self.molfracs_atoms_sun[indO]
        CtoO_relative = CtoO_absolute/CtoO_solar

        gas_names = self.gas_names
        condensate_names = self.condensate_names

        gases = {}
        for key in gas_names:
            gases[key] = np.empty(len(P))
        condensates = {}
        for key in condensate_names:
            condensates[key] = np.empty(len(P))

        for i in range(len(P)):
            converged = self.solve_metallicity(P_cgs[i], T[i], metallicity, CtoO_relative)
            if not converged:
                raise Exception('The equilibrium chemistry solver failed to converge.')
            molfracs_species_gas = self.molfracs_species_gas
            molfracs_species_condensate = self.molfracs_species_condensate
            for j,key in enumerate(gas_names):
                gases[key][i] = molfracs_species_gas[j]
            for j,key in enumerate(condensate_names):
                condensates[key][i] = molfracs_species_condensate[j]

        return gases, condensates
    
# These thermodynamics are from the easyCHEM library.
THERMODYNAMICS = \
"""
atoms:
- {name: Si, mass: 28.086}
- {name: Ni, mass: 58.693}
- {name: H, mass: 1.00797}
- {name: Al, mass: 26.982}
- {name: P, mass: 30.974}
- {name: S, mass: 32.06}
- {name: Ca, mass: 40.078}
- {name: Ti, mass: 47.867}
- {name: Mg, mass: 24.305}
- {name: E, mass: 0.000548579909}
- {name: O, mass: 15.9994}
- {name: N, mass: 14.0067}
- {name: Na, mass: 22.99}
- {name: He, mass: 4.002602}
- {name: C, mass: 12.011}
- {name: V, mass: 50.942}
- {name: Fe, mass: 55.845}
- {name: Cl, mass: 35.453}
- {name: K, mass: 39.098}

species:
- name: H
  composition: {H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 25473.708, -0.44668285]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 25473.70801, -0.446682853]
    - [60.7877425, -0.1819354417, 2.500211817, -1.226512864e-07, 3.73287633e-11, -5.68774456e-15,
      3.410210197e-19, 25474.86398, -0.448191777]
    - [217375769.4, -131203.5403, 33.991742, -0.00381399968, 2.432854837e-07, -7.69427554e-12,
      9.64410563e-17, 1067638.086, -274.2301051]
- name: H2
  composition: {H: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [13461.248, -617.64441, 14.330385, -0.092796176, 0.00038644403, -7.5794807e-07,
      5.6379947e-10, 1233.7399, -51.825747]
    - [40783.2321, -800.918604, 8.21470201, -0.01269714457, 1.753605076e-05, -1.20286027e-08,
      3.36809349e-12, 2682.484665, -30.43788844]
    - [560812.801, -837.150474, 2.975364532, 0.001252249124, -3.74071619e-07, 5.9366252e-11,
      -3.6069941e-15, 5339.82441, -2.202774769]
    - [496688412.0, -314754.7149, 79.8412188, -0.00841478921, 4.75324835e-07, -1.371873492e-11,
      1.605461756e-16, 2488433.516, -669.572811]
- name: He
  composition: {He: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 0.92872397]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 0.928723974]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 0.928723974]
    - [3396845.42, -2194.037652, 3.080231878, -8.06895755e-05, 6.25278491e-09, -2.574990067e-13,
      4.429960218e-18, 16505.1896, -4.04881439]
- name: O
  composition: {O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-2653.0464, 136.94021, 0.49434809, 0.019900289, -9.1598216e-05, 1.9412796e-07,
      -1.5496298e-10, 28574.37, 13.742886]
    - [-7953.6113, 160.7177787, 1.966226438, 0.00101367031, -1.110415423e-06, 6.5175075e-10,
      -1.584779251e-13, 28403.62437, 8.40424182]
    - [261902.0262, -729.872203, 3.31717727, -0.000428133436, 1.036104594e-07, -9.43830433e-12,
      2.725038297e-16, 33924.2806, -0.667958535]
    - [177900426.4, -108232.8257, 28.10778365, -0.002975232262, 1.854997534e-07, -5.79623154e-12,
      7.191720164e-17, 889094.263, -218.1728151]
- name: C
  composition: {C: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [315.08059, -8.9488244, 2.7359074, -0.0014819262, 2.8944865e-06, 9.8709379e-10,
      -5.72872e-12, 85335.377, 3.7162639]
    - [649.503147, -0.964901086, 2.504675479, -1.281448025e-05, 1.980133654e-08, -1.606144025e-11,
      5.314483411e-15, 85457.6311, 4.747924288]
    - [-128913.6472, 171.9528572, 2.646044387, -0.000335306895, 1.74209274e-07, -2.902817829e-11,
      1.642182385e-15, 84105.9785, 4.130047418]
    - [443252801.0, -288601.8412, 77.3710832, -0.00971528189, 6.64959533e-07, -2.230078776e-11,
      2.899388702e-16, 2355273.444, -640.512316]
- name: N
  composition: {N: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 56104.638, 4.193905]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 56104.6378, 4.193905036]
    - [88765.0138, -107.12315, 2.362188287, 0.0002916720081, -1.7295151e-07, 4.01265788e-11,
      -2.677227571e-15, 56973.5133, 4.865231506]
    - [547518105.0, -310757.498, 69.1678274, -0.00684798813, 3.8275724e-07, -1.098367709e-11,
      1.277986024e-16, 2550585.618, -584.8769753]
- name: Mg
  composition: {Mg: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 16946.588, 3.6343301]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 16946.58761, 3.63433014]
    - [-536483.155, 1973.709576, -0.36337769, 0.002071795561, -7.73805172e-07, 1.359277788e-10,
      -7.766898397e-15, 4829.18811, 23.39104998]
    - [2166012586.0, -1008355.665, 161.9680021, -0.00879013035, -1.925690961e-08,
      1.725045214e-11, -4.234946112e-16, 8349525.9, -1469.355261]
- name: Si
  composition: {Si: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [4701.9718, -173.53795, 6.6048656, -0.025337932, 6.3918158e-05, -5.7342917e-08,
      -1.1204641e-12, 53749.172, -12.742382]
    - [98.3614081, 154.6544523, 1.87643667, 0.001320637995, -1.529720059e-06, 8.95056277e-10,
      -1.95287349e-13, 52635.1031, 9.69828888]
    - [-616929.885, 2240.683927, -0.444861932, 0.001710056321, -4.10771416e-07, 4.55888478e-11,
      -1.889515353e-15, 39535.5876, 26.79668061]
    - [-928654894.0, 544398.989, -120.6739736, 0.01359662698, -7.60649866e-07, 2.149746065e-11,
      -2.474116774e-16, -4293792.12, 1086.382839]
- name: Fe
  composition: {Fe: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-245.79327, -2.9155868, 2.5341912, -0.0026519931, 4.978765e-05, -1.6421547e-07,
      1.622728e-10, 49120.374, 6.976637]
    - [67908.2266, -1197.218407, 9.84339331, -0.01652324828, 1.917939959e-05, -1.149825371e-08,
      2.832773807e-12, 54669.9594, -33.8394626]
    - [-1954923.682, 6737.1611, -5.48641097, 0.00437880345, -1.116286672e-06, 1.544348856e-10,
      -8.023578182e-15, 7137.37006, 65.0497986]
    - [1216352511.0, -582856.393, 97.8963451, -0.00537070443, 3.19203792e-08, 6.26767143e-12,
      -1.480574914e-16, 4847648.29, -869.728977]
- name: S
  composition: {S: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-1752.5904, 68.603477, 1.2599651, 0.0091793411, -1.0141575e-05, -3.7096911e-08,
      6.658775e-11, 32205.493, 11.141912]
    - [-317.484182, -192.4704923, 4.68682593, -0.0058413656, 7.53853352e-06, -4.86358604e-09,
      1.256976992e-12, 33235.9218, -5.718523969]
    - [-485424.479, 1438.830408, 1.258504116, 0.000379799043, 1.630685864e-09, -9.54709585e-12,
      8.041466646e-16, 23349.9527, 15.59554855]
    - [-130200541.4, 69093.6202, -11.76228025, 0.00160154085, -1.05053334e-07, 4.34182902e-12,
      -7.675621927e-17, -526148.503, 132.2195251]
- name: AL
  composition: {Al: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [3138.218, -109.36351, 5.2232862, -0.01845106, 5.0081829e-05, -4.8203535e-08,
      6.3121244e-13, 39187.015, -6.6035623]
    - [5006.60889, 18.61304407, 2.412531111, 0.0001987604647, -2.432362152e-07, 1.538281506e-10,
      -3.944375734e-14, 38874.1268, 6.086585765]
    - [-29208.20938, 116.7751876, 2.356906505, 7.73723152e-05, -1.529455262e-08, -9.97167026e-13,
      5.053278264e-16, 38232.8865, 6.600920155]
    - [-504068232.0, 380232.265, -108.2347159, 0.01549444292, -1.070103856e-06, 3.5921109e-11,
      -4.696039394e-16, -2901050.501, 949.188316]
- name: Ca
  composition: {Ca: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 20638.928, 4.3845483]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 20638.92786, 4.38454833]
    - [7547341.24, -21486.42662, 25.30849567, -0.01103773705, 2.293249636e-06, -1.209075383e-10,
      -4.015333268e-15, 158586.2323, -160.9512955]
    - [2291781634.0, -1608862.96, 431.246636, -0.0539650899, 3.53185621e-06, -1.16440385e-10,
      1.527134223e-15, 12586514.34, -3692.10161]
- name: Na
  composition: {Na: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 12183.829, 4.2440282]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 12183.82949, 4.24402818]
    - [952572.338, -2623.807254, 5.16259662, -0.001210218586, 2.306301844e-07, -1.249597843e-11,
      7.22677119e-16, 29129.63564, -15.19717061]
    - [1592533392.0, -971783.666, 223.8443963, -0.02380930558, 1.352018117e-06, -3.93697111e-11,
      4.630689121e-16, 7748677.26, -1939.615505]
- name: Ni
  composition: {Ni: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-1737.0427, 86.397939, 1.1738848, 0.015441733, -7.3662609e-05, 1.6584631e-07,
      -1.3732604e-10, 50567.385, 12.968832]
    - [-32358.1055, 601.526462, -1.079270657, 0.01089505519, -1.369578748e-05, 8.31772579e-09,
      -2.019206968e-12, 48138.1081, 27.188292]
    - [-493826.221, 1092.909991, 2.410485014, -1.599071827e-05, -1.047414069e-08,
      4.62479521e-12, -4.448865218e-17, 43360.7217, 9.6771956]
    - [349266988.0, -165422.7575, 33.4986936, -0.0035270859, 3.24006024e-07, -1.604177606e-11,
      2.935430214e-16, 1409017.848, -267.2455567]
- name: P
  composition: {P: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [61.101776, -0.55899283, 2.4761772, 0.00034584974, -1.7750417e-06, 3.9868691e-09,
      -3.3103101e-12, 37305.828, 5.4771031]
    - [50.4086657, -0.763941865, 2.504563992, -1.381689958e-05, 2.245585515e-08, -1.866399889e-11,
      6.227063395e-15, 37324.2191, 5.359303481]
    - [1261794.642, -4559.83819, 8.91807931, -0.00438140146, 1.454286224e-06, -2.030782763e-10,
      1.021022887e-14, 65417.2396, -39.15974795]
    - [-22153925.45, -45669.1118, 28.37245428, -0.00448324404, 3.57941308e-07, -1.255311557e-11,
      1.590290483e-16, 337090.576, -205.6960928]
- name: K
  composition: {K: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [9.6651439, -0.14580595, 2.5008659, -2.6012193e-06, 4.1873066e-09, -3.4397221e-12,
      1.131569e-15, 9959.4935, 5.0358223]
    - [9.66514393, -0.1458059455, 2.500865861, -2.601219276e-06, 4.18730658e-09, -3.43972211e-12,
      1.131569009e-15, 9959.49349, 5.03582226]
    - [-3566422.36, 10852.89825, -10.54134898, 0.00800980135, -2.696681041e-06, 4.71529415e-10,
      -2.97689735e-14, -58753.3701, 97.3855124]
    - [920578659.0, -693530.028, 191.1270788, -0.02305931672, 1.430294866e-06, -4.40933502e-11,
      5.366769166e-16, 5395082.19, -1622.158805]
- name: Ti
  composition: {Ti: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-8270.0205, 386.76597, -3.5971309, 0.057867893, -0.00024343813, 4.6781171e-07,
      -3.388326e-10, 54576.92, 33.526802]
    - [-45701.794, 660.809202, 0.429525749, 0.00361502991, -3.54979281e-06, 1.759952494e-09,
      -3.052720871e-13, 52709.4793, 20.26149738]
    - [-170478.6714, 1073.852803, 1.181955014, 0.0002245246352, 3.091697848e-07, -5.74002728e-11,
      2.927371014e-15, 49780.6991, 17.40431368]
    - [1152797766.0, -722240.838, 177.7167465, -0.02008059096, 1.221052354e-06, -3.81145208e-11,
      4.798092423e-16, 5772614.54, -1518.080466]
- name: CO
  composition: {C: 1, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-259.90541, 17.097647, 3.1206164, 0.0036221034, -1.6656393e-05, 3.611594e-08,
      -2.8412631e-11, -14473.531, 5.4566327]
    - [14890.45326, -292.2285939, 5.72452717, -0.00817623503, 1.456903469e-05, -1.087746302e-08,
      3.027941827e-12, -13031.31878, -7.85924135]
    - [461919.725, -1944.704863, 5.91671418, -0.000566428283, 1.39881454e-07, -1.787680361e-11,
      9.62093557e-16, -2466.261084, -13.87413108]
    - [886866296.0, -750037.784, 249.5474979, -0.039563511, 3.29777208e-06, -1.318409933e-10,
      1.998937948e-15, 5701421.13, -2060.704786]
- name: OH
  composition: {O: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-1998.859, 93.001362, 3.0508542, 0.0015295293, -3.157891e-06, 3.3154462e-09,
      -1.1387627e-12, 2991.2142, 4.6741108]
    - [-1998.85899, 93.0013616, 3.050854229, 0.001529529288, -3.157890998e-06, 3.31544618e-09,
      -1.138762683e-12, 2991.214235, 4.67411079]
    - [1017393.379, -2509.957276, 5.11654786, 0.000130529993, -8.28432226e-08, 2.006475941e-11,
      -1.556993656e-15, 20196.40206, -11.01282337]
    - [284723419.3, -185953.2612, 50.082409, -0.00514237498, 2.875536589e-07, -8.22881796e-12,
      9.56722902e-17, 1468393.908, -402.355558]
- name: SH
  composition: {S: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [6389.4347, -374.79609, 7.5481458, -0.012888755, 1.9077863e-05, -1.2650337e-08,
      3.2351587e-12, 17429.024, -17.607618]
    - [6389.43468, -374.796092, 7.54814577, -0.01288875477, 1.907786343e-05, -1.265033728e-08,
      3.23515869e-12, 17429.02395, -17.60761843]
    - [1682631.601, -5177.15221, 9.19816852, -0.002323550224, 6.54391478e-07, -8.46847042e-11,
      3.86474155e-15, 48992.1449, -37.70400275]
- name: N2
  composition: {N: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [328.86255, -6.4111658, 3.4936394, 0.00049317578, -3.3711214e-06, 9.1658979e-09,
      -8.1000544e-12, -1010.424, 3.0576042]
    - [22103.71497, -381.846182, 6.08273836, -0.00853091441, 1.384646189e-05, -9.62579362e-09,
      2.519705809e-12, 710.846086, -10.76003744]
    - [587712.406, -2239.249073, 6.06694922, -0.00061396855, 1.491806679e-07, -1.923105485e-11,
      1.061954386e-15, 12832.10415, -15.86640027]
    - [831013916.0, -642073.354, 202.0264635, -0.03065092046, 2.486903333e-06, -9.70595411e-11,
      1.437538881e-15, 4938707.04, -1672.09974]
- name: O2
  composition: {O: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-836.51231, 46.529355, 2.5447984, 0.0092075237, -4.4729157e-05, 1.0426252e-07,
      -8.7332889e-11, -1205.5462, 8.8388289]
    - [-34255.6342, 484.700097, 1.119010961, 0.00429388924, -6.83630052e-07, -2.0233727e-09,
      1.039040018e-12, -3391.45487, 18.4969947]
    - [-1037939.022, 2344.830282, 1.819732036, 0.001267847582, -2.188067988e-07, 2.053719572e-11,
      -8.19346705e-16, -16890.10929, 17.38716506]
    - [497529430.0, -286610.6874, 66.9035225, -0.00616995902, 3.016396027e-07, -7.4214166e-12,
      7.27817577e-17, 2293554.027, -553.062161]
- name: SiO
  composition: {Si: 1, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [483.95177, -11.817714, 3.5032674, 0.0018156349, -1.7373891e-05, 6.1320493e-08,
      -6.2686183e-11, -13067.42, 5.2970823]
    - [-47227.7105, 806.313764, -1.636976133, 0.01454275546, -1.723202046e-05, 1.04239734e-08,
      -2.559365273e-12, -16665.85903, 33.557957]
    - [-176513.4162, -31.9917709, 4.47744193, 4.59176471e-06, 3.55814315e-08, -1.327012559e-11,
      1.613253297e-15, -13508.4236, -0.838695733]
- name: TiO
  composition: {Ti: 1, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 220.0, 1000.0, 6000.0, 20000.0]
    data:
    - [4575.7642, -169.59342, 6.9140811, -0.020414362, 4.6409322e-05, -1.7758519e-09,
      -6.1842245e-11, 6032.5467, -7.6991426]
    - [-11681.5246, 454.256565, -0.1139144613, 0.01275432333, -1.727656935e-05, 1.187369403e-08,
      -3.23657937e-12, 2924.306353, 27.02903947]
    - [2330644.03, -7415.79386, 12.81799311, -0.00434455595, 1.186303111e-06, -1.367644275e-10,
      5.70321225e-15, 51448.4136, -57.9399424]
    - [166028814.7, -105185.3502, 27.49141313, -0.001681501753, 4.88407837e-08, -4.72138975e-13,
      -2.405919722e-18, 839915.607, -203.0813444]
- name: SiS
  composition: {Si: 1, S: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [5070.2331, -236.38843, 7.4211234, -0.031539046, 0.00013142689, -2.4078104e-07,
      1.6468862e-10, 12546.898, -10.746348]
    - [35994.4929, -423.97233, 4.65401442, 0.001588470782, -3.31025436e-06, 2.706096479e-09,
      -8.11351782e-13, 14115.15571, -1.183201858]
    - [-2102323.897, 6228.83618, -3.004120882, 0.00449549993, -1.368821364e-06, 1.998097253e-10,
      -9.8820358e-15, -27955.38166, 54.05828786]
- name: H2O
  composition: {H: 2, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-355.76273, 25.918122, 3.3974525, 0.006009717, -2.8930102e-05, 6.6918567e-08,
      -5.452159e-11, -30292.927, 2.4644947]
    - [-39479.6083, 575.573102, 0.931782653, 0.00722271286, -7.34255737e-06, 4.95504349e-09,
      -1.336933246e-12, -33039.7431, 17.24205775]
    - [1034972.096, -2412.698562, 4.64611078, 0.002291998307, -6.83683048e-07, 9.42646893e-11,
      -4.82238053e-15, -13842.86509, -7.97814851]
- name: C2
  composition: {C: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 280.0, 1000.0, 6000.0, 20000.0]
    data:
    - [11685.429, -645.41207, 15.256326, -0.11724298, 0.00069375124, -1.8086129e-06,
      1.6698055e-09, 101535.78, -48.200202]
    - [555963.451, -9980.12644, 66.8162037, -0.1743432724, 0.0002448523051, -1.70346758e-07,
      4.68452773e-11, 144586.9634, -344.82297]
    - [-968926.793, 3561.09299, -0.506413893, 0.002945154879, -7.13944119e-07, 8.67065725e-11,
      -4.07690681e-15, 76817.9683, 33.3998524]
    - [6315145.92, 13654.20661, -3.99690367, 0.001937561376, -1.58444658e-07, 5.52086166e-12,
      -7.25373534e-17, 9387.02499, 66.1432992]
- name: CH
  composition: {C: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [424.40853, -12.865772, 3.6619433, -0.00078967114, 1.6478561e-06, -9.2604227e-10,
      -2.4917558e-13, 70371.076, 1.3106254]
    - [22205.90133, -340.541153, 5.53145229, -0.00579496426, 7.96955488e-06, -4.46591159e-09,
      9.59633832e-13, 72407.8327, -9.10767305]
    - [2060763.44, -5396.20666, 7.85629385, -0.000796590745, 1.764308305e-07, -1.976386267e-11,
      5.03042951e-16, 106223.6592, -31.54757439]
    - [-806836869.0, 457545.054, -98.4397508, 0.01235244098, -8.48560857e-07, 3.040410624e-11,
      -4.40031517e-16, -3595851.59, 895.347744]
- name: CN
  composition: {C: 1, N: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-426.03422, 23.186614, 3.035168, 0.0042843308, -1.9595042e-05, 4.2680883e-08,
      -3.3707156e-11, 51126.124, 6.4189836]
    - [3949.14857, -139.1590572, 4.93083532, -0.00630467051, 1.256836472e-05, -9.8783005e-09,
      2.843137221e-12, 52284.5538, -2.763115585]
    - [-2228006.27, 5040.73339, -0.2121897722, 0.001354901134, 1.325929798e-07, -6.93700637e-11,
      5.49495227e-15, 17844.96132, 32.82563919]
    - [-179479811.8, 105434.6069, -17.2962417, 0.00219489553, -8.50893803e-08, 9.31869299e-13,
      6.35813993e-18, -796259.412, 191.3139639]
- name: CS
  composition: {C: 1, S: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [278.56408, -4.4896834, 3.4061721, 0.0024418992, -1.9142905e-05, 6.2124292e-08,
      -6.1080084e-11, 32626.894, 5.600394]
    - [-49248.4412, 816.69681, -1.542998408, 0.01380324735, -1.574407905e-05, 9.16971493e-09,
      -2.169700595e-12, 28651.82876, 33.08541327]
    - [-971957.476, 2339.201284, 1.709390402, 0.001577178949, -4.14633591e-07, 4.50475708e-11,
      -5.94545773e-16, 16810.20727, 18.7404822]
- name: SiC
  composition: {Si: 1, C: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-6223.3309, 314.17905, 0.38931308, 0.011875075, -1.6392772e-05, 1.1318082e-08,
      -3.0453242e-12, 86062.277, 23.101667]
    - [-6223.33089, 314.1790457, 0.389313083, 0.0118750754, -1.639277197e-05, 1.131808223e-08,
      -3.045324231e-12, 86062.2773, 23.10166717]
    - [-62688.0603, 720.983692, 2.162879732, 0.002201299585, -6.56946659e-07, 9.17711026e-11,
      -4.96916674e-15, 83212.2585, 16.01675317]
- name: NH
  composition: {N: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [19.96752, 1.7751348, 3.430861, 0.00078339598, -3.5956359e-06, 7.5834652e-09,
      -5.8566053e-12, 44236.24, 2.1049372]
    - [13596.5132, -190.0296604, 4.51849679, -0.002432776899, 2.377587464e-06, -2.592797084e-10,
      -2.659680792e-13, 42809.7219, -3.886561616]
    - [1958141.991, -5782.8613, 9.33574202, -0.002292910311, 6.07609248e-07, -6.64794275e-11,
      2.384234783e-15, 78989.1234, -41.169704]
    - [95246367.9, -85858.2691, 29.80445181, -0.002979563697, 1.656334158e-07, -4.74479184e-12,
      5.57014829e-17, 696143.427, -222.9027419]
- name: SiH
  composition: {Si: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-1591.6198, 100.9056, 2.2798003, 0.013872264, -7.447563e-05, 1.7795168e-07,
      -1.5369571e-10, 43783.73, 9.0769473]
    - [-6426.6763, 74.1725121, 3.9734916, -0.00414940888, 1.022918384e-05, -8.59238636e-09,
      2.567093743e-12, 42817.458, 2.24693715]
    - [404208.649, -2364.796524, 7.62749914, -0.002496591233, 1.10843641e-06, -1.943991955e-10,
      1.136251507e-14, 57047.3768, -24.48054429]
- name: 'NO'
  composition: {N: 1, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [1671.684, -56.705856, 4.9347042, -0.0077706491, 1.2497277e-05, 9.1085269e-09,
      -2.6517399e-11, 9932.9802, -1.1998353]
    - [-11439.16503, 153.6467592, 3.43146873, -0.002668592368, 8.48139912e-06, -7.68511105e-09,
      2.386797655e-12, 9098.21441, 6.72872549]
    - [223901.8716, -1289.651623, 5.43393603, -0.00036560349, 9.88096645e-08, -1.416076856e-11,
      9.38018462e-16, 17503.17656, -8.50166909]
    - [-957530354.0, 591243.448, -138.4566826, 0.01694339403, -1.007351096e-06, 2.912584076e-11,
      -3.29510935e-16, -4677501.24, 1242.081216]
- name: SN
  composition: {S: 1, N: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-4556.5142, 222.46769, -0.25454972, 0.036950676, -0.00016569368, 3.4950445e-07,
      -2.7404131e-10, 29747.207, 22.696071]
    - [-68354.1235, 1147.567483, -2.877802574, 0.0172486432, -2.058999904e-05, 1.26136964e-08,
      -3.139030141e-12, 25641.43612, 42.24006964]
    - [-483728.446, 1058.07559, 3.086198804, 0.000911136078, -2.764061722e-07, 4.15737011e-11,
      -2.128351755e-15, 23793.45477, 10.33222139]
- name: SiN
  composition: {Si: 1, N: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-14646.722, 137.49935, 3.6785086, -0.0061584992, 2.3094171e-05, -2.2944615e-08,
      7.3956657e-12, 46732.13, 6.4945641]
    - [-14646.72152, 137.4993497, 3.67850858, -0.0061584992, 2.309417067e-05, -2.294461481e-08,
      7.39566568e-12, 46732.1299, 6.494564115]
    - [-2932685.132, 5853.68859, 1.321451677, 0.001258329284, -3.77388636e-07, 6.88776104e-11,
      -4.18984259e-15, 6527.14881, 25.53145732]
- name: SO
  composition: {S: 1, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-33427.57, 640.38625, -1.0066412, 0.013815127, -1.7044864e-05, 1.0612949e-08,
      -2.6457962e-12, -3371.2922, 30.93862]
    - [-33427.57, 640.38625, -1.006641228, 0.01381512705, -1.704486364e-05, 1.06129493e-08,
      -2.645796205e-12, -3371.29219, 30.93861963]
    - [-1443410.557, 4113.87436, -0.538369578, 0.002794153269, -6.63335226e-07, 7.83822119e-11,
      -3.56050907e-15, -27088.38059, 36.15358329]
- name: S2
  composition: {S: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [4564.2801, -206.03445, 6.9557592, -0.027890588, 0.00011696312, -2.1177286e-07,
      1.4150025e-10, 15160.836, -8.1524147]
    - [35280.9178, -422.215658, 4.67743349, 0.001724046361, -3.86220821e-06, 3.33615634e-09,
      -9.93066154e-13, 16547.67715, -0.7957279032]
    - [-15881.28788, 631.548088, 2.449628069, 0.001986240565, -6.50792724e-07, 1.002813651e-10,
      -5.59699005e-15, 10855.08427, 14.58544515]
- name: C2H
  composition: {C: 2, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [13436.695, -506.79707, 7.7721074, -0.0065123398, 1.0301179e-05, -5.8801477e-09,
      1.2269019e-12, 68922.7, -18.718816]
    - [13436.69487, -506.797072, 7.77210741, -0.00651233982, 1.030117855e-05, -5.88014767e-09,
      1.226901861e-12, 68922.6999, -18.71881626]
    - [3922334.57, -12047.51703, 17.5617292, -0.00365544294, 6.98768543e-07, -6.82516201e-11,
      2.719262793e-15, 143326.6627, -95.6163438]
- name: HCN
  composition: {H: 1, C: 1, N: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 210.0, 1000.0, 6000.0]
    data:
    - [9168.4919, -428.56318, 10.687891, -0.059060527, 0.00025431946, -4.8166209e-07,
      3.435882e-10, 16721.129, -28.116449]
    - [90982.8693, -1238.657512, 8.72130787, -0.00652824294, 8.87270083e-06, -4.80888667e-09,
      9.3178985e-13, 20989.1545, -27.46678076]
    - [1236889.278, -4446.73241, 9.73887485, -0.000585518264, 1.07279144e-07, -1.013313244e-11,
      3.34824798e-16, 42215.1377, -40.05774072]
- name: C2H2
  composition: {C: 2, H: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 210.0, 1000.0, 6000.0]
    data:
    - [11413.687, -518.04687, 11.938503, -0.070084422, 0.0003215002, -6.037318e-07,
      4.1263119e-10, 27996.422, -34.376267]
    - [159811.2089, -2216.644118, 12.65707813, -0.00797965108, 8.05499275e-06, -2.433307673e-09,
      -7.52923318e-14, 37126.1906, -52.443389]
    - [1713847.41, -5929.10666, 12.36127943, 0.0001314186993, -1.362764431e-07, 2.712655786e-11,
      -1.302066204e-15, 62665.7897, -58.1896059]
- name: CH4
  composition: {C: 1, H: 4}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-683.72166, 58.471036, 2.4348751, 0.018632044, -0.00010989286, 3.0190433e-07,
      -2.695561e-10, -10176.076, 5.933564]
    - [-176685.0998, 2786.18102, -12.0257785, 0.0391761929, -3.61905443e-05, 2.026853043e-08,
      -4.97670549e-12, -23313.1436, 89.0432275]
    - [3730042.76, -13835.01485, 20.49107091, -0.001961974759, 4.72731304e-07, -3.72881469e-11,
      1.623737207e-15, 75320.6691, -121.9124889]
- name: ALH
  composition: {Al: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-2152.6181, 106.53124, 1.4487735, 0.017153794, -6.9713833e-05, 1.3607243e-07,
      -9.7792665e-11, 28552.609, 11.7155]
    - [-37591.1403, 508.900223, 1.128086896, 0.00398866091, -2.150790303e-07, -2.176790819e-09,
      1.020805902e-12, 26444.31827, 16.50021856]
    - [6802018.43, -21784.16933, 30.32713047, -0.01503343597, 4.49214236e-06, -6.17845037e-10,
      3.11520526e-14, 165830.1221, -187.6766425]
- name: ALOH
  composition: {Al: 1, O: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [7262.5943, -357.64797, 9.8875647, -0.05339046, 0.00027255349, -5.8270928e-07,
      4.5564096e-10, -23181.138, -22.824325]
    - [58764.9318, -944.942269, 7.82059918, 0.000585888847, -4.08366681e-06, 4.58722934e-09,
      -1.563936726e-12, -19932.83011, -20.65043885]
    - [788206.811, -2263.671626, 7.82395488, 0.0001821171456, -8.26372932e-08, 1.265414876e-11,
      -6.87597253e-16, -10398.08093, -22.09032458]
- name: AL2O
  composition: {Al: 2, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [1531.4253, -100.84957, 5.6743732, -0.0057280138, 7.9852977e-05, -2.3964583e-07,
      2.3630184e-10, -18604.185, -2.5252597]
    - [7776.5307, -129.4235361, 4.91250952, 0.00860422345, -1.217703648e-05, 8.31463487e-09,
      -2.237722201e-12, -18865.12879, -0.02806368311]
    - [-117107.4351, -178.3009166, 7.63321536, -5.33593177e-05, 1.180702791e-08, -1.355444579e-12,
      6.28732389e-17, -19475.80149, -14.15764167]
- name: CaOH
  composition: {Ca: 1, O: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [2492.04, -149.21752, 6.2204093, -0.023139735, 0.0001804995, -4.8016888e-07,
      4.3793457e-10, -21667.76, -5.0523908]
    - [46200.289, -928.567282, 9.17582877, -0.0039628288, 2.505447308e-06, 3.85206821e-11,
      -3.35277847e-13, -17980.09717, -25.3370485]
    - [1979972.994, -5598.88099, 11.51348706, -0.001668264707, 3.31257391e-07, -1.789056647e-11,
      -3.58071641e-16, 13401.96822, -46.4608426]
- name: MgH
  composition: {Mg: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-596.3156, 29.373498, 2.7350975, 0.0075864352, -3.5282493e-05, 7.8837619e-08,
      -6.1106393e-11, 26496.566, 6.5076108]
    - [-49586.7915, 750.027865, -0.64420475, 0.00982630101, -8.78982244e-06, 3.82335352e-09,
      -6.00372576e-13, 23022.79383, 26.57165344]
    - [-100574.8598, 1952.890106, -1.317191549, 0.0056036658, -2.13733498e-06, 3.3248805e-10,
      -1.824672746e-14, 15985.82755, 34.3123316]
- name: MgOH
  composition: {Mg: 1, O: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [844.81543, -62.766059, 4.9088654, -0.010262782, 0.00011328567, -3.1906678e-07,
      2.957071e-10, -17100.503, -0.085119958]
    - [38398.5162, -736.738364, 7.92066446, -0.000595094059, -2.112941162e-06, 3.22828211e-09,
      -1.214159329e-12, -13923.26188, -19.16078109]
    - [664866.475, -1770.750355, 7.26999927, 0.000533684276, -1.980894443e-07, 3.025677088e-11,
      -1.554849476e-15, -6149.11456, -16.71027009]
- name: PH3
  composition: {P: 1, H: 3}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [19847.224, -953.15418, 20.708128, -0.13338684, 0.00052157052, -9.5279768e-07,
      6.8223269e-10, 2858.3957, -72.267488]
    - [-6384.32534, 405.756741, -0.1565680086, 0.01338380613, -8.27539143e-06, 3.024360831e-09,
      -6.42176463e-13, -2159.842124, 23.85561888]
    - [1334801.106, -6725.46352, 14.45857073, -0.001639736883, 3.40921857e-07, -3.73627208e-11,
      1.672947506e-15, 39103.2571, -71.9878119]
- name: CO2
  composition: {C: 1, O: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [4664.813, -197.57385, 6.6378382, -0.026257318, 0.00012251332, -2.1503948e-07,
      1.3471069e-10, -47703.936, -8.7124642]
    - [49436.5054, -626.411601, 5.30172524, 0.002503813816, -2.127308728e-07, -7.68998878e-10,
      2.849677801e-13, -45281.9846, -7.04827944]
    - [117696.2419, -1788.791477, 8.29152319, -9.22315678e-05, 4.86367688e-09, -1.891053312e-12,
      6.33003659e-16, -39083.5059, -26.52669281]
    - [-1544423287.0, 1016847.056, -256.140523, 0.0336940108, -2.181184337e-06, 6.99142084e-11,
      -8.8423515e-16, -8043214.51, 2254.177493]
- name: TiO2
  composition: {Ti: 1, O: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-136.74484, 8.7435413, 3.5197901, 0.0087275803, -1.9433594e-05, 4.7938569e-08,
      -4.8691704e-11, -38033.055, 9.2093269]
    - [-1710.545601, 272.1435528, 0.596137896, 0.01925463599, -2.665500165e-05, 1.811109197e-08,
      -4.87671047e-12, -39122.4177, 24.08605889]
    - [154629.9764, -1046.25688, 7.78898583, -0.0001546805714, -7.05993595e-08, 3.100244802e-11,
      -2.49472543e-15, -32663.3675, -15.9153466]
- name: Si2C
  composition: {Si: 2, C: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-4553.3662, 131.47964, 2.4691069, 0.012765207, -1.6569108e-05, 1.0652897e-08,
      -2.739193e-12, 64700.499, 14.68839]
    - [-4553.3662, 131.4796415, 2.469106923, 0.0127652068, -1.656910776e-05, 1.065289663e-08,
      -2.739192976e-12, 64700.4992, 14.6883898]
    - [-125382.9442, -341.427779, 7.25436533, -0.0001017635503, 2.250902158e-08, -2.584074852e-12,
      1.198884876e-16, 66080.0938, -11.46216579]
- name: SiO2
  composition: {Si: 1, O: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [11412.182, -549.21241, 12.517074, -0.068509291, 0.00032040107, -6.7042371e-07,
      5.3311645e-10, -38018.855, -34.461352]
    - [-33629.4878, 473.407892, 0.2309770671, 0.01850230806, -2.242786671e-05, 1.364981554e-08,
      -3.35193503e-12, -42264.8749, 22.95803206]
    - [-146403.1193, -626.144106, 7.96456371, -0.0001854119096, 4.09521467e-08, -4.69720676e-12,
      2.17805428e-16, -37918.3477, -20.45285414]
- name: FeO
  composition: {Fe: 1, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [2565.4576, -108.66222, 5.2096771, -0.012877321, 4.6943547e-05, -5.9480727e-08,
      1.8145181e-11, 29581.148, 1.332307]
    - [15692.82213, -64.6018888, 2.45892547, 0.00701604736, -1.021405947e-05, 7.17929787e-09,
      -1.978966365e-12, 29645.72665, 13.26115545]
    - [-119597.148, -362.486478, 5.51888075, -0.000997885689, 4.37691383e-07, -6.79062946e-11,
      3.63929268e-15, 30379.85806, -3.63365542]
- name: NH2
  composition: {N: 1, H: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-515.6143, 30.091701, 3.3628751, 0.0062038136, -3.0525865e-05, 7.2559803e-08,
      -6.0066915e-11, 21676.23, 3.3151659]
    - [-31182.40659, 475.424339, 1.372395176, 0.00630642972, -5.98789356e-06, 4.49275234e-09,
      -1.414073548e-12, 19289.39662, 15.40126885]
    - [2111053.74, -6880.62723, 11.32305924, -0.001829236741, 5.64389009e-07, -7.88645248e-11,
      4.07859345e-15, 65037.7856, -53.59155744]
- name: NH3
  composition: {N: 1, H: 3}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [2587.587, -107.94339, 5.6469426, -0.011808841, 3.8145881e-05, -3.7452368e-08,
      1.1054345e-11, -6150.3969, -7.2055293]
    - [-76812.2615, 1270.951578, -3.89322913, 0.02145988418, -2.183766703e-05, 1.317385706e-08,
      -3.33232206e-12, -12648.86413, 43.66014588]
    - [2452389.535, -8040.89424, 12.71346201, -0.000398018658, 3.55250275e-08, 2.53092357e-12,
      -3.32270053e-16, 43861.9196, -64.62330602]
- name: CH2
  composition: {C: 1, H: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [2374.7123, -102.55758, 5.6398219, -0.012556411, 4.6727387e-05, -7.2002025e-08,
      4.3100623e-11, 45638.019, -6.9318795]
    - [32189.2173, -287.7601815, 4.20358382, 0.00345540596, -6.74619334e-06, 7.65457164e-09,
      -2.870328419e-12, 47336.2471, -2.143628603]
    - [2550418.031, -7971.62539, 12.28924487, -0.001699122922, 2.991728605e-07, -2.767007492e-11,
      1.05134174e-15, 96422.1689, -60.9473991]
- name: CH3
  composition: {C: 1, H: 3}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [1915.4984, -81.919921, 5.1091418, -0.0078895426, 3.7674315e-05, -5.2765053e-08,
      2.509889e-11, 16676.225, -4.9729264]
    - [-28761.88806, 509.326866, 0.2002143949, 0.01363605829, -1.433989346e-05, 1.013556725e-08,
      -3.027331936e-12, 14082.71825, 20.22772791]
    - [2760802.663, -9336.53117, 14.87729606, -0.001439429774, 2.444477951e-07, -2.224555778e-11,
      8.39506576e-16, 74818.0948, -79.196824]
- name: H2S
  composition: {H: 2, S: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 400.0, 1000.0, 6000.0]
    data:
    - [1270.3805, -53.415103, 4.8684821, -0.006893286, 2.396278e-05, -3.8876199e-08,
      3.7088694e-11, -3417.5039, -1.9620066]
    - [9543.80881, -68.7517508, 4.05492196, -0.0003014557336, 3.76849775e-06, -2.239358925e-09,
      3.086859108e-13, -3278.45728, 1.415194691]
    - [1430040.22, -5284.02865, 10.16182124, -0.000970384996, 2.154003405e-07, -2.1696957e-11,
      9.31816307e-16, 29086.96214, -43.49160391]
- name: V
  composition: {V: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-8368.1996, 388.32053, -3.4160696, 0.055893281, -0.00022389058, 4.0517095e-07,
      -2.7340245e-10, 59534.307, 32.887469]
    - [-55353.7602, 559.333851, 2.675543482, -0.00624304963, 1.565902337e-05, -1.372845314e-08,
      4.16838881e-12, 58206.6436, 9.52456749]
    - [1200390.3, -5027.0053, 10.58830594, -0.0050443261, 1.488547375e-06, -1.785922508e-10,
      8.113013866e-15, 91707.4091, -47.6833632]
    - [2456040166.0, -1339992.028, 278.1039851, -0.02638937359, 1.303527149e-06, -3.21468033e-11,
      3.099999094e-16, 10871520.43, -2439.95438]
- name: VO
  composition: {V: 1, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [-13116.198, 374.7817, 0.093008349, 0.012449777, -1.68854e-05, 1.1424434e-08,
      -3.0405893e-12, 15237.899, 25.368118]
    - [-13116.19784, 374.781697, 0.0930083486, 0.01244977714, -1.688540028e-05, 1.142443381e-08,
      -3.04058932e-12, 15237.8992, 25.36811755]
    - [2986190.283, -10113.44974, 17.18161749, -0.00787670503, 2.562279547e-06, -3.54740035e-10,
      1.770268056e-14, 79612.5488, -87.8999301]
    - [100453029.2, -90087.8023, 31.8610772, -0.002862639937, 1.440482244e-07, -3.80286147e-12,
      4.19509266e-17, 701140.129, -233.6810859]
- name: VO2
  composition: {V: 1, O: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 250.0, 1000.0, 6000.0]
    data:
    - [1546.044, -75.464645, 4.900099, -0.00308254, 3.0401797e-05, -5.3067893e-08,
      3.0103508e-11, -28960.33, 3.7078165]
    - [-6678.58586, 391.159758, -1.028549847, 0.02401523419, -3.33737881e-05, 2.283623543e-08,
      -6.1991431e-12, -30746.09276, 32.7791238]
    - [121063.2401, -1627.832993, 9.2527131, -0.001572703139, 5.23143016e-07, -6.47607114e-11,
      2.847226026e-15, -20973.06345, -25.47380687]
- name: NaCL
  composition: {Na: 1, Cl: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-683.14977, 11.459325, 3.1204677, 0.0034977989, 2.1445713e-05, -1.0293602e-07,
      1.1805314e-10, -22981.351, 8.5379025]
    - [43623.7835, -758.303446, 8.259173, -0.00964091514, 1.358854616e-05, -9.66703225e-09,
      2.74626129e-12, -19504.09477, -19.36687551]
    - [331449.876, -896.831565, 5.27728738, -0.0001475674008, -1.491128988e-08, 2.465673596e-11,
      -2.730355213e-15, -17362.77667, -3.99828856]
- name: KCL
  composition: {K: 1, Cl: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-864.63315, 21.240242, 2.9916229, 0.0069410678, 3.9125314e-06, -6.9449355e-08,
      9.6441722e-11, -27092.922, 9.9272961]
    - [9058.35151, -245.6801212, 5.68069619, -0.002900127425, 4.13098306e-06, -2.907340629e-09,
      8.22385087e-13, -25973.04884, -3.677976854]
    - [-212294.5722, 934.61589, 2.866264958, 0.001468386693, -5.83426078e-07, 1.255777709e-10,
      -9.150148e-15, -32737.8764, 14.01864636]
- name: MgSiO3(c)
  composition: {Mg: 1, Si: 1, O: 3}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 903.0, 1258.0, 1850.0]
    data:
    - [-16786.564, 903.23983, -18.409955, 0.18063492, -0.00049580534, 7.1610415e-07,
      -4.3844548e-10, -190655.6, 78.707724]
    - [663462.463, -10988.28549, 66.6158264, -0.1408979556, 0.000205614418, -1.506280668e-07,
      4.45600695e-11, -136597.7193, -370.410078]
    - [0.0, 0.0, 14.47351774, 0.0, 0.0, 0.0, 0.0, -191616.864, -76.6569042]
    - [0.0, 0.0, 14.72512607, 0.0, 0.0, 0.0, 0.0, -191737.1327, -78.29669776]
- name: Mg2SiO4(c)
  composition: {Mg: 2, Si: 1, O: 4}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 800.0, 2171.0]
    data:
    - [-29399.506, 1521.1035, -30.533663, 0.29047244, -0.0008179883, 1.1935707e-06,
      -7.1496188e-10, -268885.96, 130.96483]
    - [921203.503, -15081.56349, 90.8510222, -0.1898897815, 0.0002890690717, -2.266693109e-07,
      7.23068825e-11, -193609.783, -505.965007]
    - [3714548.59, -20415.18891, 57.8800914, -0.0362987449, 2.183358474e-05, -6.50480908e-09,
      7.90813368e-13, -149487.0518, -357.660837]
- name: SiC(c)
  composition: {Si: 1, C: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 298.15, 3105.0]
    data:
    - [-21679.252, 1140.5145, -21.886196, 0.19406912, -0.00078222618, 1.6753409e-06,
      -1.4268303e-09, -12964.748, 95.336096]
    - [-2285.496383, 0.0, -0.534910062, 0.01271547084, 0.0, 0.0, 0.0, -9193.1749,
      1.241441354]
    - [-126910.6658, 0.0, 3.75728696, 0.003481744565, -1.620660748e-06, 2.611097948e-10,
      0.0, -10466.6776, -21.09198538]
- name: Fe(c)
  composition: {Fe: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 500.0, 800.0, 1042.0, 1184.0, 1665.0, 1809.0]
    data:
    - [-1209.1263, 138.77536, -3.9162251, 0.054687805, -0.00017289373, 2.420782e-07,
      -1.1010449e-10, -956.72404, 15.516389]
    - [13504.90931, -780.380625, 9.44017147, -0.02521767704, 5.35017051e-05, -5.09909473e-08,
      1.993862728e-11, 2416.521408, -47.4900285]
    - [3543032.74, -24471.50531, 65.6102093, -0.0704392968, 3.18105287e-05, 0.0, 0.0,
      134505.9978, -413.378869]
    - [2661026334.0, -7846827.97, -728.921228, 26.13888297, -0.0349474214, 1.763752622e-05,
      -2.907723254e-09, 52348684.7, -15290.522]
    - [248192305.2, 0.0, -559.434909, 0.327170494, 0.0, 0.0, 0.0, 646750.343, 3669.16872]
    - [1442428576.0, -5335491.34, 8052.828, -6.30308963, 0.002677273007, -5.75004553e-07,
      4.71861196e-11, 32642642.5, -55088.5217]
    - [-345019003.0, 0.0, 705.750152, -0.544297789, 0.0001190040139, 0.0, 0.0, -804572.575,
      -4545.18032]
- name: AL2O3(c)
  composition: {Al: 2, O: 3}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 500.0, 1200.0, 2327.0]
    data:
    - [-63534.069, 3288.0548, -62.433037, 0.54354416, -0.0020863714, 4.2309726e-06,
      -3.4548933e-09, -214005.96, 272.69159]
    - [-5391549.97, 103667.6983, -817.322915, 3.38825872, -0.00751240036, 8.65924882e-06,
      -4.06608567e-09, -666013.465, 4235.50223]
    - [-604208.7868, 0.0, 14.75480816, 0.0008272285438, 0.0, 0.0, 0.0, -207923.5447,
      -81.3602948]
    - [0.0, 0.0, 12.93774378, 0.001992781294, 0.0, 0.0, 0.0, -206078.7581, -69.66603728]
- name: Na2S(c)
  composition: {Na: 2, S: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 307.66, 1024.25, 1276.0]
    data:
    - [27426.645, -1129.6505, 11.705785, 0.047767388, -0.00032456044, 8.6501905e-07,
      -8.5097923e-10, -41556.881, -64.532566]
    - [5227030.5, -60601.567, 293.24349, -0.68467374, 0.00090382384, -6.1424791e-07,
      1.6840644e-10, 254933.09, -1664.0116]
    - [10.37037, -11.626084, -2916.0337, 10.786069, -0.014789779, 8.9196229e-06, -1.9887548e-09,
      584235.93, 14300.257]
- name: KCL(c)
  composition: {K: 1, Cl: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 500.0, 1044.0]
    data:
    - [-5097.158, 405.23919, -6.1488739, 0.10368027, -0.00038595963, 6.7789473e-07,
      -4.5068136e-10, -55431.397, 27.405079]
    - [1179895.024, -22178.24961, 173.093063, -0.654543415, 0.001415897824, -1.598259237e-06,
      7.38753415e-10, 45607.8273, -899.152715]
    - [288.8789664, 0.0, 5.28708855, 0.00400409216, -4.34344901e-06, 2.753186288e-09,
      0.0, -54217.8564, -21.21648362]
- name: e-
  composition: {E: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -745.375, -11.720812]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -745.375, -11.72081224]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -745.375, -11.72081224]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -745.375, -11.72081224]
- name: H+
  composition: {H: 1, E: -1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 184021.49, -1.1406466]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 184021.4877, -1.140646644]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 184021.4877, -1.140646644]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 184021.4877, -1.140646644]
- name: H-
  composition: {H: 1, E: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 15976.155, -1.1390139]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 15976.15494, -1.139013868]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 15976.15494, -1.139013868]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 15976.15494, -1.139013868]
- name: Na+
  composition: {Na: 1, E: -1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 72565.371, 3.5508451]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 72565.3707, 3.55084508]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 72565.3707, 3.55084508]
    - [34012.0299, -21.37774622, 2.505443851, -7.18663169e-07, 5.18879639e-11, -1.944511626e-15,
      2.959355125e-20, 72734.1362, 3.50390406]
- name: K+
  composition: {K: 1, E: -1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 61075.169, 4.3474044]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 61075.1686, 4.34740444]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 61075.1686, 4.34740444]
    - [21779012.45, -14150.7663, 6.24895427, -0.00051874368, 3.95964068e-08, -1.584335542e-12,
      2.603558905e-17, 172275.3576, -27.8172899]
- name: H2O(c)
  composition: {H: 2, O: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 220.0, 273.15]
    data:
    - [23785.22, -1215.9927, 26.01892, -0.25625959, 0.0014904343, -4.062093e-06, 4.3465118e-09,
      -31763.793, -109.18119]
    - [-402677.748, 2747.887946, 57.3833663, -0.826791524, 0.00441308798, -1.054251164e-05,
      9.69449597e-09, -55303.1499, -190.2572063]
- name: TiO(c)
  composition: {Ti: 1, O: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 500.0, 1265.0, 1810.0, 2030.0]
    data:
    - [-14445.25, 771.10045, -15.304723, 0.14431671, -0.00047161843, 7.7528955e-07,
      -5.0884345e-10, -68624.951, 65.98637]
    - [-11261150.23, 210724.066, -1601.946203, 6.36071784, -0.01380180278, 1.563169158e-05,
      -7.23851478e-09, -1014208.223, 8368.05683]
    - [-206651.7895, 0.0, 8.24311527, -0.00489670216, 3.80667915e-06, 0.0, 0.0, -68153.8441,
      -42.6531047]
    - [-140008.2506, 0.0, 6.17330426, 0.001455287203, 0.0, 0.0, 0.0, -67563.5779,
      -32.49519859]
    - [208045.9342, 0.0, 5.795530945, 0.001404532558, 0.0, 0.0, 0.0, -66279.6409,
      -29.33709305]
- name: VO(c)
  composition: {V: 1, O: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 400.0, 500.0, 2063.0]
    data:
    - [-13046.063, 720.39521, -14.049331, 0.1299657, -0.00037874998, 5.6122519e-07,
      -3.5353088e-10, -55233.818, 60.913391]
    - [-20402608.24, 378310.228, -2848.005957, 11.18080448, -0.02407270159, 2.710471323e-05,
      -1.249526447e-08, -1758506.851, 14906.54242]
    - [-197001.5037, 0.0, 6.36572376, 0.00157663031, 0.0, 0.0, 0.0, -54441.6812, -33.8174157]
- name: NaOH
  composition: {Na: 1, O: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [-7142.1952, 335.91307, -2.2362077, 0.046920646, -9.9974914e-05, 5.9515812e-08,
      3.609932e-11, -25589.42, 31.15422]
    - [34420.3674, -792.321818, 8.9979323, -0.00407984452, 3.065783937e-06, -5.11918934e-10,
      -1.541016409e-13, -20869.51091, -25.1059009]
    - [875378.776, -2342.514649, 7.97846989, 0.0001016451512, -6.26853195e-08, 1.022715136e-11,
      -5.71328641e-16, -9509.90171, -22.02310401]
- name: KOH
  composition: {K: 1, O: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 1000.0, 6000.0]
    data:
    - [1658.0001, -127.19335, 5.1662287, -0.013021811, 0.00016275718, -5.1337221e-07,
      5.1949029e-10, -28842.942, -1.3008937]
    - [17706.84196, -615.320522, 8.68407572, -0.00396284951, 3.40865059e-06, -9.60197222e-10,
      8.49405497e-15, -26779.03261, -21.74495666]
    - [891727.195, -2334.179072, 7.97257871, 0.0001038863156, -6.31589347e-08, 1.027938106e-11,
      -5.73668582e-16, -14436.96469, -20.76401416]
- name: Fe(L)
  composition: {Fe: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [1809.0, 6000.0]
    data:
    - [0.0, 0.0, 5.535383324, 0.0, 0.0, 0.0, 0.0, -1270.608703, -29.48115042]
- name: Mg2SiO4(L)
  composition: {Mg: 2, Si: 1, O: 4}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [2170.0, 6000.0]
    data:
    - [0.0, 0.0, 24.65761662, 0.0, 0.0, 0.0, 0.0, -266935.7683, -134.6103798]
- name: SiC(L)
  composition: {Si: 1, C: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [3103.0, 6000.0]
    data:
    - [0.0, 0.0, 7.577115188, 0.0, 0.0, 0.0, 0.0, -7787.459, -43.67596159]
- name: MgSiO3(L)
  composition: {Mg: 1, Si: 1, O: 3}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [1850.0, 6000.0]
    data:
    - [0.0, 0.0, 17.6125833, 0.0, 0.0, 0.0, 0.0, -188021.0286, -95.12270574]
- name: H2O(L)
  composition: {H: 2, O: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [273.15, 373.15, 600.0]
    data:
    - [1326371304.0, -24482953.88, 187942.8776, -767.899505, 1.761556813, -0.002151167128,
      1.092570813e-06, 110176047.6, -977970.097]
    - [1263631001.0, -16803802.49, 92782.3479, -272.237395, 0.447924376, -0.000391939743,
      1.425743266e-07, 81131768.8, -513441.808]
- name: TiO(L)
  composition: {Ti: 1, O: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [2030.0, 6000.0]
    data:
    - [0.0, 0.0, 8.419016875, 0.0, 0.0, 0.0, 0.0, -63149.0383, -43.70051573]
- name: VO(L)
  composition: {V: 1, O: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [2063.0, 6000.0]
    data:
    - [0.0, 0.0, 8.419016875, 0.0, 0.0, 0.0, 0.0, -49213.5039, -43.29727468]
- name: NaAlSi3O8(c)
  composition: {Na: 1, Al: 1, Si: 3, O: 8}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 403.63, 1120.08, 1400.0]
    data:
    - [-100223.69, 4735.0542, -83.259797, 0.63377342, -0.0015233961, 1.6695346e-06,
      -5.9930187e-10, -493503.04, 379.82889]
    - [11077038.0, -102027.39, 385.95363, -0.64171717, 0.00066519537, -3.6117315e-07,
      8.0247335e-11, 53382.16, -2289.4261]
    - [441849.24, 92539.166, -339.90234, 0.58965922, -0.00045004626, 1.7045676e-07,
      -2.5557834e-11, -941993.16, 2089.3875]
- name: KAlSi3O8(c)
  composition: {K: 1, Al: 1, Si: 3, O: 8}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 406.45, 1031.03, 1400.0]
    data:
    - [-98789.209, 4650.7955, -81.367469, 0.61408217, -0.0014357536, 1.4901046e-06,
      -4.5928991e-10, -496132.37, 375.12372]
    - [17402913.0, -163692.22, 630.79146, -1.1504957, 0.0012499265, -7.1403836e-07,
      1.6758255e-10, 368048.71, -3724.0184]
    - [-7912.793, 443.35841, 2.3546959, 0.092764959, -9.717217e-05, 4.7964916e-08,
      -8.8775163e-12, -491387.51, -7.0521297]
- name: MgAl2O4(c)
  composition: {Mg: 1, Al: 2, O: 4}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 402.82, 946.26, 1343.02, 1800.0]
    data:
    - [-70033.967, 3181.1695, -54.232026, 0.40716738, -0.0010423006, 1.2977606e-06,
      -6.1765916e-10, -289896.97, 243.65099]
    - [19517300.0, -190075.12, 758.84176, -1.5154351, 0.0017305294, -1.0371204e-06,
      2.5525945e-10, 699726.97, -4458.3749]
    - [2643.8715, 11.826826, 6.1067203, 0.038160784, -3.8252232e-05, 1.915316e-08,
      -3.7425957e-12, -283033.75, -34.8553]
    - [-249772.04, -86930.854, 290.03991, -0.33616643, 0.00021273129, -6.6942185e-08,
      8.408156e-12, 142369.7, -1808.719]
- name: FeO(c)
  composition: {Fe: 1, O: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 301.21, 1652.69, 1800.0]
    data:
    - [-14445.113, 930.15124, -22.890598, 0.25561796, -0.0010765661, 2.3034246e-06,
      -1.9989845e-09, -36701.492, 95.888774]
    - [313255.22, -2995.1645, 16.344185, -0.017551243, 1.8761948e-05, -9.6900314e-09,
      1.9651225e-12, -18464.727, -89.747354]
    - [212575.13, -70621.247, 308.90857, -0.46134288, 0.00033304245, -1.1555153e-07,
      1.5595822e-11, 301578.5, -1858.7598]
- name: Fe2O3(c)
  composition: {Fe: 2, O: 3}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 304.84, 948.36, 1800.0]
    data:
    - [-19364.654, 993.60022, -17.570992, 0.10192777, 0.00033122702, -2.0540354e-06,
      2.725111e-09, -104351.79, 81.498671]
    - [3697565.0, -41040.409, 180.37234, -0.33606169, 0.00038148474, -2.2814199e-07,
      6.2201434e-11, 106256.73, -1048.9086]
    - [3666626300.0, -16587635.0, 31039.584, -30.672314, 0.016901445, -4.922452e-06,
      5.9262271e-10, 97939946.0, -205423.53]
- name: Fe3O4(c)
  composition: {Fe: 3, O: 4}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 304.84, 845.2, 1800.0]
    data:
    - [-24938.986, 1247.5058, -20.890301, 0.096178794, 0.00074535598, -3.6086626e-06,
      4.5293414e-09, -141019.26, 101.77304]
    - [6720718.9, -71502.041, 293.78326, -0.49001551, 0.0004382476, -1.7468615e-07,
      4.4281385e-11, 227811.8, -1730.246]
    - [2702854300.0, -12874739.0, 25325.737, -26.227587, 0.015116611, -4.593361e-06,
      5.752072e-10, 75307929.0, -166363.79]
- name: CaMgSi2O6(c)
  composition: {Ca: 1, Mg: 1, Si: 2, O: 6}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 409.68, 1314.0, 1600.0]
    data:
    - [-45003.873, 2374.3043, -45.24662, 0.35088809, -0.00058189866, -3.0465436e-09,
      6.3037915e-10, -397099.3, 202.74789]
    - [2384242.8, -24432.888, 105.38694, -0.13283056, 0.00013246531, -6.6205627e-08,
      1.3166716e-11, -265191.4, -617.58457]
    - [-238833.46, -60519.954, 309.3785, -0.49579003, 0.0004239925, -1.7419635e-07,
      2.7780181e-11, -128975.07, -1814.5702]
- name: Fe2SiO4(c)
  composition: {Fe: 2, Si: 1, O: 4}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 508.06, 1491.86, 1800.0]
    data:
    - [-53099.156, 2796.0357, -55.397897, 0.48387654, -0.0014408282, 2.1186819e-06,
      -1.2328092e-09, -189899.7, 246.04516]
    - [42385091.0, -317347.31, 977.05889, -1.4999273, 0.0012932723, -5.7822011e-07,
      1.0542439e-10, 1537203.4, -5984.0874]
    - [-376774.82, 117678.16, -506.44992, 0.87088876, -0.00066349626, 2.4210178e-07,
      -3.4263359e-11, -718418.66, 3052.0674]
- name: PH2
  composition: {P: 1, H: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [1703.1154, -66.607927, 4.9225765, -0.0056667058, 1.3097672e-05, 7.5232858e-09,
      -2.6788099e-11, 14269.736, -1.56963]
    - [15552.68372, -184.1602025, 4.89589604, -0.0034954366, 1.053418945e-05, -8.37756292e-09,
      2.27076615e-12, 14098.39468, -2.210564792]
    - [1127884.913, -4715.23825, 10.214983, -0.00116757382, 2.150542671e-07, -1.624213739e-11,
      3.76622524e-16, 41830.7463, -42.3162325]
- name: P2
  composition: {P: 2}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [3455.6029, -156.40436, 6.0621803, -0.020459898, 8.354975e-05, -1.4029617e-07,
      8.3920277e-11, 16867.859, -5.3145874]
    - [30539.2251, -324.617759, 4.02246381, 0.00323209479, -5.51105245e-06, 4.19557293e-09,
      -1.21503218e-12, 17969.1087, 1.645350331]
    - [-780693.649, 2307.91087, 1.41174313, 0.00210823742, -7.36085662e-07, 1.25936012e-10,
      -7.07975249e-15, 1329.82474, 21.69741365]
- name: PS
  composition: {P: 1, S: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [-768.97868, -46.378241, 4.1046492, 0.0015552623, -2.3157573e-06, 1.6614252e-09,
      -4.6312384e-13, 17078.758, 4.2306522]
    - [-768.97868, -46.3782407, 4.10464917, 0.001555262273, -2.315757288e-06, 1.661425178e-09,
      -4.6312384e-13, 17078.75808, 4.230652171]
    - [-270272.9081, 888.354822, 3.16919012, 0.001022480817, -3.80374048e-07, 7.01986188e-11,
      -4.26912231e-15, 11215.10462, 11.47334049]
- name: PO
  composition: {P: 1, O: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [-4686.0338, 227.90542, -0.3307207, 0.037264106, -0.00016583848, 3.4463762e-07,
      -2.636505e-10, -4795.9011, 23.16909]
    - [-68457.5406, 1141.295708, -2.77955606, 0.01678458047, -1.974879516e-05, 1.19260232e-08,
      -2.927460912e-12, -9847.74504, 41.84328297]
    - [-336666.744, 622.935584, 3.56560546, 0.000651662072, -2.061770841e-07, 3.18441323e-11,
      -1.573691908e-15, -8939.79039, 6.954859188]
- name: P4O6
  composition: {P: 4, O: 6}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [21749.274, -1136.6729, 21.099511, -0.14457794, 0.001052711, -2.5630229e-06,
      2.1791479e-09, -192130.8, -67.747867]
    - [376089.475, -5685.83262, 29.1422545, 0.0225153212, -4.51026963e-05, 3.58831269e-08,
      -1.05757206e-11, -168856.248, -145.1359851]
    - [-1008997.24, -887.275399, 28.6606483, -0.000263601894, 5.81094072e-08, -6.64808696e-12,
      3.07439193e-16, -199713.89, -128.1853821]
- name: PH
  composition: {P: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [69.408194, 1.616182, 3.4015044, 0.0011238551, -5.0830804e-06, 9.8589429e-09,
      -5.7152119e-12, 26724.507, 4.0852051]
    - [22736.33198, -397.267406, 6.23369766, -0.0091817846, 1.523328123e-05, -1.085888585e-08,
      2.929760547e-12, 28527.68404, -10.95191197]
    - [781473.065, -3038.451204, 7.46748102, -0.001837522255, 7.1659477e-07, -1.142128853e-10,
      6.17541056e-15, 45362.6018, -24.6729814]
- name: TiO2(c)
  composition: {Ti: 1, O: 2}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 500.0, 2185.0]
    data:
    - [-22530.85, 1380.6723, -26.760556, 0.24222692, -0.0008198002, 1.3993445e-06,
      -9.4895016e-10, -119336.76, 116.7831]
    - [17122641.57, -299705.9314, 2143.439285, -7.99723044, 0.01661745261, -1.817331921e-05,
      8.17940336e-09, 1253121.023, -11325.09587]
    - [-124420.1863, 0.0, 7.60057734, 0.001421666301, 0.0, 0.0, 0.0, -116283.1515,
      -38.3405343]
- name: TiO2(L)
  composition: {Ti: 1, O: 2}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [2185.0, 6000.0]
    data:
    - [0.0, 0.0, 12.02716696, 0.0, 0.0, 0.0, 0.0, -114326.1561, -65.51584491]
- name: H3PO4(L)
  composition: {H: 3, P: 1, O: 4}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [315.5, 1000.0]
    data:
    - [41090.8111, -441.03435, 8.54779165, 0.0319826133, 5.16752953e-06, -3.2562166e-09,
      8.32314874e-13, -154317.538, -41.5539865]
- name: H3PO4(c)
  composition: {H: 3, P: 1, O: 4}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 315.5]
    data:
    - [2733.3655, 0.0, 1.1093479, 0.041290246, -7.8103533e-06, 0.0, 0.0, -156575.32,
      -4.9734731]
    - [2733.365471, 0.0, 1.109347863, 0.0412902457, -7.81035332e-06, 0.0, 0.0, -156575.3156,
      -4.97347313]
- name: H3PO4
  composition: {H: 3, P: 1, O: 4}
  thermo:
    model: NASA9
    temperature-ranges: [50.0, 200.0, 1000.0, 6000.0]
    data:
    - [-4295.47479, 150.5758206, 5.21139255, -0.1075803753, 0.001423367904, -5.80379814e-06,
      8.54807975e-09, -137270.5851, 13.5717078]
    - [68778.2841, -633.654622, -0.58115031, 0.0729251273, -0.0001175926437, 8.89871481e-08,
      -2.573888504e-11, -132900.8776, 23.68825177]
    - [1123325.607, -4156.50131, 19.45097544, 0.001410063043, -4.52922012e-07, 6.30754405e-11,
      -3.2962265e-15, -113748.6346, -81.5316799]
- name: HCL
  composition: {H: 1, Cl: 1}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [151.294, -2.0421803, 3.4634402, 0.00062492228, -3.1362016e-06, 6.6633709e-09,
      -4.8821514e-12, -12130.519, 2.6637315]
    - [20625.88287, -309.3368855, 5.27541885, -0.00482887422, 6.1957946e-06, -3.040023782e-09,
      4.91679003e-13, -10677.82299, -7.309305408]
    - [915774.951, -2770.550211, 5.97353979, -0.000362981006, 4.73552919e-08, 2.810262054e-12,
      -6.65610422e-16, 5674.95805, -16.42825822]
- name: SiH4
  composition: {Si: 1, H: 4}
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 200.0, 1000.0, 6000.0]
    data:
    - [11196.753, -491.44835, 11.704194, -0.055726075, 0.00018350469, -1.892093e-07,
      2.5824251e-11, 4993.2366, -33.59372]
    - [78729.9329, -552.608705, 2.498944303, 0.01442118274, -8.46710731e-06, 2.726164641e-09,
      -5.43675437e-13, 6269.66906, 4.96546183]
    - [1290378.74, -7813.39978, 18.28851664, -0.001975620946, 4.15650215e-07, -4.59674561e-11,
      2.072777131e-15, 47668.8795, -98.0169746]
- name: NH4CL(c)
  composition: {N: 1, H: 4, Cl: 1}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 457.7, 1000.0, 1500.0]
    data:
    - [-19047.875, 852.32706, -12.854248, 0.13602874, -0.00030567764, 2.7045329e-07,
      1.7617684e-11, -42746.675, 57.522706]
    - [159338.9657, 0.0, -5.96585494, 0.0649419317, -5.39145689e-05, 0.0, 0.0, -37928.815,
      29.3301304]
    - [-16418201.57, 163926.2758, -665.452963, 1.447086973, -0.00169825416, 1.056255415e-06,
      -2.697574651e-10, -878797.59, 3897.9369]
    - [840959266.0, -3426760.78, 5548.09147, -4.4356743, 0.001772486202, -2.81716012e-07,
      0.0, 20633631.45, -37659.7636]
- name: SiO2(c)
  composition: {Si: 1, O: 2}
  condensate: true
  thermo:
    model: NASA9
    temperature-ranges: [60.0, 300.0, 848.0, 1200.0, 1996.0]
    data:
    - [-14220.599, 602.64596, -9.3755632, 0.086944822, -0.00022239816, 2.9935609e-07,
      -1.5533946e-10, -112535.48, 41.897011]
    - [-577689.55, 7214.66111, -31.45730294, 0.0741217715, -8.67007782e-06, -1.080461312e-07,
      8.31632491e-11, -146239.8375, 184.2424399]
    - [23176.35074, 0.0, 7.026511484, 0.001241925261, 0.0, 0.0, 0.0, -111701.2474,
      -35.80751356]
    - [-535641.9079, 0.0, 9.331036946, -0.0007306503931, 3.339944266e-07, 0.0, 0.0,
      -113432.6721, -49.98768383]
- name: Fe+
  composition: {Fe: 1, E: -1}
  thermo:
    model: NASA9
    temperature-ranges: [298.15, 1000.0, 6000.0, 20000.0]
    data:
    - [-56912.3162, 184.713439, 4.19697212, -0.00597827597, 1.054267912e-05, -8.05980432e-09,
      2.256925874e-12, 140120.6571, -0.360254258]
    - [-817645.009, 1925.359408, 1.717387154, 0.000338533898, -9.81353312e-08, 2.228179208e-11,
      -1.483964439e-15, 128635.2466, 15.00256262]
    - [106521749.1, -28839.23997, -2.821752459, 0.002712846797, -3.107069182e-07,
      1.543726493e-11, -2.725133516e-16, 414298.169, 40.5349733]
- name: He+
  composition: {He: 1, E: -1}
  thermo:
    model: NASA9
    temperature-ranges: [298.15, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 285323.3739, 1.621665557]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 285323.3739, 1.621665557]
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 285323.3739, 1.621665557]
- name: Ca+
  composition: {Ca: 1, E: -1}
  thermo:
    model: NASA9
    temperature-ranges: [298.15, 1000.0, 6000.0, 20000.0]
    data:
    - [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 92324.1779, 5.07767498]
    - [3747070.82, -11747.07738, 16.72546969, -0.00833479771, 2.394593294e-06, -2.988243468e-10,
      1.356563002e-14, 166432.9088, -95.8282126]
    - [911712841.0, -622042.846, 168.3741136, -0.0214086267, 1.452947686e-06, -4.92079088e-11,
      6.575369235e-16, 4959472.06, -1422.600719]
- name: FeH
  composition: {Fe: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [298.15, 1500.0, 3000.0, 6000.0]
    data:
    - [53963.6804, 12.01165134, 1.998571629, 0.00638488226, -4.25289601e-06, 1.014284902e-09,
      -9.68295538e-15, 53250.3638, 12.47340794]
    - [-1583699.238, 3244.8417, 1.401347727, 0.002880192403, -1.000969824e-06, 1.591975439e-10,
      -1.047710944e-14, 30779.81097, 21.1939291]
    - [2236139.588, -4045.84955, 7.71039353, -0.0002751532522, -5.66561224e-08, 2.507012919e-12,
      5.67533294e-16, 79834.3529, -25.13344168]
"""
