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

        # Change the atomic composition to Lodders (2020)
        gas = self.gdat.gas
        molfracs_atoms_sun = gas.molfracs_atoms_sun
        for i,atom in enumerate(gas.atoms_names):
            molfracs_atoms_sun[i] = LODDERS2020_SUN_COMP[atom]
        molfracs_atoms_sun /= np.sum(molfracs_atoms_sun)
        gas.molfracs_atoms_sun = molfracs_atoms_sun

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

class EquilibriumChemistry(ChemEquiAnalysis):
    """Compute equilibrium gas and condensate abundances for a given P–T
    profile, optionally with custom elemental ratios."""

    def __init__(self, thermofile, atoms=None, species=None, method=None):
        """Create an equilibrium chemistry solver.

        Parameters
        ----------
        thermofile : str
            Path to a YAML thermodynamics file.
        atoms : list[str], optional
            Subset of atom symbols to include; defaults to all in the thermo
            file.
        species : list[str], optional
            Species names to load; defaults to the bundled list when
            `thermofile` is None.
        method : str, optional
            Optional preset for configuring the solver. ``"sonora-approx"`` loads
            the Sonora-specific nominal species list and mass-tolerance tweaks;
            any other value leaves ``atoms``/``species`` untouched so you can
            supply custom thermo data. Use ``None`` when providing your own
            configuration.
        """

        if method == 'sonora-approx':
            atoms = None
            species = SONORA_NOMINAL_SPECIES

        super().__init__(
            thermofile,
            atoms,
            species 
        )

        if method == 'sonora-approx':
            # matches the easyCHEM internals
            self.mass_tol = 1.0e-2 

            # Change the atomic composition to Lodders (2020)
            molfracs_atoms_sun = self.molfracs_atoms_sun
            for i,atom in enumerate(self.atoms_names):
                molfracs_atoms_sun[i] = LODDERS2020_SUN_COMP[atom]
            molfracs_atoms_sun /= np.sum(molfracs_atoms_sun)
            self.molfracs_atoms_sun = molfracs_atoms_sun

        self._molfracs_atoms_sun_save = self.molfracs_atoms_sun.copy()
        self.method = method

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

    def equilibrate_atmosphere(self, P, T, log10mh, CtoO_relative):
        """Solve for equilibrium chemistry across a vertical profile.

        Parameters
        ----------
        P : ndarray
            Pressures in bar.
        T : ndarray
            Temperatures in Kelvin aligned with `P`.
        log10mh : float
            log10 metallicity relative to solar.
        CtoO_relative : float
            The C/O ratio relative to solar.

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

        # Some conversions and copies
        P_cgs = P*1e6
        metallicity = 10.0**log10mh
        gas_names = self.gas_names
        condensate_names = self.condensate_names

        gases = {}
        for key in gas_names:
            gases[key] = np.empty(len(P))
        condensates = {}
        for key in condensate_names:
            condensates[key] = np.empty(len(P))

        for i in range(len(P)):
            if i > 0:
                self.use_prev_guess = True
            # Try many perturbations on T to try to get convergence
            for eps in [0.0, 1.0e-12, -1.0e-12, 1.0e-8, -1.0e-8, 1.0e-6, -1.0e-6, 1.0e-4, -1.0e-4]:
                converged = self.solve_metallicity(P_cgs[i], T[i] + T[i]*eps, metallicity, CtoO_relative)
                if converged:
                    break
            if not converged:
                # We will not enforce convergence.
                pass
            molfracs_species_gas = self.molfracs_species_gas
            molfracs_species_condensate = self.molfracs_species_condensate
            for j,key in enumerate(gas_names):
                gases[key][i] = molfracs_species_gas[j]
            for j,key in enumerate(condensate_names):
                condensates[key][i] = molfracs_species_condensate[j]
        self.use_prev_guess = False

        return gases, condensates
    
# The composition of the Sun from Lodders (2020)
LODDERS2020_SUN_COMP = {
    'H':  9.082387E-01,
    'He': 9.046346E-02,
    'Li': 2.050745E-09,
    'C':  3.286959E-04,
    'N':  7.893027E-05,
    'O':  5.982842E-04,
    'F':  4.577235E-08,
    'Na': 2.083182E-06,
    'Mg': 3.712245E-05,
    'Si': 3.604122E-05,
    'P':  2.977005E-07,
    'S':  1.575001E-05,
    'Cl': 1.906580E-07,
    'K':  1.301448E-07,
    'Ti': 8.862536E-08,
    'V':  9.911335E-09,
    'Cr': 4.732212E-07,
    'Fe': 3.142794E-05,
    'Rb': 2.584155E-10,
    'Cs': 1.326317E-11,
}
# Normalize so they sum to 1
tot = sum(LODDERS2020_SUN_COMP.values())
for key in LODDERS2020_SUN_COMP:
    LODDERS2020_SUN_COMP[key] /= tot

SONORA_NOMINAL_SPECIES = [

# Initial 31 species (up to COS) from *reported output* of original Sonora grids (Summer 2015).
# Li-bearing species and OH, C-gr added with updates following Gharib-Nezhad et al (2021).
# Additional atomics and ions added to output list (summer 2024).
# First 50 species (up to O+) are currently included in output of Sonora chemistry grids.
#
#
# NOTE: For stability and flexibility, this version uses component metal oxides as stoichiometric proxies
# for oxide and silicate condensates in substellar atmopsheres. More precise condensation curves
# may be found by replacing these oxides with the expected species in the condensate sequence.


    'e-','H2', 'H', 'H+', 'H-','H2-', 'H2+','H3+',
    'He',
    'H2O',
    'CH4','CO',
    'NH3','N2',
    'PH3','H2S',
    'TiO','VO','Fe','FeH','CrH',
    'Na','K','Rb','atCs',
    'CO2','HCN','C2H2','C2H4','C2H6','COS',
    'SiO','MgH',
    'Li','LiOH','LiH','LiCl','Li+','LiF',
    'OH','C-gr',
    'Mg','Mg+','Si','Fe+','Ti','Ti+','C','O','C+','O+',
    # Additional species included in calculation
    # MWE list that still approximates Sonora chemistry grids
    'He+',
    'C2','CH','CN',
    'CS','C2H','CH2','CH3','C3H8','HCHO','CH2OH','CH3OH','CH3O',
    'N','NH','NH2','NO','N2H2','N2H4',
    'O2','H2O2',
    'P','PH2', 'P2','PO','PH','P4O6(Gurvich)',
    #'P4O6(Gurvich)','P4O6(JANAF)','HPO2','H3PO4','PN','PS',
    'S','SH','SN',
    'SO', 'S2','SO2','S-','SH-',
    'Cr','Cr+','CrO','CrO2',
    'FeO','FeOH','FeS','Fe(OH)2','FeCl',
    'MgO','MgOH','MgS','Mg(OH)2',
    'Si+','SiS','SiH','SiO2','SiH2','SiH3','SiH4',
    'Na+', 'NaCl','NaOH','NaH',
    'K+','KCl','KH','KOH',
    'V', 'V+','VO2',
    'TiO2',
    'Cl-','Cl','HCl',#'Cl2',
    'RbCl','Rb+','RbH','RbO','RbOH','RbF',
    'CsCl','Cs+','CsH',
    'F','F-','HF','NaF',#'F2',
    # Al- and Ca-bearing species (optional)
    #'Al','AlH','AlO','AlOH','Al2O','AlCl','AlCl2','AlCl3',
    #'Ca','Ca+','CaH','CaO','CaOH',
# Condensates included in the calculation
    'NH4H2PO4(c)',
    'VO(c)','VO(L)',     #proxy
    'TiO2(c)','TiO2(L)', #proxy
    'MgO(c)','MgO(L)',   #proxy
    'SiO2(c)','SiO2(L)', #proxy
    #'MgSiO3(c)','Mg2SiO4(c)',
    'Cr(c)','Cr(L)',
    'Fe(c)','Fe(L)',
    'H2O(L)','H2O(c)',
    'Na2S(c)',
    'KCl(c)',
    'RbCl(c)',
    'CsCl(c)',
    'Li2S(c)',
    'LiF(cr)'
    #'Gehlenite(c)','Grossite(c)',
    # 'Al2O3(c)','CaO(c)'
    # Additional species as needed:
]
