Tutorials
=========

Step-by-step introductions to each part of PICASO, from a first spectrum to 3D models and fitting data. If you have not set up PICASO yet, start with :doc:`getting_started`. Every tutorial is a Jupytext ``.py`` file you can run as a notebook (see :ref:`notebook_workflow`).

.. toctree::
   :caption: Spectral modeling basics
   :maxdepth: 1

   Getting Started <notebooks/A_basics/1_GetStarted.py>
   Simple Clouds <notebooks/A_basics/2_AddingClouds.py>
   Surface Reflectivity <notebooks/A_basics/3_AddingSurfaceReflectivity.py>
   Plot Diagnostics <notebooks/A_basics/4_PlotDiagnostics.py>
   Thermal Emission Spectroscopy <notebooks/A_basics/5_AddingThermalFlux.py>
   Transmission Spectroscopy <notebooks/A_basics/6_AddingTransitSpectrum.py>
   Brown Dwarf Spectroscopy <notebooks/A_basics/7_BrownDwarfs.py>

.. toctree::
   :caption: Chemistry
   :maxdepth: 1

   Chemical Equilibrium & Disequilibrium Hacks <notebooks/B_chemistry/1_ChemicalEquilibrium.py>
   Full Kinetics/Photochemistry <notebooks/B_chemistry/2_Photochemistry.py>

.. toctree::
   :caption: Clouds
   :maxdepth: 1

   Virga (Ackerman & Marley Clouds) <notebooks/C_clouds/1_PairingPICASOToVIRGA.py>
   Patchy Clouds <notebooks/C_clouds/2_PatchyClouds.py>

1D climate modeling: cite `Mukherjee et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023ApJ...942...71M/abstract>`_ and `Mang et al. 2026 <https://ui.adsabs.harvard.edu/abs/2026ApJ..1000...98M/abstract>`_.

.. toctree::
   :caption: 1D climate modeling
   :maxdepth: 1

   Brown Dwarfs <notebooks/D_climate/1_BrownDwarf_PreW.py>
   Brown Dwarfs w/ Resort-Rebin <notebooks/D_climate/1b_BrownDwarf_ResortRebin_Chemeq.py>
   Planets <notebooks/D_climate/2_Exoplanet_PreW.py>
   Planets w/ Resort-Rebin <notebooks/D_climate/2b_Exoplanet-ResortRebin-Chemeq.py>
   Planets w/ Photochemistry <notebooks/D_climate/3_Exoplanet-Photochemistry.py>
   Brown Dwarfs w/ Disequilibrium Chemistry (Self-Consistent Kzz) <notebooks/D_climate/4_BrownDwarf_DEQ_SC_kzz.py>
   Brown Dwarfs w/ Disequilibrium Chemistry (Constant Kzz) <notebooks/D_climate/4b_BrownDwarf_DEQ_const_kzz.py>
   Brown Dwarfs w/ Clouds <notebooks/D_climate/5_CloudyBrownDwarf_PreW.py>
   Brown Dwarfs w/ Clouds and Disequilibrium Chemistry <notebooks/D_climate/6_CloudyBrownDwarf_DEQ.py>
   Creating a Grid of Models for Fitting <notebooks/D_climate/7_CreateModelGrid.py>
   Brown Dwarfs w/ Energy Injection <notebooks/D_climate/8_EnergyInjection.py>
   Brown Dwarfs w/ Moist Adiabat <notebooks/D_climate/9_BrownDwarf_Moistgrad.py>

3D spectra: cite `Adams et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022ApJ...926..157A/abstract>`_. Phase curves: cite `Robbins-Blanch et al. 2022 <http://arxiv.org/abs/2204.03545>`_.

.. toctree::
   :caption: 3D spectra and phase curves
   :maxdepth: 1

   Non-Zero Phase and Spherical Integration <notebooks/E_3dmodeling/1_SphericalIntegration.py>
   Basics of a 3D Calculation <notebooks/E_3dmodeling/2_3DInputsWithPICASOandXarray.py>
   Post-Processing Chemistry for 3D Runs <notebooks/E_3dmodeling/3_PostProcess3Dinput-Chemistry.py>
   Post-Processing Clouds for 3D Runs <notebooks/E_3dmodeling/4_PostProcess3Dinput-Clouds.py>
   Modeling a 3D Spectrum (Adams et al. 2022) <notebooks/E_3dmodeling/5_3DSpectra.py>
   Thermal Phase Curve pt 1 (Robbins-Blanch et al. 2022) <notebooks/E_3dmodeling/6_PhaseCurves.py>
   Thermal Phase Curve pt 2 (Robbins-Blanch et al. 2022) <notebooks/E_3dmodeling/7_PhaseCurves-wChemEq.py>
   Reflected Light Phase Curve (Hamill et al. 2024) <notebooks/E_3dmodeling/8_ReflectedPhaseCurve.py>

Fitting data: cite the papers listed under *Grid fits* and *Retrievals* in :doc:`credit`.

.. toctree::
   :caption: Fitting data
   :maxdepth: 1

   Grid Search Analysis <notebooks/F_fitdata/1_GridSearch.py>
   Parsing Observational Data <notebooks/J_driver&retrievals/2_Data_Parser.py>
   PT, Chemistry and Cloud Parameterizations <notebooks/J_driver&retrievals/3_Parameterizations.py>
   Retrieval Setup <notebooks/J_driver&retrievals/4_Retrieval_Setup.py>
   Retrieval Analysis <notebooks/J_driver&retrievals/5_Retrieval_Analysis.py>

Looking for configuration-file workflows, opacities or troubleshooting? Those are in :doc:`howto`. For end-to-end lessons from summer schools, see :doc:`workshops`.
