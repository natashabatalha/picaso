Tutorials
=========

.. note::
   PICASO tutorials are now provided as Jupytext ``.py`` files. To learn how to download and run these as notebooks, please see our :ref:`Tutorial Workflow <notebook_workflow>` page.

Getting Data and Setting up Environments after Installation
-----------------------------------------------------------

This quickstart notebook has some overlap with our installation instructions, but it is designed to be a quick way to get up and running with PICASO after you have installed it. It includes instructions for getting reference data, setting up environments, and checking environments have been successfully setup. 
.. toctree::
   :maxdepth: 1

   Quickstart </notebooks/Quickstart.py>

Basics of Spectral Modeling 
---------------------------

.. toctree::
   :maxdepth: 1

   Getting Started </notebooks/A_basics/1_GetStarted.py>
   Simple Clouds  </notebooks/A_basics/2_AddingClouds.py>
   Surface Reflectivity </notebooks/A_basics/3_AddingSurfaceReflectivity.py>
   Plot Diagnostics </notebooks/A_basics/4_PlotDiagnostics.py>
   Thermal Emission Spectroscopy </notebooks/A_basics/5_AddingThermalFlux.py>
   Transmission Spectroscopy </notebooks/A_basics/6_AddingTransitSpectrum.py>
   Brown Dwarf Spectroscopy </notebooks/A_basics/7_BrownDwarfs.py>

Chemistry
---------

.. toctree::
   :maxdepth: 1

   Chemical Equilibrium & Disequilibrium Hacks </notebooks/B_chemistry/1_ChemicalEquilibrium.py>
   Full Kinetics/Photochemistry  </notebooks/B_chemistry/2_Photochemistry.py>
 

Clouds
------

.. toctree::
   :maxdepth: 1

   Virga (Ackerman & Marley Clouds) </notebooks/C_clouds/1_PairingPICASOToVIRGA.py>
   Patchy Clouds </notebooks/C_clouds/2_PatchyClouds.py>

1D Climate Modeling
-------------------
Relevant Citatons: `Mukherjee et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022arXiv220807836M/abstract>`_

.. toctree::
   :maxdepth: 1

   Brown Dwarfs </notebooks/D_climate/1_BrownDwarf_PreW.py>
   Planet </notebooks/D_climate/2_Exoplanet_PreW.py>
   Planet w/ Photochemistry </notebooks/D_climate/3_Exoplanet-Photochemistry.py>
   Brown Dwarfs w/ Disequilibrium Chemistry (Self-Consistent Kzz) </notebooks/D_climate/4_BrownDwarf_DEQ_SC_kzz.py>
   Brown Dwarfs w/ Disequilibrium Chemistry (Constant Kzz) </notebooks/D_climate/4b_BrownDwarf_DEQ_const_kzz.py>
   Brown Dwarfs w/ Clouds </notebooks/D_climate/5_CloudyBrownDwarf_PreW.py>
   Brown Dwarfs w/ Clouds and Disequilibrium Chemistry </notebooks/D_climate/6_CloudyBrownDwarf_DEQ.py>
   Creating a grid of models to use with PICASO's fitting tools </notebooks/D_climate/7_CreateModelGrid.py>
   Brown Dwarfs w/ Energy Injection </notebooks/D_climate/8_EnergyInjection.py>

3D Spectra and Phase Curves
---------------------------
Relevant Citatons: `Adams et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022ApJ...926..157A/abstract>`_ for 3D spectra and `Robbins-Blanch et al. 2022 <http://arxiv.org/abs/2204.03545>`_ for Phase Curves. 

.. toctree::
   :maxdepth: 1

   Non-Zero Phase and Spherical Integration </notebooks/E_3dmodeling/1_SphericalIntegration.py>
   Basics of a 3D Calculation </notebooks/E_3dmodeling/2_3DInputsWithPICASOandXarray.py>
   Post-Processing Chemistry for 3D runs </notebooks/E_3dmodeling/3_PostProcess3Dinput-Chemistry.py>
   Post-Processing Clouds for 3D runs </notebooks/E_3dmodeling/4_PostProcess3Dinput-Clouds.py>
   Modeling a 3D Spectrum (Adams et al. 2022)</notebooks/E_3dmodeling/5_3DSpectra.py>
   Modeling a Thermal Phase Curve pt 1 (Robbins-Blanch et al. 2022)</notebooks/E_3dmodeling/6_PhaseCurves.py>
   Modeling a Thermal Phase Curve pt 2 (Robbins-Blanch et al. 2022)</notebooks/E_3dmodeling/7_PhaseCurves-wChemEq.py>
   Modeling a Reflected Light Phase Curve (Hamill et al. 2024)</notebooks/E_3dmodeling/8_ReflectedPhaseCurve.py>


Fitting models to data
----------------------

.. toctree::
   :maxdepth: 1

   Grid Search Analysis </notebooks/F_fitdata/1_GridSearch.py>


Opacities
---------

.. toctree::
   :maxdepth: 1

   Query Opacites </notebooks/G_opacities/1_QueryOpacities.py>
   Opacity Factory: Creating Custom Databases </notebooks/G_opacities/2_CreatingOpacityDb.py>
   What Resampling do I Need? </notebooks/G_opacities/3_ResamplingOpacities.py>
   Using Correlated-K Tables </notebooks/G_opacities/4_CorrelatedKTables.py>


Radiative Transfer Techniques 
-----------------------------

.. toctree::
   :maxdepth: 1

   Toon Radiative Transfer in Reflected Light </notebooks/H_radiativetransfer/1_AnalyzingApproximationsReflectedLightToon.py>
   Spherical Harmonics Radiative Transfer in Reflected Light </notebooks/H_radiativetransfer/2_AnalyzingApproximationsReflectedLightSH.py>
   Spherical Harmonics Radiative Transfer in Thermal Emission </notebooks/H_radiativetransfer/3_AnalyzingApproximationsThermal.py>


Useful Tools
------------

.. toctree::
   :maxdepth: 1
   How to store and reuse models </notebooks/I_usefultools/ModelStorage.py>
   Spectral and Molecular Contribution Functions </notebooks/I_usefultools/ContributionFunctions.py>
   Common Climate Issues </notebooks/I_usefultools/CommonClimateBDIssues.py>
   Integrated Level Fluxes </notebooks/I_usefultools/Level_Fluxes.py>
   Misc FAQs </notebooks/I_usefultools/FAQs.py>
   SQLITE Tutorial </notebooks/I_usefultools/Sqlite3Tutorial.py>
   Data Storage Uniformity </notebooks/I_usefultools/data_uniformity_tutorial.py>

References
----------

.. toctree::
   :maxdepth: 2

   Citation Tools </notebooks/References.py>

