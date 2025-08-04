Tutorials
=========

Getting Set Up After Installation
---------------

.. toctree::
   :maxdepth: 1

   Installation Instructions </notebooks/0_GetDataFunctions.ipynb>

Basics of Spectral Modeling 
---------------------------

.. toctree::
   :maxdepth: 1

   Getting Started </notebooks/A_basics/1_GetStarted.ipynb>
   Simple Clouds  </notebooks/A_basics/2_AddingClouds.ipynb>
   Surface Reflectivity </notebooks/A_basics/3_AddingSurfaceReflectivity.ipynb>
   Plot Diagnostics </notebooks/A_basics/4_PlotDiagnostics.ipynb>
   Thermal Emission Spectroscopy </notebooks/A_basics/5_AddingThermalFlux.ipynb>
   Transmission Spectroscopy </notebooks/A_basics/6_AddingTransitSpectrum.ipynb>
   Brown Dwarf Spectroscopy </notebooks/A_basics/7_BrownDwarfs.ipynb>

Chemistry
---------

.. toctree::
   :maxdepth: 1

   Chemical Equilibrium & Disequilibrium Hacks </notebooks/B_chemistry/1_ChemicalEquilibrium.ipynb>
   Full Kinetics/Photochemistry  </notebooks/A_basics/2_Photochemistry.ipynb>
 

Clouds
------

.. toctree::
   :maxdepth: 1

   Virga (Ackerman & Marley Clouds) </notebooks/C_clouds/1_PairingPICASOToVIRGA.ipynb>
   Patchy Clouds </notebooks/C_clouds/2_PatchyClouds.ipynb>

1D Climate Modeling
-------------------
Relevant Citatons: `Mukherjee et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022arXiv220807836M/abstract>`_

.. toctree::
   :maxdepth: 1

   Brown Dwarfs </notebooks/D_climate/1_BrownDwarf_PreW.ipynb>
   Planet </notebooks/D_climate/2_Exoplanet_PreW.ipynb>
   Planet w/ Photochemistry </notebooks/D_climate/3_Exoplanet-Photochemistry.ipynb>
   Brown Dwarfs w/ Disequilibrium Chemistry (Self-Consistent Kzz) </notebooks/D_climate/4_BrownDwarf_DEQ_SC_kzz.ipynb>
   Brown Dwarfs w/ Disequilibrium Chemistry (Constant Kzz) </notebooks/D_climate/4b_BrownDwarf_DEQ_const_kzz.ipynb>
   Brown Dwarfs w/ Clouds </notebooks/D_climate/5_CloudyBrownDwarf_PreW.ipynb>
   Brown Dwarfs w/ Clouds and Disequilibrium Chemistry </notebooks/D_climate/6_CloudyBrownDwarf_DEQ.ipynb>
   Creating a grid of models </notebooks/D_climate/7_CreateModelGrid.ipynb>
   Brown Dwarfs w/ Energy Injection </notebooks/D_climate/8_EnergyInjection.ipynb>

3D Spectra and Phase Curves
---------------------------
Relevant Citatons: `Adams et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022ApJ...926..157A/abstract>`_ for 3D spectra and `Robbins-Blanch et al. 2022 <http://arxiv.org/abs/2204.03545>`_ for Phase Curves. 

.. toctree::
   :maxdepth: 1

   Non-Zero Phase and Spherical Integration </notebooks/E_3dmodeling/1_SphericalIntegration.ipynb>
   Basics of a 3D Calculation </notebooks/E_3dmodeling/2_3DInputsWithPICASOandXarray.ipynb>
   Post-Processing Chemistry for 3D runs </notebooks/E_3dmodeling/3_PostProcess3Dinput-Chemistry.ipynb>
   Post-Processing Clouds for 3D runs </notebooks/E_3dmodeling/4_PostProcess3Dinput-Clouds.ipynb>
   Modeling a 3D Spectrum (Adams et al. 2022)</notebooks/E_3dmodeling/5_3DSpectra.ipynb>
   Modeling a Thermal Phase Curve pt 1 (Robbins-Blanch et al. 2022)</notebooks/E_3dmodeling/6_PhaseCurves.ipynb>
   Modeling a Thermal Phase Curve pt 2 (Robbins-Blanch et al. 2022)</notebooks/E_3dmodeling/7_PhaseCurves-wChemEq.ipynb>
   Modeling a Reflected Light Phase Curve (Hamill et al. 2024)</notebooks/E_3dmodeling/8_ReflectedPhaseCurve.ipynb>


Fitting models to data
----------------------

.. toctree::
   :maxdepth: 1

   Grid Search Analysis </notebooks/F_fitdata/1_GridSearch.ipynb>
   Basics of Retrievals </notebooks/F_fitdata/2_GridRetrieval.ipynb>
   Basics of Grid fitting </notebooks/F_fitdata/3_GridRetrieval_GridFitting.ipynb>
   Basics of Grid-trievals </notebooks/F_fitdata/4_GridRetrieval_GridFittingWithClouds.ipynb>
   Creating retrieval templates </notebooks/F_fitdata/5_GridRetrieval_CreatingTemplates.ipynb>


Opacities
---------

.. toctree::
   :maxdepth: 1

   Query Opacites </notebooks/G_opacities/1_QueryOpacities.ipynb>
   Opacity Factory: Creating Custom Databases </notebooks/G_opacities/2_CreatingOpacityDb.ipynb>
   What Resampling do I Need? </notebooks/G_opacities/3_ResamplingOpacities.ipynb>
   Using Correlated-K Tables </notebooks/G_opacities/4_CorrelatedKTables.ipynb>


Radiative Transfer Techniques 
-----------------------------

.. toctree::
   :maxdepth: 1

   Toon Radiative Transfer in Reflected Light </notebooks/H_radiative_transfer/1_AnalyzingApproximationsReflectedLightToon.ipynb>
   Spherical Harmonics Radiative Transfer in Reflected Light </notebooks/H_radiative_transfer/2_AnalyzingApproximationsReflectedLightSH.ipynb>
   Spherical Harmonics Radiative Transfer in Thermal Emission </notebooks/H_radiative_transfer/3_AnalyzingApproximationsThermal.ipynb>


Useful Tools
------------

.. toctree::
   :maxdepth: 1
   How to store and reuse models </notebooks/I_usefultools/ModelStorage.ipynb>
   Spectral and Molecular Contribution Functions </notebooks/I_usefultools/2_ContributionFunctions.ipynb>
   Data Storage Uniformity </notebooks/I_usefultools/3_data_uniformity_tutorial.ipynb>
   SQLITE Tutorial </notebooks/I_usefultools/4_Sqlite3Tutorial.ipynb>
   Misc FAQs </notebooks/I_usefultools/5_FAQs.ipynb>
   Common Climate Issues </notebooks/I_usefultools/6_CommonClimateBDIssues.ipynb>
   Integrated Level Fluxes </notebooks/I_usefultools/7_Level_Fluxes.ipynb>

References
----------

.. toctree::
   :maxdepth: 2

   Citation Tools </notebooks/References.ipynb>

