Version 4.0 beta
----------------
Major functionality:
1. Cloudy equilibrium
2. Cloudy/clear photochemistry
3. Patchy cloud climates
4. New correlated-k format (bin files -> hdf5) and computed from OS code within picaso
5. Run photochem to create a spectrum
6. Correlated-K forward modeling
7. Patchy clouds in forward modeling
8. Some integration testing

Enhancements:
1. Climate model has been completely restructured to improve workflow and readability (user input to climate code has changed dramatically for diseq and cloudy runs)
2. Level fluxes output in forward modeling for diagnostics
3. Leave one out climate modeling for individual molecule contributions
4. New get data tutorial auto downloads data that you need incl sonora grids, ck tables, virga files, stellar files, opacities 
5. Deprecates pysynphot in place of stsynphot
6. Opacity files starting with v3 now has zenodo DOI in meta data.

Version 3.3
-----------
- Reflected light phase curves (Hamill et al. 2024) 

Version 3.2
-----------
- Disequilibrium Climate (Mukherjee et al. 2024) Elf-OWL models

Version 3.1 
--------------------------
- Spherical harmonics with reflected light (Rooney et al. 2023a)
- Spherical harmonics with thermal emission (Rooney et al. 2023b) 


Version 3.0
-----------
- First version of climate code (Mukherjee et al. 2022)
- First version of grid fitting tools (JWST Transiting Exoplanet ERS et al. 2022)
- Code help for xarrays
- Reference/citation tools for opacities
- Resampling tutorials
- CK tutorials 
- Contribution functions 
- Better plotting tools 
