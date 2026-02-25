.. :changelog:

History
-------

4.0 (2026-2-25)
~~~~~~~~~~~~~~~~
Major functionality:

* Cloudy equilibrium
* Cloudy/clear photochemistry
* Patchy cloud climates
* New correlated-k format (bin files -> hdf5) and computed from OS code within picaso
* Run photochem to create a spectrum
* Correlated-K forward modeling
* Patchy clouds in forward modeling
* Some integration testing

Enhancements:

* Climate model has been completely restructured to improve workflow and readability (user input to climate code has changed dramatically for diseq and cloudy runs)
* Level fluxes output in forward modeling for diagnostics
* Leave one out climate modeling for individual molecule contributions
* New get data tutorial auto downloads data that you need incl sonora grids, ck tables, virga files, stellar files, opacities 
* Deprecates pysynphot in place of stsynphot
* Opacity files starting with v3 now has zenodo DOI in meta data.

3.3 (2024-11-12)
~~~~~~~~~~~~~~~~
* Reflected light phase curves 

3.2 (2024-7-8)
~~~~~~~~~~~~~~
* Disequilibrium climate modeling

3.1 (2023-3-23)
~~~~~~~~~~~~~~~
* SH harmonics documentation for reflected light 
* SH harmonics documentation for thermal emission 
* Added output xarray resources for model preservation and reusibility

3.0 (2022-10-25)
~~~~~~~~~~~~~~~~
* First version of climate code (Mukherjee et al. 2022)
* First version of grid fitting tools (JWST Transiting Exoplanet ERS et al. 2022)
* Code help for xarrays
* Reference/citation tools for opacities
* Resampling tutorials
* CK tutorials 
* Contribution functions 
* Better plotting tools 

2.3 (2022-5-27)
~~~~~~~~~~~~~~~
* fixes correlated k 3d runs 
* forces ref pressure to be in user grid for less sensitivity to layers
* reference pressure in docs  

2.3 (2022-4-06)
~~~~~~~~~~~~~~~
* lots of fun 3D functionality (Adams et al 2022)
* phase curves (Robbins-Blanche et al 2022)
* 3D tutorials with xarray
* clearer radiative transfer to be compatible with Toon 89
* Faster chemeq interpolation 
* opacity tables pull four nearest neighbors (though option for 1 remains)
* Improvements and additions to plotting functionality 
* Minor bug fixes  

2.2 (2021-7-12)
~~~~~~~~~~~~~~~~~~
* Add evolution tracks 
* Add ability to use pre mixed c-k tables 
* Expand chemistry to include new Visscher tables 
* Add ability to pull out contribution from individual species without running full RT
* Young planet table from ZJ Zhang. 
* Separate workshop notebooks for Sagan School 2020, 2021 and ERS 
* Add explicit "hard surface" term for thermal flux boundary condition for terrestrial calculations
* Minor bug fixes/improvements 

2.1 (2020-11-02)
~~~~~~~~~~~~~~~~~~

* Transit Spectroscopy added 
* Transit spectroscopy tutorial 
* FAQ notebook 
* Minor bug fixes/improvements

2.0.1 (2020-07-21)
~~~~~~~~~~~~~~~~~~

* Ability to load targets from exoplanet archive 

2.0 (2020-04-21)
~~~~~~~~~~~~~~~~~~

* Explicit Brown Dwarf tutorials 
* Coupling to brand new python cloud code `Virga`!!
* Ability to specify wave range in `opannection` and ability to resample (consult your local theorist before doing this)
* Added "Opacity Factor" so that users can easiy query opacity data without going through SQL 
* Removed opacity from git-lfs (was a bad system and users were having trouble)
* Fixed critical spherical integration bug and added more robust steps for 1d vs. 3d. Also added full tutorial to explain differences here. 
* Added notebook for surface reflectivity

1.0 (2020-2-7)
~~~~~~~~~~~~~~

* Thermal emission added 
* Changed the way the code output is returned to single dictionary. 
* Tutorials to map 3D input onto Gauss/Chebysev Angles 

0.0 (2019-4-19)
~~~~~~~~~~~~~~~

* Reflected light component only 
* added git-lfs opacity database from only 0.3-1 micron 
* made opacity database sqlite to speed up queries 
* tutorials added for initial setup, adding clouds, visualizations, analyzing approximations, and opacity queries
