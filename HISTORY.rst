.. :changelog:

History
-------

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