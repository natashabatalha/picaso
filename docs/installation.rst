Installation
============

Python Version
--------------

Python >= 3.8 

Install with Git (recommended)
------------------------------

The Github repository contains the reference folder and helpful tutorials.  

.. code-block:: bash 

	git clone https://github.com/natashabatalha/picaso.git
	cd picaso
	python setup.py install 

Install with Pip
----------------

.. code-block:: bash 

	pip install picaso

With a pip install you will need to download the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_ (explained below). This can simply be done by downloading a zip of the ``PICASO`` code from Github (which does not require git setup if that is not available to you). 


Download PICASO Reference Data
------------------------------

.. note::
	`PICASO` 3.0 will not work with PICASO 2.3 reference folder. Please download the new reference folder if you are using PICASO 3.0 

1) Download the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_. You should already this if you did a Git clone. **Make sure that your reference folder matches the version number of ``PICASO``**. Check the version number in the file ``reference/version.md``. 

2) Download a `Resampled Opacity File from Zenodo <https://doi.org/10.5281/zenodo.3759675>`_. Note that there are a few different versions. Any is acceptable depending on your resolution and wavelength needs. Put this db file in the `Opacities reference Folder you downloaded from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_. **You may need to rename it opacities.db and place into `reference/opacities` folder**. The file placed into reference/opacities will serve as your default file. You can point to other opacity files using justdoit.opannection. 


Create Environment Variable 
---------------------------

There are several ways to create environment variables. Below are the three most popular methods. You only need to choose one that works best for you. 

Method 1: ``bash_profile`` or ``zshrc`` file
````````````````````````````````````````````

As you might guess ``~/.bash_profile`` is used for the ``Bash`` command line shell. ``~/.zshrc`` is used for the ``Zsh`` command line shell. The steps below are identical.

.. code-block:: bash

	vi ~/.bash_profile

Add add this line:

.. code-block:: bash

	export picaso_refdata="/path/to/picaso/reference/"

Once you edit a bash profile file, you must source it. Alternatively you can open up a new terminal. 

.. code-block:: bash

	source ~/.bash_profile

Now you can check that your variable has been defined properly: 

.. code-block:: bash

	echo $picaso_refdata
	/Users/nbatalh1/Documents/codes/PICASO/picaso/reference
	cd $picaso_refdata
	ls
	base_cases chemistry config.json evolution opacities version.md

Your opacities folder shown above should include the file ``opacities.db`` `file downloaded from zenodo <https://doi.org/10.5281/zenodo.3759675>`_. This is mostly a matter of preference, as PICASO allows you to point to an opacity directory. Personally, I like to store something with the reference data so that I don't have to constantly specify a folder path when running the code. 

Method 2: Add directly to python code
````````````````````````````````````````

Sometimes it is too troublesome to go through bash settings and you may prefer to set it directly in your python code. 

.. code-block:: python

	import os
	os.environ['picaso_refdata'] = 'your_path' #THIS MUST GO BEFORE YOUR IMPORT STATEMENT
	os.environ['PYSYN_CDBS'] = 'your_path' #this is for the stellar data discussed below.
	import picaso.justdoit as jdi

Method 3: Add it to your conda enviornment
````````````````````````````````````````````

This is my method of choice! It involves creating conda environment specific variables. If you are interested in learning more about environment variables, you can `read more about them here <https://natashabatalha.github.io/picaso/contribution.html#using-conda-enviornments>`_

If you already an evironment setup, you can do the following -- which mimics the `bash_profile/method 1` example.  

.. code-block:: bash

	conda activate your_env_name
	cd $CONDA_PREFIX
	mkdir -p ./etc/conda/activate.d
	mkdir -p ./etc/conda/deactivate.d
	touch ./etc/conda/activate.d/env_vars.sh
	touch ./etc/conda/deactivate.d/env_vars.sh

The ``env_vars.sh`` file is similar to your ``bash_profile`` file. Therefore you can directly add your export statement there. 

.. code-block:: bash 

	vi ./etc/conda/activate.d/env_vars.sh

Now add the line: 

.. code-block:: bash 

	export picaso_refdata="/path/to/picaso/reference/"

Finally, you want to make sure that your environment variable is unset when you deactivate your environment. 

.. code-block:: bash 

	vi ./etc/conda/deactivate.d/env_vars.sh

.. code-block:: bash 
	
	unset picaso_refdata

Notice here that I do **not** have a tilda (~) in front of ``./etc``. The full path of the ``env_vars.sh`` should look something like this: 

.. code-block:: bash 

	conda activate your_environment
	cd $CONDA_PREFIX
	cd ./etc/conda/activate.d/
	pwd
	/Users/nbatalh1/.conda/envs/picaso/etc/conda/activate.d


Download and Link Pysynphot Stellar Data
----------------------------------------

In order to get stellar spectra you will have to download the stellar spectra here from PySynphot: 

1) PICASO uses the `Pysynphot package <https://pysynphot.readthedocs.io/en/latest/appendixa.html>`_ which has several download options for stellar spectra. The Defulat for ``PICASO`` is Castelli-Kurucz Atlas: `ck04models <https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/>`_. 

You can download them by doing this: 

.. code-block:: bash

	wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz

When you untar this you should get a directory structure that looks like this ``<path>/grp/redcat/trds/grid/ck04models``. Some other people have reported a directory structure that looks like this ``<path>/grp/hst/cdbs/grid/ck04models``. **The full directory structure does not matter**. Only the last portion ``grid/ck04models``. You will need to create an enviornment variable that points to where ``grid/`` is located. See below.


2) Create environment variable via bash 

.. code-block:: bash

	vi ~/.bash_profile

Add add this line:

.. code-block:: bash

	export PYSYN_CDBS="<your_path>/grp/redcat/trds"

Then always make sure to source your bash profile after you make changes. 

.. code-block:: bash

	source ~/.bash_profile

Now you should be able to check the path:

.. code-block:: bash

	cd $PYSYN_CDBS
	ls
	grid

Where the folder ``grid/`` contains whatever ``pysynphot`` data files you have downloaded (e.g. a folder called ``ck04models/``). 

.. note::

	1. STScI serves these files in a few different places, with a few different file structures. **PySynphot only cares that the environment variable points to a path with a folder called `grid`. So do not worry if `grp/hst/cdbs` appears different.** 

