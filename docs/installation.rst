Installation
============

There are a few avenues for installing PICASO (git, pip, or conda). To help determine which is best for you, we have a few questions to help you decide:
1. Are you a researcher who wants to stay up-to-date with the latest beta changes and contribute to the code base? Please follow the git clone instructions below.
2. Are you a student or researcher who just wants to get started with PICASO quickly and use it for science? Please follow the pip or conda install instructions below.
3. Do you want to use PICASO for an school assignment or workshop? `Jump to the picaso-lite section of the Quickstart Notebook </notebooks/Quickstart.py>`_


Python >= 3.11 is recommended. It is also recommended you use environments (either conda or pip). Please check out `our conda environment tutorial <https://natashabatalha.github.io/picaso/contribution.html#using-conda-enviornments>`_.  

Users can 1) install from source or 2) install from pip or 3) conda 

Install from source via git
---------------------------

The Github repository contains the reference folder and helpful tutorials.  

.. code-block:: bash 

	git clone https://github.com/natashabatalha/picaso.git
	cd picaso
	pip install .


Install with Pip
----------------

.. code-block:: bash 

	pip install picaso

Install with conda 
-------------------

.. code-block:: bash 

	conda install conda-forge::picaso


Reference Data 
==============

PICASO has the ability to injest lots of different kinds of data. But not all are required. **Only two are required: 1) The `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_ and 2) the resampled opacity file which will let you do some basic spectral modeling.**

+----------------------------------+------+-------------------------------+-------------------------------------------------------------+
| Data Type                        | Req? | What it is primarily used for | Where it should go                                          |
+==================================+======+===============================+=============================================================+
| Reference                        | Yes  | everything                    | $picaso_refdata                                             |
| Resampled Opacities              | Yes  | Spectroscopic modeling        | $picaso_refdata/opacities/opacities*.db                     |
| Stellar Database                 | No   | Exoplanet modeling            | $PYSYN_CDBS/grid                                            |
| Preweighted correlatedK Tables   | No   | Chemical equilibrium climate  | Your choice (default=$picaso_refdata/opacities/preweighted) |
| By molecule correlatedK Tables   | No   | Disequilibrium climate        | Your choice (default=$picaso_refdata/opacities/resortrebin) |
| Sonora grid models               | No   | Initial guess/grid fitting    | Your choice (default=$picaso_refdata/sonora_grids)          |
| Virga Mieff files                | No   | Virga cloud modeling          | Your choice (default=$picaso_refdata/virga)                 |
+----------------------------------+------+-------------------------------+-------------------------------------------------------------+

Get Required Data 
-----------------

.. code-block:: python

    #using python
	import picaso.data as d
	#where do you want to store your reference data? This should be the path to the reference folder you downloaded from github.
	path_to_reference = "/path/to/picaso/reference/"
	d.os.environ['picaso_refdata'] = path_to_reference #only needed if you are using python to set your environmnet variables otherwise you do not need this
	#get required data 1) reference data 
	d.get_reference(d.os.environ['picaso_refdata'])
	#get required data 2) resampled opacities
	d.get_data(category_download='resampled_opacity', target_download='default') #7Gb -- ENSURE STABLE WIFI! 

Check it's all good:

.. code-block:: python

	d.check_environ()

where the printout in a terminal will look something like this: 

.. code-block:: python

	--- PICASO Environment Check ---
	[INFO] Found picaso_refdata environment variable: /Users/nbatalh1/Documents/codes/PICASO/picaso/reference
	[SUCCESS] Basic picaso reference data seems to be in place.
	[INFO] Reference data version: 4.0
	[INFO] Files in reference directory:
	- input_tomls
	- references
	- stellar_grids
	- config.json
	- chemistry
	- sonora_grids
	- opacities
	- version.md
	- evolution
	- scripts
	- base_cases
	- virga
	- climate_INPUTS


Permanently Set Environment Variables
-------------------------------------

Setting environment variables help packages find data wihtout having to set a path everytime you run the code. 
Above we set the environment using `python`'s `os.environ` which is a temporary way to set environment variables. 

You can feel free to continue using this non-permanent. If so, whenever you install picaso you would need to add this line to your code **before** you import picaso.

.. code-block:: python

	import os
	os.environ['picaso_refdata'] = 'your_path/picaso/reference/' #THIS MUST GO BEFORE YOUR IMPORT STATEMENT
	import picaso.justdoit as jdi

While this is the easiest way to set things up, it can become tedious to have to define this before every single notebook. If you want to **permanently** set your environment variables, see the next to subsections for some options that may suit your needs. 

Permanent Method 1: ``bash_profile`` or ``zshrc`` file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As you might guess ``~/.bash_profile`` is used for the ``Bash`` command line shell. ``~/.zshrc`` is used for the ``Zsh`` command line shell. The steps below are identical.

.. code-block:: bash

	vi ~/.bash_profile

Add add this line:

.. code-block:: bash

	export picaso_refdata="/path/to/picaso/reference/"

Once you edit a bash profile file, you must source it. Alternatively you can open up a new terminal. 

.. code-block:: bash

	source ~/.bash_profile

If you have already downloaded reference data you can check that your variable has been defined properly: 

.. code-block:: bash

	echo $picaso_refdata
	/Users/nbatalh1/Documents/codes/PICASO/picaso/reference
	cd $picaso_refdata
	ls
	base_cases chemistry config.json evolution opacities version.md


Permanent Method 2: Add it to your conda enviornment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Getting Optional Reference Data 
-------------------------------

You can continue using `get_data` to get all additional data that you might need (see `Quickstart Notebook </notebooks/Quickstart.py>`_ for more examples). The most common one is the stellar grids for exoplanet modeling. The stellar grids are handled by another package called `stsynphot` and require the setting of an additional environment variable. Below we provide additional details on setting that up if needed. 


Getting ``stsynphot`` Optional Stellar Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to get stellar spectra needed for many exoplanet PICASO has already isntalled for you the `stsynphot package <https://stsynphot.readthedocs.io/en/latest/>`_ which has several download options for stellar spectra. This package also requires setting an environment variable called ``$PYSYN_CDBS``. Below are instructions for getting the stellar data manually and setting the environment variable (note you can also use picaso auto download to get these data). 

The Default for ``PICASO`` is Castelli-Kurucz Atlas: `ck04models <https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/>`_ but we recommend ``pheonix`` models for climate modeling. 

Here is how you would do it with PICASO:

.. code-block:: python

	import picaso.data as d
	d.os.environ['PYSYN_CDBS'] = d.os.path.join(d.os.environ['picaso_refdata'],'stellar_grids') #this is a path of your choosing
	d.get_data(category_download='stellar_grids') #recommend downloading both ck04models and phoenix

As the default this will store your stellar grids in ``$picaso_refdata/stellar_grids``. However, you can always choose a different path.

If you do not like that you can always use `wget` or just downloading the link below. 

.. code-block:: bash

	wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz

When you untar this you should get a directory structure that looks like this ``<path>/grp/redcat/trds/grid/ck04models``.  **The full directory structure does not matter**. Only the last file ``grid`` is needed. You will need to create an enviornment variable that points to where ``grid/`` is located. We have a nice placeholder location in the picaso reference data file for these grids ``$picaso_refdata/stellar_grids``. So you can imagine something like: ``$picaso_refdata/stellar_grids/grid/ck04models``. Though it is not required for you to put them here as long as you make your environment variable point to the desired location where ``grid`` is. 


Help Permanently Setting Stellar Data Environment Variable
``````````````````````````````````````````````````````````

Similar to PICASO's environment variable instructions, follow the same steps to create an environment variable for ``PYSYN_CDBS``. PICASO recommends the path: ``PYSYN_CDBS=$picaso_refdata/stellar_grids`` but you might have this elsewhere.

We've gone over how to set permanent paths before but here is a quick refresher using the bash_profile method:

.. code-block:: bash

	vi ~/.bash_profile

Add this line:

.. code-block:: bash

	export PYSYN_CDBS="/your/path/to/picaso_refdata/stellar_grids"

Then always make sure to source your bash profile after you make changes. 

.. code-block:: bash

	source ~/.bash_profile

Now you should be able to check the path:

.. code-block:: bash

	cd $PYSYN_CDBS
	ls
	grid

Where the folder ``grid/`` contains whatever ``stsynphot`` data files you have downloaded (e.g. a folder called ``ck04models/``). 


Using Quickstart Notebook
=========================

If you are already very familiar with package installs, environment variables **OR you are a student using PICASO for classwork you can jump straight** to our `Quickstart Notebook </notebooks/Quickstart.py>`_.

.. toctree::
   :maxdepth: 1
	Simple Install </notebooks/Quickstart.py>



