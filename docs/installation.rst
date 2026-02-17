


Quickstart
==========

If you want to get started quickly and are already familiar with many coding principles (pip, conda, environment variables, etc) then feel free to use our quickstart otherwise skip and read through the installation. 

.. toctree::
   :maxdepth: 1

   Simple Install </notebooks/Quickstart.ipynb>

Installation
============

Python >= 3.11 is recommended. It is also recommended you use environments (either conda or pip). Please check out `our conda environment tutorial <https://natashabatalha.github.io/picaso/contribution.html#using-conda-enviornments>`_.  

Users can 1) install from source, 2) install from pip, 3) install from conda. 

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

With a pip or conda install you will need to download the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_ (explained below). This can simply be done by downloading a zip of the ``PICASO`` code from Github (which does not require git setup if that is not available to you). 


Reference Data 
==============

PICASO has the ability to injest lots of different kinds of data. But not all are required. Only two are required: 1) The `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_ and 2) the resampled opacity file which will let you do some basic spectral modeling. 

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



Picaso relies on the user setting a basic "environment variable" called ``$picaso_refdata``. We recommend  **first** setting defining this environment variable and then using the PICASO donwload tools to automatically grab everything you need. 

A Python environment variable is a variable that is created and stored outside of your script that your program can access to get absolute path information. This makes it so that while you are running the code you dont have 
to constantly set paths. ``$picaso_refdata`` will point to the basic contents of the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_

As a basic example, my ``$picaso_refdata`` path looks like this: ``'/Users/nbatalh1/Documents/codes/PICASO/picaso/reference'``. If I check inside this path I will see the basic contents of the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_. 


Create PICASO Environment Variable
----------------------------------

There are several ways to create environment variables. Below are the three most popular methods. You only need to choose one that works best for you. If you have cloned the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_ then the path you create below should point to here. 

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

If you have already downloaded reference data you can check that your variable has been defined properly: 

.. code-block:: bash

	echo $picaso_refdata
	/Users/nbatalh1/Documents/codes/PICASO/picaso/reference
	cd $picaso_refdata
	ls
	base_cases chemistry config.json evolution opacities version.md


Method 2: Add directly to python code (easiest)
```````````````````````````````````````````````

Sometimes it is too troublesome to go through bash settings and you may prefer to set it directly in your python code. 

.. code-block:: python

	import os
	os.environ['picaso_refdata'] = 'your_path' #THIS MUST GO BEFORE YOUR IMPORT STATEMENT
	#if you are using stellar grid models
	os.environ['PYSYN_CDBS'] = 'your_path' #this is for the stellar data discussed below.
	import picaso.justdoit as jdi

Method 3: Add it to your conda enviornment
```````````````````````````````````````````

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


Download PICASO Reference Data
------------------------------

If you haven't yet gotten the data yet from Git, you can do that now that you have your environment variable set.

Here is how you would do it in python after your environment variable is already set: 

.. code-block:: python

    #using python
	import picaso.data as d
	d.os.environ['picaso_refdata'] = "/path/to/picaso/reference/" #only needed if you are using python to set your environmnet variables otherwise you do not need this
	d.get_reference(d.os.environ['picaso_refdata'])

Check it's all good (you will see an error because we haven't downloaded the opacities that is expected): 

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



Download Resampled Opacities 
----------------------------

Let's use ``picaso.data.get_data()`` to help you downloaded opacities since there are a few different databases available (see tutorial below). Any of the dbs are acceptable as the default depending on your resolution and wavelength needs. Only one opacities*.db file needs to exist in your referece/opacities folder (note the only naming specification is that it start with "opacities" and end with "db" e.g., opacities*.db). The others you just specify using the keyword in ``picaso.justdoit.opannection``. 


.. code-block:: python

	import picaso.data as d
	d.os.environ['picaso_refdata'] = "/path/to/picaso/reference/" #only needed if you are using python to set your environmnet variables otherwise you do not need this
	d.get_data(category_download='resampled_opacity', target_download='default')

If that doesn't suit your needs feel free to do this manually: 

1) Download a the recommended default `Resampled Opacity File from Zenodo <https://zenodo.org/records/14861730>`_. 
2) Once this is download add it to ``reference/opacities/``. Note PICASO will look for something called "opacities*db". If youve added multiple files here it will choose the first one.

Create ``stsynphot`` Environment Variable for Stellar Data if needed
--------------------------------------------------------------------

In order to get stellar spectra needed for many exoplanet use cases you will have to install the `stsynphot package <https://stsynphot.readthedocs.io/en/latest/>`_ which has several download options for stellar spectra. This package also requires setting an environment variable called ``$PYSYN_CDBS``. Below are instructions for getting the stellar data manually and setting the environment variable (note you can also use picaso auto download to get these data). 

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


Additional Help with Stellar Data Environment Variable
``````````````````````````````````````````````````````

Follow the same environment variable above instructions to create an environment variable. As a default PICASO recommends the path: ``PYSYN_CDBS=$picaso_refdata/stellar_grids``. For example:

.. code-block:: bash

	vi ~/.bash_profile

Add add this line:

.. code-block:: bash

	export PYSYN_CDBS="/data/picaso_refdata/stellar_grids"

Then always make sure to source your bash profile after you make changes. 

.. code-block:: bash

	source ~/.bash_profile

Now you should be able to check the path:

.. code-block:: bash

	cd $PYSYN_CDBS
	ls
	grid

Where the folder ``grid/`` contains whatever ``stsynphot`` data files you have downloaded (e.g. a folder called ``ck04models/``). 


