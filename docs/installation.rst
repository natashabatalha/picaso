Installation
============

Python  Version
---------------

Python >= 3.11 

It is recommended you use Conda environments. Please check out `our conda environment tutorial <https://natashabatalha.github.io/picaso/contribution.html#using-conda-enviornments>`_.  

Install with Git (recommended)
------------------------------

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

PICASO uses a lot of different kinds of reference data. The most important is the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_ and the second most important is the resampled opacity file which will let you do some basic spectral modeling.

+----------------------------------+------+-------------------------------+-------------------------------------------------------------+
| Data Type                        | Req? | What it is primarily used for | Where it should go                                          |
+==================================+======+===============================+=============================================================+
| Reference                        | Yes  | everything                    | $picaso_refdata                                             |
| Resampled Opacities              | Yes  | Spectroscopic modeling        | $picaso_refdata/opacities/opacities.db                      |
| Stellar Database                 | No   | Exoplanet modeling            | $PYSYN_CDBS/grid                                            |
| Preweighted correlatedK Tables   | No   | Chemical equilibrium climate  | Your choice (default=$picaso_refdata/opacities/preweighted) |
| By molecule correlatedK Tables   | No   | Disequilibrium climate        | Your choice (default=$picaso_refdata/opacities/resortrebin) |
| Sonora grid models               | No   | Initial guess/grid fitting    | Your choice (default=$picaso_refdata/sonora_grids)          |
| Virga Mieff files                | No   | Virga cloud modeling          | Your choice (default=$picaso_refdata/virga)                 |
+----------------------------------+------+-------------------------------+-------------------------------------------------------------+

To make path handling simpler, picaso relies on the user setting a basic "environment variable" called ``$picaso_refdata``. 
A Python environment variable created and stored outside of your script that your program can access to get absolute path information. This makes it so that while you are running the code you dont have 
to constantly set paths. 

As a basic example, my ``$picaso_refdata`` path looks like this: ``'/Users/nbatalh1/Documents/codes/PICASO/picaso/reference'`` 
and includes the basic contents of this the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_. 


Create PICASO Environment Variable [Mandatory]
----------------------------------------------

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

If you have already downloaded reference data you can check that your variable has been defined properly: 

.. code-block:: bash

	echo $picaso_refdata
	/Users/nbatalh1/Documents/codes/PICASO/picaso/reference
	cd $picaso_refdata
	ls
	base_cases chemistry config.json evolution opacities version.md


Method 2: Add directly to python code
````````````````````````````````````````

Sometimes it is too troublesome to go through bash settings and you may prefer to set it directly in your python code. 

.. code-block:: python

	import os
	os.environ['picaso_refdata'] = 'your_path' #THIS MUST GO BEFORE YOUR IMPORT STATEMENT
	#if you are using stellar grid models
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


Download PICASO Reference Data
------------------------------

Option 1) Here is how you would do in python after your environment variable is already set: 

.. code-block:: python

        #using python
	import picaso.data as d
	d.get_reference(d.os.environ['picaso_refdata'])

Option 2) You can also do it manually by downloading the `Reference Folder from Github <https://github.com/natashabatalha/picaso/tree/master/reference>`_. You should already this if you did a Git clone. **Make sure that your reference folder matches the version number of ``PICASO``**. Check the version number in the file ``reference/version.md``. 


Download Resampled Opacities 
----------------------------

.. note::
	Note you can use ``picaso.data.get_data()`` to help you downloaded opacities since there are a few different databases available (see tutorial below). Any is acceptable as the default depending on your resolution and wavelength needs. Only one opacities.db file needs to exist in your referece/opacities folder. The others you just specify a path to in ``picaso.justdoit.opannection``. 

Option 1: Here is how you would do in python:

.. code-block:: python

	import picaso.data as d
	d.get_data(category_download='resampled_opacity', target_download='default')

Option 2: Here is how you would do it manually:

1) Download a the recommended default `Resampled Opacity File from Zenodo <https://zenodo.org/records/14861730>`_. 
2) Once this is download rename and add it to ``reference/opacities/opacities.db``. You will likely have to change the name of the file. ultimately, PICASO is looking for the default file reference/opacities/opacities.db. 

Create ``stsynphot`` Environment Variable for Stellar Data if needed
--------------------------------------------------------------------

In order to get stellar spectra needed for many exoplanet use cases you will have to install the `stsynphot package <https://stsynphot.readthedocs.io/en/latest/>`_ which has several download options for stellar spectra. This package also requires setting an environment variable called ``$PYSYN_CDBS``. Below are instructions for getting the stellar data manually and setting the environment variable (note you can also use picaso auto download to get these data). 

Download Stellar Data
`````````````````````

The Defulat for ``PICASO`` is Castelli-Kurucz Atlas: `ck04models <https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/>`_ but we recommend ``pheonix`` models for climate modeling. 

Option 1: Use PICASO python

.. code-block:: python

	import picaso.data as d
	d.get_data(category_download='stellar_grids') #recommend downloading both ck04models and phoenix


Option 2: You can use `wget` you can download them by doing this or just downloading the link below. 

.. code-block:: bash

	wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz

When you untar this you should get a directory structure that looks like this ``<path>/grp/redcat/trds/grid/ck04models``. Some other people have reported a directory structure that looks like this ``<path>/grp/hst/cdbs/grid/ck04models``. **The full directory structure does not matter**. Only the last portion ``grid/ck04models``. You will need to create an enviornment variable that points to where ``grid/`` is located. We have a nice placeholder location in the picaso reference data file for these grids ``$picaso_refdata/stellar_grids``. Though it is not required for you to put them here as long as you make your environment variable point to the desired location. 


Set Stellar Data Environment Variable
``````````````````````````````````````

Follow the same environment variable above instructions to create an environment variable. For example:

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

Where the folder ``grid/`` contains whatever ``stsynphot`` data files you have downloaded (e.g. a folder called ``ck04models/``). 






Check Environment and Download Data Helper
==========================================

.. toctree::
   :maxdepth: 1

   Simple Data Grabbing </notebooks/0_GetDataFunctions.ipynb>
