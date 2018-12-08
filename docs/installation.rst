Installation
============

Install with Pip
----------------

.. code-block:: bash 

	pip install picaso

Install with Conda
------------------

.. code-block:: bash 

	conda config --add channels conda-forge
	conda install picaso

Install Dev Version with Git
----------------------------

.. code-block:: bash 

	git clone https://github.com/natashabatalha/picaso.git
	cd picaso
	python setup.py develop 

Download and Link Reference Documentation
-----------------------------------------

Download `Reference Data Here <https://natashabatalha.github.io>`_. 

.. code-block:: bash

	vi ~/.bash_profle

Add add this line:

.. code-block:: bash

	export picaso_refdata="/path/to/picaso/reference/data"

Should look something like this 

.. code-block:: bash

	cd /path/to/picaso/reference/data
	ls
	base_cases	config.json	opacities