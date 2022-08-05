Contributing to ``PICASO``
==========================

PEP 8 Style Guide
-----------------

We generally follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/#descriptive-naming-styles>`_ styling. Below we emphasize the "must haves" of our code. 

- Your code should read like a book. If you have ~10 lines of code without a comment you are commenting too infrequently.
- It is really important to style function headers uniformly using `NumPy style Python Docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_. This enables `sphinx <http://www.sphinx-doc.org/en/master/>`_ to auto read function headers. Below is an example, but `look here <https://numpydoc.readthedocs.io/en/latest/format.html#sections>`_ for a full list of headers.

.. code-block:: python

	def foo(in1, in2,in3=None):
	    """
	    Describe function thoroughly. 

	    Add any relevant citations. 

	    Parameters
	    ----------
	    in1 : int 
	        This variable is for something cool.
	    in2 : list of float
	        This variable is for something else cool.
	    in3 : str,optional
	        (Optional) Default=None, this variable is options.

	    Returns
	    -------
	    int
	        Cool output
	    float 
	    	Other cool output

	    Examples
	    --------
	    This is how to use this. 

	    >>a = foo(5,[5.0,4.0],in3='hello')


	    Warnings
	    --------
	    Garbage in garbage out
	    """

- Variable names should explain what the variable is. Avoid undescriptive names like `thing1` or `thing2`
- If you have an equation in your code it should be referenced with a paper and equation number. For example:

.. code-block:: python

	#now define terms of Toon et al 1989 quadrature Table 1 
	#https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
	#see table of terms 
	sq3 = sqrt(3.)
	g1	= (sq3*0.5)*(2. - w0*(1.+cosbar))	#table 1
	g2	= (sq3*w0*0.5)*(1.-cosbar)		   #table 1
	g3	= 0.5*(1.-sq3*cosbar*ubar0)		  #table 1
	lamda = sqrt(g1**2 - g2**2)					 #eqn 21
	gama  = (g1-lamda)/g2							  #eqn 22


Github Workflow
---------------

Before contributing, consider submitting an issue request. Sometimes we may already be aware of an issue and can help you fix something faster. 

1) Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^

Clone the repository that you are interested in working on.

.. code-block:: bash

	git clone https://github.com/natashabatalha/picaso.git

This will download a copy of the code to your computer. You will automatically be in the ``master`` branch upon downloading. You can track the dev branch like so:

.. code-block:: bash

	git checkout dev 

**Side note: Important distinction between ``master`` and ``dev``**

``master`` always represents the released production code. Here is the workflow we will follow. All major development will be done on branches off of ``dev``. The only exceptions are what we call "hotfixes", which can go directly from the fixed branch to master, and minor bugs that can be directly fixed on ``dev``. See the overall schematic below.

.. image:: github_flow.jpg


2) Create a branch off of ``dev`` with a useful name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's likely you will be working on a specific subset of a bigger code project. Any changes you make on a new branch will not affect ``master`` or ``dev``, so you can feel free to beat up the code without damaging anything that is fully tested.

.. code-block:: bash

	git checkout -b myfeature dev


3) Work work work work...
^^^^^^^^^^^^^^^^^^^^^^^^^
Let's pretend that ``myfeature`` entails working on ``file1.py`` and ``file2.py``. After you are happy with an initial change, commit and push your changes.

.. code-block:: bash

	#commit changes
	git add file1.py file2.py
	git commit -m 'added cool thing'

	#switch to dev branch
	git checkout dev 

	#merge your changes
	git merge --no-ff myfeature

	#delete old branch
	git branch -d myfeature 

	#push to dev
	git push origin dev


Many people ask: "How often should I commit??". Choose something that works for you and stick to it. I try and work on smaller, individual tasks and commit when I feel I have finished something. If you try and do too much at once, your commit comments won't make too much sense with what you have actually done. Remember, eventually someone will have to review your commits. If they are hard to parse, it will delay the merge of your work.

4) Final merge to ``master``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``master`` is generally a protected branch, so talk to the admin or the team before proceeding. In general, merges to master are easiest done through `Github Online <https://github.com/natashabatalha/picaso>`_. Near where the branches are listed, go to "New Pull Request". Write a description of the new dev capability, and request a merge to master. And if all good then, done!!! 

Using Conda Enviornments
------------------------

Package control and version control is a pain. To make sure everyone is running on the same enviornment it will be beneficial if we are all work in the same environment. Here are the most pertinent commands you need to know. 

Create your own environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^
To create your own environment with a specific name and python package:

.. code-block:: bash

	conda create --name your_env_name python=3.7 -y


If you have specific environment variables that need to be tied to here, then you can specify them. For example, in PICASO there is the environment variable ``picaso_refdata`` and ``PYSYN_CDBS``: 

.. code-block:: bash

	conda activate your_env_name
	cd $CONDA_PREFIX
	mkdir -p ./etc/conda/activate.d
	mkdir -p ./etc/conda/deactivate.d
	touch ./etc/conda/activate.d/env_vars.sh
	touch ./etc/conda/deactivate.d/env_vars.sh


Edit ``./etc/conda/activate.d/env_vars.sh``

.. code-block:: bash

	#!/bin/sh

	export MY_VAR='path/to/wherever/you/need'


And edit ``./etc/conda/deactivate.d/env_vars.sh``

.. code-block:: bash

	#!/bin/sh

	unset MY_VAR

No whenever you activate your environment, your variable name will be there. Whenever you deactivate your environment, it will go away. 

Export your environment
^^^^^^^^^^^^^^^^^^^^^^^
Another great aspect of conda enviornments is that they can be passed to one another. 

.. code-block:: bash

	conda env export > my_environment.yml

Create enviornment from a ``.yml`` file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If someone passes you an environment file, you can easily create an environment from it ! 

.. code-block:: bash

	conda env create -f environment.yml







