#!/usr/bin/env python

# This sample setup.py can be used as a template for any project using d2to1.
# Simply copy this file and, if desired, delete all the comments.  Also remove
# the 'namespace_packages' and 'packages' arguments to setup.py if this project
# does not contain any packages beloning to a namespace package.

# This import statement attempts to import the setup() function from setuptools
# (this replaces the setup() one uses from distutils.core when using plain
# distutils).
#
# If the import fails (the user doesn't have the setuptools package) it then
# uses the ez_setup bootstrap script to install setuptools, then retries the
# import.  This is common practice for packages using setuptools.
try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup

# The standard setup() call.  Notice, however, that most of the arguments
# normally passed to setup() are absent.  They will instead be read from the
# setup.cfg file using d2to1.
#
# In order for this to work it is necessary to specify setup_requires=['d2to1']
# If the user does not have d2to1, this will boostrap it.  Also require
# stsci.distutils to use any of the common setup_hooks included in
# stsci.distutils (see the setup.cfg for more details).
#
# The next line, which defines namespace_packages and packages is only
# necessary if this projet contains a package belonging to the stsci namespace
# package, such as stsci.distutils or stsci.tools.  This is not necessary for
# projects with their own namespace, such as acstools or pyraf.
#
# d2to1=True is required to enable d2to1 and read the remaning project metadata
# from the setup.cfg file.
#
# use_2to3 and zip_safe are common options support by setuptools; these can
# also be placed in the setup.cfg, as will be demonstrated in a future update
# to this sample package.
setup(
    name='picaso', 
    version = '2.2',
    description = 'planetary intesity code for atmospheric scattering observations',
    long_description = 'README.rst',
    author = 'Natasha E. Batalha',
    author_email = 'natasha.e.batalha@gmail.com',
    url = 'https://natashabatalha.github.io/picaso',
    license = 'GPL-3.0',
    download_url = 'https://github.com/natashabatalha/picaso',
    classifiers = [
                  'Intended Audience :: Science/Research',
                  'License :: OSI Approved :: BSD License', 
                  'Operating System :: OS Independent' , 
                  'Programming Language :: Python',
                  'Programming Language :: Python :: 3',
                  'Topic :: Scientific/Engineering :: Astronomy',
                  'Topic :: Software Development :: Libraries :: Python Modules'
  ],
  packages=['picaso'],
  install_requires=[
          'numpy',
          'bokeh<2.4.0',
          'holoviews',
          'numba',
          'pandas',
          'joblib',
          'photutils',
          'astropy>=3.2.3',
          'matplotlib',
          'pysynphot',
          'sphinx',
          'scipy',
          'virga-exo'
          ], 
    zip_safe = False,
)
