#!/usr/bin/env python

# Import useful things
from distutils.core import setup
from setuptools import find_packages

#
setup(name='positive',
      version='1.0',
      description='Low Level Algorithms for Data Analysis and Manipulation',
      author='Lionel London',
      author_email='lionel.london@ligo.org',
      packages=find_packages(),
      include_package_data=True,
      package_dir={'positive': 'positive'},
      url='https://github.com/llondon6/positive',
      download_url='https://github.com/llondon6/positive/archive/master.zip',
      install_requires=['h5py','numpy','scipy','matplotlib'],
     )
