#!/usr/bin/env python

# Import useful things
from distutils.core import setup
from setuptools import find_packages

# Load README for long description
with open('README.md', 'r') as f:
    readme = f.read()

#
setup(
    name='positive',
    version='0.2.0',
    description='Low level python API for NR+LIGO packages.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Lionel London',
    author_email='lionel.london@ligo.org',
    packages=find_packages(),
    include_package_data=True,
    package_dir={'positive': 'positive'},
    package_data={
        'positive': [
            'data/berti_kerrcw_fitcoeffsWEB.dat', 'data/berti_swsh_fits.dat',
            'data/ksm2_cw.py', 'data/ysmodels.pickle', 'data/kerr/note.rm',
            'data/kerr/*/*.dat']},
    url='https://github.com/llondon6/positive',
    download_url='https://github.com/llondon6/positive/archive/master.zip',
    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'six'],
    classifiers=(
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering'
    ),
    license='MIT'
)
