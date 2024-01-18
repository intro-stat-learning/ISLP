#!/usr/bin/env python
''' Installation script for ISLP package '''

import os
import sys
from os.path import join as pjoin, dirname, exists
from distutils.version import LooseVersion
# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'): os.remove('MANIFEST')

# Unconditionally require setuptools
import setuptools

# Package for getting versions from git tags
import versioneer

# Import distutils _after_ setuptools import, and after removing
# MANIFEST
from distutils.core import setup
from distutils.extension import Extension

# Get various parameters for this version, stored in ISLP/info.py

class Bunch(object):
    def __init__(self, vars):
        for key, name in vars.items():
            if key.startswith('__'):
                continue
            self.__dict__[key] = name

def read_vars_from(ver_file):
    """ Read variables from Python text file

    Parameters
    ----------
    ver_file : str
        Filename of file to read

    Returns
    -------
    info_vars : Bunch instance
        Bunch object where variables read from `ver_file` appear as
        attributes
    """
    # Use exec for compabibility with Python 3
    ns = {}
    with open(ver_file, 'rt') as fobj:
        exec(fobj.read(), ns)
    return Bunch(ns)

info = read_vars_from(pjoin('ISLP', 'info.py'))

# Try to preempt setuptools monkeypatching of Extension handling when Pyrex
# is missing.  Otherwise the monkeypatched Extension will change .pyx
# filenames to .c filenames, and we probably don't have the .c files.
sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))

# Define extensions
EXTS = []

cmdclass=versioneer.get_cmdclass()

# get long_description

long_description = open('README.md', 'rt', encoding='utf-8').read()

def main(**extra_args):
    setup(#name=info.NAME,
          #maintainer=info.MAINTAINER,
          #maintainer_email=info.MAINTAINER_EMAIL,
          #description=info.DESCRIPTION,
          #url=info.URL,
          #download_url=info.DOWNLOAD_URL,
          #license=info.LICENSE,
          #classifiers=info.CLASSIFIERS,
          #author=info.AUTHOR,
          #author_email=info.AUTHOR_EMAIL,
          #platforms=info.PLATFORMS,
          version=versioneer.get_version(),
          #requires=info.REQUIRES,
          #provides=info.PROVIDES,
          packages     = ['ISLP',
                          'ISLP.models',
                          'ISLP.models',
                          'ISLP.bart',
                          'ISLP.torch',
                          'ISLP.data'
                          ],
          ext_modules = EXTS,
          package_data = {"ISLP":["data/*csv", "data/*npy", "data/*data"]},
          include_package_data=True,
          data_files=[],
          scripts=[],
          long_description=long_description,
          cmdclass = cmdclass,
          **extra_args
         )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main()
