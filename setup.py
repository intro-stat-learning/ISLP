#!/usr/bin/env python
''' Installation script for ISLP package '''

import os
import sys
from os.path import join as pjoin, dirname, exists
# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'): os.remove('MANIFEST')

# Unconditionally require setuptools
import setuptools

# Package for getting versions from git tags
import versioneer

from setuptools import setup


# Define extensions
EXTS = []

cmdclass = versioneer.get_cmdclass()

# get long_description

long_description = open('README.md', 'rt', encoding='utf-8').read()

def main(**extra_args):
    setup(version=versioneer.get_version(),
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
