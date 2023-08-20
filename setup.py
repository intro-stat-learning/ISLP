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

class SetupDependency(object):
    """ SetupDependency class

    Parameters
    ----------
    import_name : str
        Name with which required package should be ``import``ed.
    min_ver : str
        Distutils version string giving minimum version for package.
    req_type : {'install_requires', 'setup_requires'}, optional
        Setuptools dependency type.
    heavy : {False, True}, optional
        If True, and package is already installed (importable), then do not add
        to the setuptools dependency lists.  This prevents setuptools
        reinstalling big packages when the package was installed without using
        setuptools, or this is an upgrade, and we want to avoid the pip default
        behavior of upgrading all dependencies.
    install_name : str, optional
        Name identifying package to install from pypi etc, if different from
        `import_name`.
    """

    def __init__(self, import_name,
                 min_ver,
                 req_type='install_requires',
                 heavy=False,
                 install_name=None):
        self.import_name = import_name
        self.min_ver = min_ver
        self.req_type = req_type
        self.heavy = heavy
        self.install_name = (import_name if install_name is None
                             else install_name)

    def check_fill(self, setuptools_kwargs):
        """ Process this dependency, maybe filling `setuptools_kwargs`

        Run checks on this dependency.  If not using setuptools, then raise
        error for unmet dependencies.  If using setuptools, add missing or
        not-heavy dependencies to `setuptools_kwargs`.

        A heavy dependency is one that is inconvenient to install
        automatically, such as numpy or (particularly) scipy, matplotlib.

        Parameters
        ----------
        setuptools_kwargs : dict
            Dictionary of setuptools keyword arguments that may be modified
            in-place while checking dependencies.
        """
        found_ver = get_pkg_version(self.import_name)
        ver_err_msg = version_error_msg(self.import_name,
                                        found_ver,
                                        self.min_ver)
        if not 'setuptools' in sys.modules:
            # Not using setuptools; raise error for any unmet dependencies
            if ver_err_msg is not None:
                raise RuntimeError(ver_err_msg)
            return
        # Using setuptools; add packages to given section of
        # setup/install_requires, unless it's a heavy dependency for which we
        # already have an acceptable importable version.
        if self.heavy and ver_err_msg is None:
            return
        new_req = '{0}>={1}'.format(self.import_name, self.min_ver)
        old_reqs = setuptools_kwargs.get(self.req_type, [])
        setuptools_kwargs[self.req_type] = old_reqs + [new_req]

def get_pkg_version(pkg_name):
    """ Return package version for `pkg_name` if installed

    Returns
    -------
    pkg_version : str or None
        Return None if package not importable.  Return 'unknown' if standard
        ``__version__`` string not present. Otherwise return version string.
    """
    try:
        pkg = __import__(pkg_name)
    except ImportError:
        return None
    try:
        return pkg.__version__
    except AttributeError:
        return 'unknown'

def version_error_msg(pkg_name, found_ver, min_ver):
    """ Return informative error message for version or None
    """
    if found_ver is None:
        return 'We need package {0}, but not importable'.format(pkg_name)
    if found_ver == 'unknown':
        return 'We need {0} version {1}, but cannot get version'.format(
            pkg_name, min_ver)
    if LooseVersion(found_ver) >= LooseVersion(min_ver):
        return None
    return 'We need {0} version {1}, but found version {2}'.format(
        pkg_name, found_ver, min_ver)



# Try to preempt setuptools monkeypatching of Extension handling when Pyrex
# is missing.  Otherwise the monkeypatched Extension will change .pyx
# filenames to .c filenames, and we probably don't have the .c files.
sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))
# Set setuptools extra arguments
extra_setuptools_args = dict(
    tests_require=['nose'],
    test_suite='nose.collector',
    zip_safe=False,
    extras_require = dict(
        doc=['Sphinx>=1.0'],
        test=['nose>=0.10.1']))

# Define extensions
EXTS = []

SetupDependency('numpy', info.NUMPY_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('scipy', info.SCIPY_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('matplotlib', info.MATPLOTLIB_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('pandas', info.PANDAS_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('statsmodels', info.STATSMODELS_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('scikit-learn', info.SKLEARN_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)

dirname = os.path.dirname(__file__)
requirements = open(os.path.join(dirname,
                                 'requirements.txt')).read().strip().split('\n')

# requirements = '''numpy
# scipy
# jupyter
# pandas
# lxml # pandas needs this for html
# scikit-learn
# joblib
# pygam # for GAM in Ch7
# lifelines'''.split('\n')
# #l0bnb # for bestsubsets



for req in requirements:
    req = req.split('#')[0]
    import sys; sys.stderr.write(req+'\n')
    SetupDependency(req, "0.0",
                    req_type='install_requires',
                    heavy=True).check_fill(extra_setuptools_args)

cmdclass=versioneer.get_cmdclass()

# get long_description

long_description = open('README.md', 'rt', encoding='utf-8').read()

def main(**extra_args):
    setup(name=info.NAME,
          maintainer=info.MAINTAINER,
          maintainer_email=info.MAINTAINER_EMAIL,
          description=info.DESCRIPTION,
          url=info.URL,
          download_url=info.DOWNLOAD_URL,
          license=info.LICENSE,
          classifiers=info.CLASSIFIERS,
          author=info.AUTHOR,
          author_email=info.AUTHOR_EMAIL,
          platforms=info.PLATFORMS,
          version=versioneer.get_version(),
          requires=info.REQUIRES,
          provides=info.PROVIDES,
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
    main(**extra_setuptools_args)
