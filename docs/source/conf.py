# Configuration file for the Sphinx documentation builder.

# -- Project information

import json
import os

project = 'ISLP'
copyright = '2023, ISLP authors'
author = 'Jonathan Taylor'

release = '0.1'
import ISLP
version = ISLP.__version__

import __main__
dirname = os.path.split(__main__.__file__)[0]
print(dirname)

docs_version = {"labs": "v2",
                "library": "v0.3.18"}

#docs_version = json.loads(open(os.path.join(dirname, 'docs_version.json')).read())
lab_version = docs_version['labs']

myst_enable_extensions = ['substitution']

myst_substitutions = {
    "ISLP_lab_link": f"[ISLP_labs/{lab_version}](https://github.com/intro-stat-learning/ISLP_labs/tree/{lab_version})",
    "ISLP_binder_code": f"[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/intro-stat-learning/ISLP_labs/{lab_version})",
    "ISLP_lab_version": "[ISLP/{0}](https://github.com/intro-stat-learning/ISLP/tree/{0})".format(docs_version['library'])
    }
myst_number_code_blocks = ['python', 'ipython3']

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'texext.math_dollar',
    'numpydoc',
    'myst_nb'
]

graphviz_dot = '/opt/homebrew/bin/dot'
numpydoc_class_members_toctree = False
nb_execution_mode = "auto"
nb_execution_timeout = 60*20 #*100
nb_execution_excludepatterns = ['Ch10*', 'Ch13*', 'imdb.ipynb']
nb_execution_allow_errors = True

#nb_kernel_rgx_aliases = {'python3': "islp_test"}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'pytorch_lightning': ('https://pytorch-lightning.readthedocs.io/en/latest/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = "sphinx_book_theme" 
html_theme_options = {
    "repository_url": "https://github.com/intro-stat-learning/ISLP.git",
    "use_repository_button": True,
}
html_title = "Introduction to Statistical Learning (Python)"
html_logo = "logo.png"

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
