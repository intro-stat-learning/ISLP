
from dataclasses import dataclass
from copy import copy

import shlex
import subprocess
import os
import sys
import json
import nbformat
from argparse import ArgumentParser

def get_version():
    import __main__
    dirname = os.path.split(__main__.__file__)[0]
    sys.path.append(os.path.join(dirname, 'source'))
    from conf import docs_version
    sys.path = sys.path[:-1]
    return docs_version


@dataclass
class Lab(object):

    labfile: str
    version: str = 'v2'

    def __post_init__(self):
        self.labfile = os.path.abspath(self.labfile)

    def fix_header(self):
        labname = os.path.split(self.labfile)[1]
        base = os.path.splitext(self.labfile)[0]
        args = shlex.split(f'jupytext --set-formats ipynb,md:myst {self.labfile}')
        subprocess.run(args)

        # successful run of jupytext
        myst = open(f'{base}.md').read().strip()
        split_myst = myst.split('\n')
        new_myst = []

        colab_code = f'''
<a target="_blank" href="https://colab.research.google.com/github/intro-stat-learning/ISLP_labs/blob/{self.version}/{labname}.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/intro-stat-learning/ISLP_labs/{self.version}?labpath={labname}.ipynb)

'''

        chapter_buffer = 200 # should use a regex...
        for l in split_myst[:chapter_buffer]: # assumes Chapter appears in first 200 linesmyst.split('\n')
            if l.strip()[:9] != '# Chapter': # exclude the line with "# Chapter"
                if 'Lab:' in l:
                    l = l.replace('Lab:', '') + '\n' + colab_code
                new_myst.append(l)

        myst = '\n'.join(new_myst + split_myst[chapter_buffer:])
        open(f'{base}.md', 'w').write(myst)

        args = shlex.split(f'jupytext --set-formats Rmd,ipynb {base}.ipynb')
        subprocess.run(args)

        args = shlex.split(f'jupytext --sync {base}.ipynb')
        subprocess.run(args)


        subprocess.run(['rm', f'{base}.md'])

def fix_Ch06(Ch06_nbfile):

    nb = nbformat.read(open(Ch06_nbfile), 4)

    md_cell = copy(nb.cells[0])
    md_cell['id'] = md_cell['id'] + '_duplicate'
    
    src = '''

```{attention}
Using `skl.ElasticNet` to fit ridge regression
throws up many warnings. We have suppressed them below by a call to `warnings.simplefilter()`.
```

'''    

    md_cell['source'] = [l +'\n' for l in src.split('\n')]

    for i, cell in enumerate(nb.cells):
        if cell['cell_type'] == 'code':
            code_cell = copy(cell)
            code_cell['id'] = code_cell['id'] + '_duplicate'
            code_cell['source'] = ['import warnings\n', 'warnings.simplefilter("ignore")\n']
            break

    nb.cells.insert(i, md_cell)
    nb.cells.insert(i+1, code_cell)    

    nbformat.write(nb, open(Ch06_nbfile, 'w'))
    subprocess.run(shlex.split(f'jupytext --sync {Ch06_nbfile}'))

if __name__ == "__main__":

    docs_version = get_version()

    parser = ArgumentParser()
    parser.add_argument('labs',
                        metavar='N',
                        type=str,
                        nargs='+')

    args = parser.parse_args()

    for labfile in args.labs:
        l = Lab(labfile=labfile, version=docs_version['labs'])
        l.fix_header()
        if '06' in labfile:
            fix_Ch06(labfile)

