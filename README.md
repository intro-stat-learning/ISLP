# ISLP
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

This package collects data sets and various helper functions
for ISLP.

## Install instructions

### Mac OS X / Linux

We generally recommend creating a [conda](https://anaconda.org) environment to isolate any code
from other dependencies. The `ISLP` package does not have unusual dependencies, but this is still
good practice. To create a conda environment in a Mac OS X or Linux environment run:

```{python}
conda create --name islp
```

To run python code in this environment, you must activate it:

```{python}
conda activate islp
```

### Windows

On windows, create a `Python` environment called `islp` in the Anaconda app. This can be done by selecting `Environments` on the left hand side of the app's screen. After creating the environment, open a terminal within that environment by clicking on the "Play" button.


## Installing `ISLP`

Having completed the steps above, we use `pip` to install the `ISLP` package:

```{python}
pip install ISLP
```

### Torch requirements

The `ISLP` labs use `torch` and various related packages for the lab on deep learning. The requirements
are included in the requirements for `ISLP` with the exception of those needed
for the labs which are included in the [requirements for the labs](https://github.com/intro-stat-learning/ISLP_labs/blob/main/requirements.txt). 

## Jupyter

### Mac OS X

If JupyterLab is not already installed, run the following after having activated your `islp` environment:

```{python}
pip install jupyterlab
```

### Windows

Either use the same `pip` command above or install JupyterLab from the `Home` tab. Ensure that the environment
is your `islp` environment. This information appears near the top left in the Anaconda `Home` page.


## Documentation

See the [docs](https://intro-stat-learning.github.io/ISLP/labs.html) for the latest documentation.

## Authors

- Jonathan Taylor
- Trevor Hastie
- Gareth James
- Robert Tibshirani
- Daniela Witten




## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/danielawitten"><img src="https://avatars.githubusercontent.com/u/12654191?v=4?s=100" width="100px;" alt="danielawitten"/><br /><sub><b>danielawitten</b></sub></a><br /><a href="https://github.com/intro-stat-learning/ISLP/commits?author=danielawitten" title="Code">💻</a> <a href="#content-danielawitten" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://web.stanford.edu/~hastie/"><img src="https://avatars.githubusercontent.com/u/13293253?v=4?s=100" width="100px;" alt="trevorhastie"/><br /><sub><b>trevorhastie</b></sub></a><br /><a href="https://github.com/intro-stat-learning/ISLP/commits?author=trevorhastie" title="Code">💻</a> <a href="#content-trevorhastie" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tibshirani"><img src="https://avatars.githubusercontent.com/u/2848609?v=4?s=100" width="100px;" alt="tibshirani"/><br /><sub><b>tibshirani</b></sub></a><br /><a href="https://github.com/intro-stat-learning/ISLP/commits?author=tibshirani" title="Code">💻</a> <a href="#content-tibshirani" title="Content">🖋</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!