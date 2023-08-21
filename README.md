# ISLP

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



