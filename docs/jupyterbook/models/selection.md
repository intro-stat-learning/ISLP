---
jupytext:
  formats: source/models///ipynb,jupyterbook/models///md:myst,jupyterbook/models///ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Model selection using `ModelSpec`


In this lab we illustrate how to run forward stepwise model selection
using the model specification capability of `ModelSpec`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from statsmodels.api import OLS
from ISLP import load_data
from ISLP.models import (ModelSpec,
                         Stepwise,
                         sklearn_selected)
```

### Forward Selection
 
We will  apply the forward-selection approach to the  `Hitters` 
data.  We wish to predict a baseball player’s `Salary` on the
basis of various statistics associated with performance in the
previous year.

```{code-cell} ipython3
Hitters = load_data('Hitters')
np.isnan(Hitters['Salary']).sum()
```

    
 We see that `Salary` is missing for 59 players. The
`dropna()`  method of data frames removes all of the rows that have missing
values in any variable (by default --- see  `Hitters.dropna?`).

```{code-cell} ipython3
Hitters = Hitters.dropna()
Hitters.shape
```

We first choose the best model using forward selection based on AIC. This score
is not built in as a metric to `sklearn`. We therefore define a function to compute it ourselves, and use
it as a scorer. By default, `sklearn` tries to maximize a score, hence
  our scoring function  computes the negative AIC statistic.

```{code-cell} ipython3
def negAIC(estimator, X, Y):
    "Negative AIC"
    n, p = X.shape
    Yhat = estimator.predict(X)
    MSE = np.mean((Y - Yhat)**2)
    return n + n * np.log(MSE) + 2 * (p + 1)
    
```

We need to estimate the residual variance $\sigma^2$, which is the first argument in our scoring function above.
We will fit the biggest model, using all the variables, and estimate $\sigma^2$ based on its MSE.

```{code-cell} ipython3
design = ModelSpec(Hitters.columns.drop('Salary'), intercept=True).fit(Hitters)
Y = np.array(Hitters['Salary'])
X = design.transform(Hitters)
```

Along with a score we need to specify the search strategy. This is done through the object
`Stepwise()`  in the `ISLP.models` package. The method `Stepwise.first_peak()`
runs forward stepwise until any further additions to the model do not result
in an improvement in the evaluation score. Similarly, the method `Stepwise.fixed_steps()`
runs a fixed number of steps of stepwise search.

```{code-cell} ipython3
strategy = Stepwise.first_peak(design,
                               direction='forward',
                               max_terms=len(design.terms))
```

 
We now fit a linear regression model with `Salary` as outcome using forward
selection. To do so, we use the function `sklearn_selected()`  from the `ISLP.models` package. This takes
a model from `statsmodels` along with a search strategy and selects a model with its
`fit` method. Without specifying a `scoring` argument, the score defaults to MSE, and so all 19 variables will be
selected.

```{code-cell} ipython3
hitters_MSE = sklearn_selected(OLS,
                               strategy)
hitters_MSE.fit(Hitters, Y)
hitters_MSE.selected_state_
```

 Using `neg_Cp` results in a smaller model, as expected, with just 4variables selected.

```{code-cell} ipython3
hitters_Cp = sklearn_selected(OLS,
                              strategy,
                              scoring=negAIC)
hitters_Cp.fit(Hitters, Y)
hitters_Cp.selected_state_
```
