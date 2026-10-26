---
jupytext:
  formats: source/models///ipynb,jupyterbook/models///md:myst,jupyterbook/models///ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Building design matrices with `ModelSpec`

The `ISLP` package provides a facility to build design
matrices for regression and classification tasks. It provides similar functionality to the formula
notation of `R` though uses python objects rather than specification through the special formula syntax.

Related tools include `patsy` and `ColumnTransformer` from `sklearn.compose`. 

Perhaps the most common use is to extract some columns from a `pd.DataFrame` and 
produce a design matrix, optionally with an intercept.

```{code-cell} ipython3
import pandas as pd
import numpy as np

from ISLP import load_data
from ISLP.models import (ModelSpec,
                         summarize,
                         Column,
                         Feature,
                         build_columns)

import statsmodels.api as sm
```

```{code-cell} ipython3
Carseats = load_data('Carseats')
Carseats.columns
```

We'll first build a design matrix that we can use to model `Sales`
in terms of the categorical variable `ShelveLoc` and `Price`.

We see first that `ShelveLoc` is a categorical variable:

```{code-cell} ipython3
Carseats['ShelveLoc']
```

This is recognized by `ModelSpec` and only 2 columns are added for the three levels. The
default behavior is to drop the first level of the categories. Later, 
we will show other contrasts of the 3 columns can be produced.  

This simple example below illustrates how the first argument (its `terms`) is
used to construct a design matrix.

```{code-cell} ipython3
MS = ModelSpec(['ShelveLoc', 'Price'], intercept=True)
X = MS.fit_transform(Carseats)
X.iloc[:10]
```

We note that a column of ones, named `intercept`, has been added: `intercept=True`, as
statsmodels' models need it. Its default, `True` for now, will change to `False` in ISLP 0.6
(glmstar's estimators and most scikit-learn estimators fit the intercept themselves), so pass
`intercept` explicitly: leaving it out gives a `FutureWarning`.

```{code-cell} ipython3
MS_no1 = ModelSpec(['ShelveLoc', 'Price'], intercept=False)
MS_no1.fit_transform(Carseats)[:10]
```

We see that `ShelveLoc` still only contributes
two columns to the design. The `ModelSpec` object does no introspection of its arguments to effectively include an intercept term
in the column space of the design matrix.

To include this intercept via `ShelveLoc` we can use 3 columns to encode this categorical variable. Following the nomenclature of
`R`, we call this a `Contrast` of the categorical variable.

```{code-cell} ipython3
from ISLP.models import contrast
shelve = contrast('ShelveLoc', None)
MS_contr = ModelSpec([shelve, 'Price'], intercept=False)
MS_contr.fit_transform(Carseats)[:10]
```

This example above illustrates that columns need not be identified by name in `terms`. The basic
role of an item in the `terms` sequence is a description of how to extract a column
from a columnar data object, usually a `pd.DataFrame`.

```{code-cell} ipython3
shelve
```

The `Column` object can be used to directly extract relevant columns from a `pd.DataFrame`. If the `encoder` field is not
`None`, then the extracted columns will be passed through `encoder`.
The `get_columns` method produces these columns as well as names for the columns.

```{code-cell} ipython3
shelve.get_columns(Carseats)
```

Let's now fit a simple OLS model with this design.

```{code-cell} ipython3
X = MS_contr.transform(Carseats)
Y = Carseats['Sales']
M_ols = sm.OLS(Y, X).fit()
summarize(M_ols)
```

## Interactions

One of the common uses of formulae in `R` is to specify interactions between variables.
This is done in `ModelSpec` by including a tuple in the `terms` argument.

```{code-cell} ipython3
ModelSpec([(shelve, 'Price'), 'Price'], intercept=True).fit_transform(Carseats).iloc[:10]
```

The above design matrix is clearly rank deficient, as `ModelSpec` has not inspected the formula
and attempted to produce a corresponding matrix that may or may not match a user's intent.

+++

## Ordinal variables

Ordinal variables are handled by a corresponding encoder)

```{code-cell} ipython3
Carseats['OIncome'] = pd.cut(Carseats['Income'], 
                             [0,50,90,200], 
                             labels=['L','M','H'])
MS_order = ModelSpec(['OIncome'], intercept=True).fit(Carseats)
```

Part of the `fit` method of `ModelSpec` involves inspection of the columns of `Carseats`. 
The results of that inspection can be found in the `column_info_` attribute:

```{code-cell} ipython3
MS_order.column_info_
```

## Structure of a `ModelSpec`

The first argument to `ModelSpec` is stored as the `terms` attribute. Under the hood,
this sequence is inspected to produce the `terms_` attribute which specify the objects
that will ultimately create the design matrix.

```{code-cell} ipython3
MS = ModelSpec(['ShelveLoc', 'Price'], intercept=True)
MS.fit(Carseats)
MS.terms_
```

Each element of `terms_` should be a `Feature` which describes a set of columns to be extracted from
a columnar data form as well as possible a possible encoder.

```{code-cell} ipython3
shelve_var = MS.terms_[0]
```

We can find the columns associated to each term using the `build_columns` method of `ModelSpec`:

```{code-cell} ipython3
df, names = build_columns(MS.column_info_,
                          Carseats, 
                          shelve_var)
df
```

The design matrix is constructed by running through `terms_` and concatenating the corresponding columns.

+++

### `Feature` objects

Note that `Feature` objects have a tuple of `variables` as well as an `encoder` attribute. The
tuple of `variables` first creates a concatenated dataframe from all corresponding variables and then
is run through `encoder.transform`. The `encoder.fit` method of each `Feature` is run once during 
the call to `ModelSpec.fit`.

```{code-cell} ipython3
new_var = Feature(('Price', 'Income', 'OIncome'), name='mynewvar', encoder=None)
build_columns(MS.column_info_,
              Carseats, 
              new_var)[0]
```

Let's now transform these columns with an encoder. Within `ModelSpec` we will first build the
arrays above and then call `pca.fit` and finally `pca.transform` within `design.build_columns`.

```{code-cell} ipython3
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(build_columns(MS.column_info_, Carseats, new_var)[0]) # this is done within `ModelSpec.fit`
pca_var = Feature(('Price', 'Income', 'OIncome'), name='mynewvar', encoder=pca)
build_columns(MS.column_info_,
              Carseats, 
              pca_var)[0]
```

The elements of the `variables` attribute may be column identifiers ( `"Price"`), `Column` instances (`price`)
or `Feature` instances (`pca_var`).

```{code-cell} ipython3
price = MS.column_info_['Price']
fancy_var = Feature(('Income', price, pca_var), name='fancy', encoder=None)
build_columns(MS.column_info_,
              Carseats, 
              fancy_var)[0]
```

## Predicting at new points

```{code-cell} ipython3
MS = ModelSpec(['Price', 'Income'], intercept=True).fit(Carseats)
X = MS.transform(Carseats)
Y = Carseats['Sales']
M_ols = sm.OLS(Y, X).fit()
M_ols.params
```

As `ModelSpec` is a transformer, it can be evaluated at new feature values.
Constructing the design matrix at any values is carried out by the `transform` method.

```{code-cell} ipython3
new_data = pd.DataFrame({'Price':[40, 50], 'Income':[10, 20]})
new_X = MS.transform(new_data)
M_ols.get_prediction(new_X).predicted_mean
```

## Using `np.ndarray`

As the basic model is to concatenate columns extracted from a columnar data
representation, one *can* use `np.ndarray` as the column data. In this case,
columns will be selected by integer indices. 

### Caveats using `np.ndarray`

If the `terms` only refer to a few columns of the data frame, the `transform` method only needs a dataframe with those columns.
However,
unless all features are floats, `np.ndarray` will default to a dtype of `object`, complicating issues.

However, if we had used an `np.ndarray`, the column identifiers would be integers identifying specific columns so,
in order to work correctly, `transform` would need another `np.ndarray` where the columns have the same meaning. 

We illustrate this below, where we build a model from `Price` and `Income` for `Sales` and want to find predictions at new
values of `Price` and `Location`. We first find the predicitions using `pd.DataFrame` and then illustrate the difficulties
in using `np.ndarray`.

+++

We will refit this model, using `ModelSpec` with an `np.ndarray` instead

```{code-cell} ipython3
Carseats_np = np.asarray(Carseats[['Price', 'Education', 'Income']])
MS_np = ModelSpec([0,2], intercept=True).fit(Carseats_np)
MS_np.transform(Carseats_np)
```

```{code-cell} ipython3
M_ols_np = sm.OLS(Y, MS_np.transform(Carseats_np)).fit()
M_ols_np.params
```

Now, let's consider finding the design matrix at new points. 
When using `pd.DataFrame` we only need to supply the `transform` method
a data frame with columns implicated in the `terms` argument (in this case, `Price` and `Income`). 

However, when using `np.ndarray` with integers as indices, `Price` was column 0 and `Income` was column 2. The only
sensible way to produce a return for predict is to extract its 0th and 2nd columns. Note this means
that the meaning of columns in an `np.ndarray` provided to `transform` essentially must be identical to those
passed to `fit`.

```{code-cell} ipython3
try:
    new_D = np.array([[40,50], [10,20]]).T
    new_X = MS_np.transform(new_D)
except IndexError as e:
    print(e)
```

Ultimately, `M` expects 3 columns for new predictions because it was fit
with a matrix having 3 columns (the first representing an intercept).

We might be tempted to try as with the `pd.DataFrame` and produce
an `np.ndarray` with only the necessary variables.

```{code-cell} ipython3
new_D = np.array([[40,50], [np.nan, np.nan], [10,20]]).T
new_X = MS_np.transform(new_D)
print(new_X)
M_ols.get_prediction(new_X).predicted_mean
```

For more complicated design contructions ensuring the columns of `new_D` match that of the original data will be more cumbersome. We expect
then that `pd.DataFrame` (or a columnar data representation with similar API) will likely be easier to use with `ModelSpec`.
