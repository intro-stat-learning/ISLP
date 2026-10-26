---
file_format: mystnb
kernelspec:
  name: python3
  display_name: python3
---

# ModelSpec, ColumnTransformer, patsy and formulaic

`ModelSpec` builds model matrices, as do `sklearn`'s
[ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html),
[patsy](https://patsy.readthedocs.io) and
[formulaic](https://matthewwardrop.github.io/formulaic/). The two
formula libraries parse a string such as `'Sales ~ Price * ShelveLoc'`
in the spirit of R's model formulae. A `ColumnTransformer` lists a
transformer and the columns it applies to, for each block of columns.
`ModelSpec` takes a list of Python objects: column names, tuples for
interactions, and *features* that wrap any `sklearn` transformer.

This page compares them on flexibility and extensibility: what it
takes to add a new kind of term, to nest one feature inside another,
to interact terms, and to work inside `sklearn`. Reproducing R's
formula semantics is not a goal of `ModelSpec`, so some defaults
differ on purpose; those are collected at the end.

## At a glance

| | `ModelSpec` | `ColumnTransformer` | patsy | formulaic |
|---|---|---|---|---|
| any `sklearn` transformer as a term | directly | directly | with a stateful-transform wrapper | with a stateful-transform wrapper |
| interacting terms (e.g. a spline with a categorical) | a tuple | no: `PolynomialFeatures` multiplies every pair of columns | `*` and `:` | `*` and `:` |
| a feature of other features (e.g. PCA of a polynomial and dummies) | yes | yes, nesting `Pipeline`s and `ColumnTransformer`s | no | no |
| categorical columns | coded from their dtype | an encoder you name, for columns you list or select by dtype | coded from their dtype | coded from their dtype |
| a model fitted elsewhere as a term | yes, with `FrozenEstimator` | with a `FunctionTransformer` around it | yes, a function calling it | yes, a function calling it |
| building a specification | Python lists of objects | lists of (name, transformer, columns) | formula strings | formula strings |
| `sklearn` transformer (`Pipeline`, refit in each CV fold) | yes | yes | needs a wrapper estimator | needs a wrapper estimator |
| tuning a term's parameters by `GridSearchCV` | yes (`modelspec__terms__0__df`) | yes (`columntransformer__spline__n_knots`) | needs a wrapper estimator | needs a wrapper estimator |
| pickling a fitted specification | yes | yes | no | yes |
| sparse model matrices | yes (`sparse_output=True`) | yes (`sparse_threshold`) | no | yes (`output='sparse'`) |
| input other than `pandas` (e.g. polars) | yes; output in the same backend | yes; output by `set_output` | no | yes |

`ModelSpec` warns, when fit, that it moves to its own package in
ISLP 0.7; the warning is silenced here.

```{code-cell} ipython3
import pickle, warnings
warnings.filterwarnings('ignore', message='ModelSpec will move')
import numpy as np, pandas as pd
import patsy, formulaic
import sklearn

from ISLP import load_data
from ISLP import models as islp
from ISLP.models import ModelSpec as MS

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

Carseats = load_data('Carseats')
new = Carseats.iloc[:5]
print(f'patsy {patsy.__version__}, formulaic {formulaic.__version__}, '
      f'scikit-learn {sklearn.__version__}, pandas {pd.__version__}')
```

The `ISLP` helpers are accessed as `islp.poly`, `islp.ns`, etc. rather
than imported by name. Formula strings are evaluated in the calling
namespace, so importing `ISLP`'s `bs` would silently replace patsy's
`bs` inside `patsy.dmatrix('bs(Price, df=5)', ...)`.

## Terms are Python objects

Any `sklearn` transformer can be used as a term with `derived_feature`,
with no registration or wrapper. Its state is fit when the `ModelSpec`
is fit, and reused on new data.

```{code-cell} ipython3
spline = islp.derived_feature(['Price'],
                              encoder=SplineTransformer(n_knots=5),
                              name='spline(Price)')
design = MS([spline, 'ShelveLoc'], intercept=True).fit(Carseats)
design.transform(new)
```

Because terms are objects, a list of them can be built with ordinary
Python, for example a natural spline for every numeric column:

```{code-cell} ipython3
numeric = [c for c in Carseats.columns.drop('Sales')
           if pd.api.types.is_numeric_dtype(Carseats[c])]
design = MS([islp.ns(c, df=3) for c in numeric] + ['ShelveLoc', 'US'],
            intercept=True)
design.fit_transform(Carseats).shape
```

The inputs of a feature may themselves be features, which are
encoded first. Here a PCA is fit to an orthogonal polynomial in
`Price`, `Income` and the one-hot encoding of `ShelveLoc`, and its
scores are interacted with `US`:

```{code-cell} ipython3
scores = islp.derived_feature([islp.poly('Price', 2),
                               'Income',
                               islp.contrast('ShelveLoc', None)],
                              encoder=PCA(n_components=2),
                              name='pc')
design = MS([scores, 'US', (scores, 'US')], intercept=True).fit(Carseats)
design.transform(new)
```

Fitting a `ModelSpec` fits copies of the encoders of its terms, so the
same terms can be used in several specifications, and a fitted
specification can be fit again. A model fit elsewhere, on other data,
can be a term too: wrapped in `sklearn`'s `FrozenEstimator`, it is used
as is and never refit. Here a nearest-neighbour fit to the first 200
stores supplies a column for the others:

```{code-cell} ipython3
from sklearn.frozen import FrozenEstimator

first, rest = Carseats.iloc[:200], Carseats.iloc[200:]
knn = KNeighborsRegressor(10).fit(first[['Price', 'Income']], first['Sales'])
knn_term = islp.derived_feature(['Price', 'Income'],
                                encoder=FrozenEstimator(knn),
                                name='knn',
                                use_transform=False)   # its predictions
MS([knn_term, 'ShelveLoc'], intercept=True).fit_transform(rest).head()
```

## The same in patsy and formulaic

Both formula libraries support stateful transforms, but an `sklearn`
transformer needs a wrapper written against each library's protocol.

```{code-cell} ipython3
from formulaic.transforms import stateful_transform

class _PatsySklearn:
    # patsy's stateful transform protocol, wrapping an sklearn transformer
    def __init__(self):
        self.chunks = []
    def memorize_chunk(self, *cols, est):
        self.est = est
        self.chunks.append(np.column_stack(cols))
    def memorize_finish(self):
        self.est_ = clone(self.est).fit(np.vstack(self.chunks))
    def transform(self, *cols, est):
        return self.est_.transform(np.column_stack(cols))

patsy_sk = patsy.stateful_transform(_PatsySklearn)

@stateful_transform
def formulaic_sk(*cols, est, _state=None):
    X = np.column_stack(cols)
    if 'est' not in _state:
        _state['est'] = clone(est).fit(X)
    return _state['est'].transform(X)

spl = SplineTransformer(n_knots=5)
X_patsy = patsy.dmatrix('patsy_sk(Price, est=spl) + ShelveLoc', Carseats)
X_formulaic = formulaic.model_matrix('formulaic_sk(Price, est=spl) + ShelveLoc', Carseats)
X_ms = MS([spline, 'ShelveLoc'], intercept=True).fit_transform(Carseats)
X_patsy.shape, X_formulaic.shape, X_ms.shape
```

With the wrapper, a learned feature can be interacted with a
categorical in either library. Nesting is a different matter: inside a
function call, `C(ShelveLoc)` is a box holding the labels rather than
its dummy columns, so the PCA above cannot be written this way.

```{code-cell} ipython3
pc = PCA(n_components=2)
for name, build in [('patsy', lambda: patsy.dmatrix(
                         'patsy_sk(Price, Price**2, Income, C(ShelveLoc), est=pc)', Carseats)),
                    ('formulaic', lambda: formulaic.model_matrix(
                         'formulaic_sk(Price, Price**2, Income, C(ShelveLoc), est=pc)', Carseats))]:
    try:
        build()
    except Exception as e:
        print(f'{name}: {type(e).__name__}')
```

## Working inside scikit-learn

A `ModelSpec` is an `sklearn` transformer, so it can be the first step
of a `Pipeline`, and refit on each fold of a cross-validation. Here the
regression fits its own intercept, so the model matrix has none:

```{code-cell} ipython3
pipe = make_pipeline(MS(['Price', 'ShelveLoc', 'Income'], intercept=False),
                     LinearRegression())
cross_val_score(pipe, Carseats, Carseats['Sales'], cv=5).round(3)
```

### Tuning the parameters of terms

The parameters of each term are parameters of the `ModelSpec`, so
`GridSearchCV` can tune them as it tunes those of any `sklearn`
estimator. A term's parameter is named by the position of the term in
`terms`:

| key | the parameter of |
|---|---|
| `'terms__<i>__<parameter>'` | the term at position `i`, e.g. `'terms__0__df'` |
| `'terms__<i>__<j>__<parameter>'` | part `j` of the interaction at position `i` |
| `'terms__<name>__<parameter>'` | the term given that name, e.g. `ns('Price', df=4, name='price')` |

`get_params()` lists them:

```{code-cell} ipython3
spec = MS([islp.ns('Price', df=4),
           'US',
           ('ShelveLoc', islp.poly('Income', 2))], intercept=False)
{k: v for k, v in spec.get_params().items()
 if k in ('terms__0__df', 'terms__2__1__degree')}
```

In a `Pipeline` each key is prefixed by the name of its step, here
`'modelspec__'`. Both parameters are tuned below. A term made by
`ns()`, `bs()`, `poly()` or `pca()` is made again with its new
parameters, so its name, and the names of its columns, follow:

```{code-cell} ipython3
pipe = make_pipeline(spec, LinearRegression())
grid = {'modelspec__terms__0__df': [1, 2, 4, 8],
        'modelspec__terms__2__1__degree': [1, 2, 3]}
search = GridSearchCV(pipe, grid, cv=5).fit(Carseats, Carseats['Sales'])
best = search.best_estimator_[0]
search.best_params_, best.terms[0].name, best.terms[2][1].name
```

The parameters of other terms, e.g. a `derived_feature` of any
transformer or `contrast()`, are those of their encoder
(`'terms__<i>__n_knots'` for a `SplineTransformer`,
`'terms__<i>__method'` for `contrast()`). Setting a parameter does not
change `terms`: it is replaced by a copy, in which a term used in
several places, e.g. also in an interaction, is changed in all of them.

So can `terms` itself, to choose among specifications:

```{code-cell} ipython3
grid = {'modelspec__terms': [['Price', 'ShelveLoc'],
                             ['Price', 'ShelveLoc', ('Price', 'ShelveLoc')]]}
search = GridSearchCV(make_pipeline(MS(intercept=False), LinearRegression()),
                      grid, cv=5)
search.fit(Carseats, Carseats['Sales'])
len(search.best_params_['modelspec__terms'])
```

`get_feature_names_out` names the columns, which also enables
`set_output`, here giving a data frame from an `np.ndarray`:

```{code-cell} ipython3
A = Carseats[['Price', 'Income']].to_numpy()
(MS([0, islp.poly(1, 2)], intercept=True)
 .set_output(transform='pandas')
 .fit_transform(A)
 .head())
```

## Compared with scikit-learn's ColumnTransformer

A `ColumnTransformer` fits as naturally into `sklearn` as a `ModelSpec`:
it is a transformer, and the parameters of its transformers can be
tuned. The same model matrix, a spline in `Price`, `ShelveLoc` and
`Income`, is written below both ways; they give the same columns, and
tuning the number of knots of the spline gives the same results.

```{code-cell} ipython3
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

spline = islp.derived_feature(['Price'], encoder=SplineTransformer(n_knots=5),
                              name='spline')
ms = MS([spline, 'ShelveLoc', 'Income'], intercept=False)

ct = ColumnTransformer([
    ('spline', SplineTransformer(n_knots=5), ['Price']),
    ('shelve', OneHotEncoder(drop='first', sparse_output=False), ['ShelveLoc']),
    ('income', 'passthrough', ['Income'])])

searches = {
    'ModelSpec': GridSearchCV(make_pipeline(ms, LinearRegression()),
                              {'modelspec__terms__0__n_knots': [3, 5, 8]}, cv=5),
    'ColumnTransformer': GridSearchCV(make_pipeline(ct, LinearRegression()),
                                      {'columntransformer__spline__n_knots': [3, 5, 8]}, cv=5)}
for name, search in searches.items():
    search.fit(Carseats, Carseats['Sales'])
    print(f'{name:18s}', search.best_params_, search.cv_results_['mean_test_score'].round(4))
```

The categorical `ShelveLoc` needs its own encoder in the
`ColumnTransformer`, chosen by name or, as here, by dtype; a
`ModelSpec` codes it from its dtype.

```{code-cell} ipython3
ct_by_dtype = ColumnTransformer([
    ('spline', SplineTransformer(n_knots=5), ['Price']),
    ('categorical', OneHotEncoder(drop='first', sparse_output=False),
     make_column_selector(dtype_include='category')),
    ('income', 'passthrough', ['Income'])])
ct_by_dtype.fit_transform(Carseats[['Price', 'ShelveLoc', 'Income']]).shape
```

Interactions are where the two differ most. In a `ModelSpec` a tuple
interacts any terms, here the spline with `ShelveLoc`. A
`ColumnTransformer` has no way to multiply one block of columns by
another: `PolynomialFeatures(interaction_only=True)` multiplies every
pair of its input columns, including pairs of spline columns and of
dummy columns.

```{code-cell} ipython3
ms_interaction = MS([spline, 'ShelveLoc', (spline, 'ShelveLoc')], intercept=False)
ct_interaction = make_pipeline(
    ColumnTransformer([('spline', SplineTransformer(n_knots=5), ['Price']),
                       ('shelve', OneHotEncoder(drop='first', sparse_output=False), ['ShelveLoc'])]),
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
print('ModelSpec        ', ms_interaction.fit_transform(Carseats).shape[1], 'columns')
print('ColumnTransformer', ct_interaction.fit_transform(Carseats).shape[1], 'columns')
```

A fitted `ModelSpec` can be pickled, with all of its encoders, as can
formulaic's `ModelSpec`; patsy's `DesignInfo` cannot.

```{code-cell} ipython3
fitted = {'ModelSpec': lambda: MS(['Price', 'ShelveLoc', islp.bs('Price', df=5)],
                                  intercept=True).fit(Carseats),
          'patsy': lambda: patsy.dmatrix('Price + ShelveLoc + bs(Price, df=5)', Carseats).design_info,
          'formulaic': lambda: formulaic.model_matrix('Price + ShelveLoc + bs(Price, df=5)', Carseats).model_spec}
for name, spec in fitted.items():
    try:
        pickle.dumps(spec())
        print(f'{name}: pickled')
    except Exception as e:
        print(f'{name}: {type(e).__name__}')
```

## Sparse model matrices

With `sparse_output=True`, a `ModelSpec` returns a
`scipy.sparse.csr_matrix`. Categorical encodings and their interactions
are built sparse throughout, so a categorical with many levels never
forms its dense columns; formulaic's `output='sparse'` does the same.
Here a categorical with 5,000 levels and 50,000 rows, whose dense model
matrix would take 2 GB:

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, K = 50_000, 5_000
B = pd.DataFrame({'x': rng.standard_normal(n),
                  'id': pd.Categorical(rng.integers(0, K, n).astype(str))})
X_ms = MS(['x', 'id', ('id', 'x')], intercept=True,
          sparse_output=True).fit_transform(B)
X_formulaic = formulaic.model_matrix('x + id + id:x', B, output='sparse')
print('ModelSpec', type(X_ms).__name__, X_ms.shape, f'{X_ms.nnz:,} nonzero')
print('formulaic', type(X_formulaic).__name__, X_formulaic.shape, f'{X_formulaic.nnz:,} nonzero')
```

## Other dataframes

Dataframes of the backends supported by
[narwhals](https://narwhals-dev.github.io/narwhals/), such as polars or
pyarrow, can be given to a `ModelSpec`, and the model matrix is returned
in the same backend. A polars `Enum` carries the order of its levels,
so `ordinal()` can score it. formulaic accepts them too; patsy expects
`pandas` or `numpy`.

```{code-cell} ipython3
import polars as pl

P = pl.from_pandas(Carseats).with_columns(
    pl.col('ShelveLoc').cast(pl.String).cast(pl.Enum(['Bad', 'Medium', 'Good'])))
design = MS(['Price', 'US', islp.ordinal('ShelveLoc')], intercept=True).fit(P)
design.transform(P.head())
```

```{code-cell} ipython3
for name, build in [('patsy', lambda: patsy.dmatrix('Price + US', P)),
                    ('formulaic', lambda: formulaic.model_matrix('Price + US', P))]:
    try:
        print(f'{name}: {type(build()).__name__}')
    except Exception as e:
        print(f'{name}: {type(e).__name__}')
```

## Model structure

All three record which columns of the model matrix belong to each term,
which is what an ANOVA or a group-wise selection procedure needs.

```{code-cell} ipython3
design = MS(['Price', 'ShelveLoc', ('Price', 'ShelveLoc')], intercept=True).fit(Carseats)
print('ModelSpec', {str(k): (v.start, v.stop) for k, v in design.column_map_.items()})
print('patsy    ', {k: (v.start, v.stop) for k, v in
                    patsy.dmatrix('Price*ShelveLoc', Carseats).design_info.term_name_slices.items()})
print('formulaic', {str(k): v for k, v in
                    formulaic.model_matrix('Price*ShelveLoc', Carseats).model_spec.term_indices.items()})
```

Where they overlap, the matrices agree: treatment and sum contrasts,
and interactions such as `Price*ShelveLoc`, give the same columns up to
order.

```{code-cell} ipython3
def same_columns(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return A.shape == B.shape and all(any(np.allclose(a, b) for b in B.T) for a in A.T)

same_columns(MS(['Price', 'ShelveLoc', ('Price', 'ShelveLoc')], intercept=True).fit_transform(Carseats),
             patsy.dmatrix('Price*ShelveLoc', Carseats))
```

## Defaults that differ

| | `ModelSpec` | `ColumnTransformer` | patsy | formulaic |
|---|---|---|---|---|
| a column of ones (intercept) | yes, with `intercept=True`, the default until ISLP 0.7 | no: `sklearn` estimators fit their own | yes | yes |
| missing values | propagate as `NaN` rows | passed through; `OneHotEncoder` codes a missing level as a level of its own | rows dropped | rows dropped |
| unused levels of a categorical | dropped | dropped (`OneHotEncoder` learns the levels it sees) | column of zeros | column of zeros |
| level unseen when fit | error | error (`OneHotEncoder`'s default) | error | encoded as all zeros, with a warning |
| string column, not categorical | error: use `astype('category')` or `contrast()` | as its transformer: `'passthrough'` keeps the strings | treated as categorical | treated as categorical |
| interactions without main effects | product of each term's coding | (no interactions) | recoded to keep the model full rank | recoded to keep the model full rank |

`ModelSpec`'s column of ones is set to change: `intercept=True` is its
default only until ISLP 0.7, when `ModelSpec` moves to its own package
and the default becomes `intercept=False`, as for a `ColumnTransformer`.
Pass `intercept=True` for statsmodels' models (`OLS`, `GLM`, ...), which
need the column, and `intercept=False` for `sklearn` and glmstar
estimators, which fit their own intercept; the examples on this page
pass it explicitly for this reason.

Ordered categoricals are coded with contrasts by `ModelSpec`, patsy and
formulaic. `ModelSpec` can
instead score one by the position of its level with `ordinal()`, which
requires an explicit order: an ordered categorical (or a polars `Enum`),
or `levels=[...]`.

## Where the others go further

- **Syntax.** A formula is shorter than the equivalent list of terms.
  `ModelSpec` relies on Python to generate lists of terms instead.
- **R's coding rules.** Both recode the terms of a formula, e.g. an
  interaction without its main effects, to keep the model matrix full
  rank, as R does; `ModelSpec` codes each term as given.
- **Native backends.** formulaic materializes a model matrix from a
  polars or pyarrow frame directly; `ModelSpec` converts the columns it
  uses to `pandas` first, copying them.
- **Speed and memory.** For 2,000,000 rows, two numeric columns and a
  categorical of 50 levels (a dense model matrix of 0.83 GB), formulaic
  built the model matrix in 0.1 s with 0.9 GB of memory besides the data;
  `ModelSpec` and a `ColumnTransformer` took about 0.5 s and 1.7 GB,
  holding the largest block of columns and the model matrix at once.
- **No extra dependency.** A `ColumnTransformer` is part of `sklearn`,
  with its own conventions for naming columns
  (`get_feature_names_out`) and choosing them (`make_column_selector`).
