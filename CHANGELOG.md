# Changelog

## 0.7.0 (2026-10-26)

`ModelSpec` has moved to its own package,
[modelmatrix](https://github.com/intro-stat-learning/modelmatrix), which
ISLP now depends on (`modelmatrix>=0.3`).

### Breaking changes

- **`ModelSpec`'s `intercept` defaults to `False`**, as 0.6.0 announced. Pass
  `intercept=True` for statsmodels' models (`OLS`, `GLM`, ...) and for
  `sklearn_sm` and `sklearn_selected` with them; most scikit-learn
  estimators, and glmstar's, fit the intercept themselves. The
  `FutureWarning` of 0.6.0 is gone.

### Changes

- `from ISLP.models import ModelSpec` and the other tools that build model
  matrices are modelmatrix's, as are `ISLP.models.model_spec`,
  `ISLP.models.columns` and `ISLP.transforms`, which now are modelmatrix's
  modules, so imports from them keep working. `ISLP.models` also exports
  modelmatrix's `wildcard` (R's `.`), `Contrast`, `Ordinal` and its
  responses `Surv`, `Binomial`, `Multinomial` and `Outcome`.
- ISLP's tests of `ModelSpec` moved to modelmatrix.

### Bug fixes

- Stepwise selection (`Stepwise`, `min_max`) counts and finds terms among
  `ModelSpec.terms_`, the fitted terms, not `terms`: with a `wildcard`, a
  single term standing for many, the terms were misaligned.

## 0.6.0 (2026-10-26)

`ModelSpec` was audited before it moves to its own package in ISLP 0.7.
This release makes it a conforming scikit-learn transformer, fixes its
categorical encodings, and adds polars and pyarrow input, sparse model
matrices and tuning of the parameters of its terms. Some results change:
see **Changes in behaviour**.

### Deprecations

- **`ModelSpec` moves to its own package in ISLP 0.7.** Every
  `ModelSpec.fit` gives a `FutureWarning` saying so: `from ISLP.models
  import ModelSpec` (and the other tools of `ISLP.models` that build model
  matrices) will keep working, but other import paths, such as
  `ISLP.models.model_spec`, may not. The default of `intercept` changes to
  `False` then, in ISLP 0.7, not in 0.6 as 0.5.2 announced: a `ModelSpec`
  relying on the default gets the same one warning, advising to pass
  `intercept=True` (e.g. for statsmodels) or `intercept=False`. Silence the
  warning with `warnings.filterwarnings('ignore', message='ModelSpec will
  move')` (#69).

### Changes in behaviour

- **Ordered categoricals are coded with contrasts**, in the order of their
  categories, like any other categorical; they were a single column of
  level codes. A score is opt-in, with `ordinal('x')` (now exported from
  `ISLP.models`), `categorical_encoders={'x': Ordinal()}`, or
  `{'ordinal': Ordinal()}` for every ordered `pd.Categorical`. `ordinal()`
  requires an explicit order: an ordered categorical, a numeric column, or
  `levels=[...]`; a string column or an unordered categorical raises
  rather than being scored alphabetically. In the Ch07 lab,
  `MS(['education', poly('age', d)])` gives 4 dummy columns; use
  `ordinal('education')` for the linear term (#62).
- **A string column used in a term raises** a `ValueError` asking to make
  it categorical (`astype('category')` or `contrast()`); it went into the
  model matrix as text. Columns no term uses may be anything (#60).
- **Categorical encodings:** unused categories are dropped (they raised);
  a missing value gives a row of `NaN` (it was coded as the baseline
  level); `contrast(..., None)` and callable contrasts follow the order of
  the categories, as `'drop'` and `'sum'` did (they were sorted). A level
  not seen at fit raises, naming it (#60).
- **The dense model matrix is all `float64`** for `pd.DataFrame` input, as
  for `np.ndarray` input; an integer column was `int64` (#67).
- **Stricter input:** 1-d, empty, complex and sparse `X` raise with
  scikit-learn's messages, and an `np.ndarray` with a different number of
  columns at transform raises; a `pd.DataFrame` is matched by name, as
  before (#61). `Column.get_columns` of a `Column` whose encoder is not fit
  raises, asking for `fit=True`: fitting a `ModelSpec` fits a copy of its
  terms (#59, #61).

### New features

- **A fitted `ModelSpec` can be fit again**, cloned and refit, used in
  several specifications, and cross-validated: `fit` fits copies of the
  encoders of its terms, leaving `terms` unchanged. A model fit elsewhere
  is a term with `FrozenEstimator` (#59).
- **Tune the parameters of the terms**, e.g. `GridSearchCV(pipe,
  {'modelspec__terms__0__df': [2, 4, 8]})` for `ns('x', df=4)`; parts of
  interactions are `terms__<i>__<j>__<parameter>`, and a named term
  `terms__<name>__<parameter>`. Terms made by `poly()`, `ns()`, `bs()` and
  `pca()` are made again, so their names follow (`ns(x, df=8)`) (#68).
- **polars, pyarrow and other dataframes** (by narwhals) as `X`; the model
  matrix is returned in the same backend. A polars `Enum` keeps its order;
  the levels of an unordered categorical are sorted, as pandas infers them
  (#63).
- **Sparse model matrices** with `ModelSpec(sparse_output=True)`: a
  `scipy.sparse.csr_matrix`, categorical encodings and their interactions
  built sparse throughout. For 200,000 rows and a categorical with 20,000
  levels, 44 MB where the dense design would be 32 GB (#64).
- **scikit-learn conformance:** `get_feature_names_out` (and so
  `set_output`), `n_features_in_` and `feature_names_in_`, defaults
  `terms=()` and `categorical_encoders=None`, and the `allow_nan` tag.
  `ModelSpec()` passes scikit-learn's `check_estimator`, but for one check
  documented in the tests (#61).

### Bug fixes

- **Refitting** a `ModelSpec`, `clone` of a fitted one, two specifications
  sharing terms, and cross-validation of a fitted one raised "encoder has
  already been fit previously"; `contrast()` terms kept the levels of their
  first fit (#59).
- **`BSpline.fit` changed its parameters** `lower_bound` and `upper_bound`,
  so a refit used the bounds of the first fit (#59).
- **`ordinal()` sorted levels alphabetically** (`hi < lo < mid`): it used
  `OrdinalEncoder` on values stripped of their dtype (#62).
- **`set_params(categorical_encoders=...)` was ignored**, and
  `check_is_fitted` passed on an unfit `ModelSpec` (#61, also fixed in
  #48).
- **`Contrast` and `Ordinal` flattened a 2-column array** into one column
  (#61); an encoder with sparse output (e.g. `OneHotEncoder()`) gave a 0-d
  object array (#64).
- **`build_sequence` with a boolean column**: the designs kept it boolean,
  so an ANOVA over them failed (#63).
- **Predicting a single row** with `poly()`, `ns()` or `bs()` raised
  `IndexError`: `Poly`, `BSpline` and `NaturalSpline` made one value 0-d
  with `np.squeeze` (#71).

### Performance

- **`fit` does not build the model matrix**, and the dense model matrix is
  built once, in place. For 2,000,000 rows and a 50-level categorical,
  `fit_transform` takes 0.3s and 1.7 GB, where it took 1.3s and 2.5 GB;
  `fit` alone 0.06s. For a polars or pyarrow frame only the columns the
  terms use are converted (#67).

### Documentation

- **A page comparing `ModelSpec` with scikit-learn's `ColumnTransformer`,
  patsy and formulaic**, executed at build time: terms as Python objects,
  features of features, interactions, tuning with `GridSearchCV`, sparse
  model matrices, polars input, and the defaults that differ (#66).
- The guide's section on ordinal variables, and two of its cells broken by
  the stricter fitting (#61, #62). Executed pages do not show warnings
  (#69).

### Packaging

- **Requires `narwhals>=1.26`** (#63).
- CI runs the tests on Linux (Python 3.12) for pull requests, into any
  branch, and on macOS, Linux and Windows for merges into main, the weekly
  schedule and by hand (#65). polars and pyarrow are in the `dev` group
  for the tests (#63); polars in the docs' requirements (#66).

Requires Python 3.11 or later, as 0.5.0.

For earlier releases, see the
[GitHub releases](https://github.com/intro-stat-learning/ISLP/releases).
