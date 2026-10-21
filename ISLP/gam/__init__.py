"""
Generalized additive models, from glmstar's `glmnet.gam` (local scoring as
R's `gam::gam`, the joint solve and REML/ML as mgcv), with models given by
a ModelSpec: smooth features (`s()`, `lo()`, `re()`) next to ordinary
columns, contrasts and bases.

>>> from ISLP.models import ModelSpec as MS
>>> from ISLP.gam import GAM, s, lo
>>> GAM(model_spec=MS([s('year', df=4), lo('age', span=0.7), 'education'])).fit(Wage, Wage['wage'])

or with a list of terms, each naming a column:

>>> from ISLP.gam import GAM, s, factor
>>> GAM(terms=[s('age', df=4), s('year', lam=1e-3), factor('education')]).fit(X, y)

`ISLP.gam.s`, `lo` and `re` are ModelSpec features that can also be given
as terms; the rest is `glmnet.gam`'s (see its documentation and references).
Smoothing parameters by REML or ML (`GAM(method='REML')`) need JAX:
`pip install ISLP[gam]`.
"""
from glmnet.gam import (GAMControl,
                        Term,
                        SplineTerm,
                        LoessTerm,
                        SlopeTerm,
                        FactorTerm,
                        ColumnsTerm,
                        RandomEffectTerm,
                        linear,
                        factor,
                        plot_term,
                        plot_terms)
from .model_spec import s, lo, re, SmoothColumn, GroupColumns, as_term, terms_from_spec
from .estimator import GAM
