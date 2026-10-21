"""
glmstar's GAM terms as ModelSpec features: `s()`, `lo()` and `re()` give a
`Feature` whose encoder carries the term (`glmnet.gam`'s), and passes its
column through to the design. A `GAM(model_spec=MS([...]))` smooths those
columns and fits every other term of the spec (columns, contrasts, `ns()`,
`poly()`, ...) as one parametric block.

>>> from ISLP.models import ModelSpec as MS
>>> from ISLP.gam import GAM, s, lo
>>> GAM(model_spec=MS([s('year', df=4), s('age', df=5), 'education'])).fit(Wage, Wage['wage'])
"""
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from glmnet.gam.terms import (SplineTerm, LoessTerm, RandomEffectTerm, ColumnsTerm, Term,
                              _level_codes)

from ..models.model_spec import Feature


class SmoothColumn(TransformerMixin, BaseEstimator):
    """
    The encoder of a smooth term in a ModelSpec: it passes its column through
    unchanged, and carries the smoother (`term`, a `glmnet`-style GAM term)
    for `GAM` to use.
    """

    def __init__(self, term=None):
        self.term = term

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(f'a smooth term takes one column, got shape {X.shape}')
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        return np.asarray(X, float).reshape((-1, 1))


class GroupColumns(TransformerMixin, BaseEstimator):
    """
    The encoder of a random effect term in a ModelSpec: it encodes its first
    column, the groups, by their codes among the levels seen in fitting (-1
    for others), and passes the second (`by`), if any, through. It carries
    the term (`term`) for `GAM`, and the levels (`levels_`).
    """

    def __init__(self, term=None):
        self.term = term

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object).reshape((np.asarray(X).shape[0], -1))
        self.levels_ = np.unique(X[:, 0])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object).reshape((np.asarray(X).shape[0], -1))
        codes = _level_codes(self.levels_, X[:, 0]).astype(float)
        if X.shape[1] == 1:
            return codes[:, None]
        return np.column_stack([codes, X[:, 1].astype(float)])


def _name(label, column, **kwargs):
    args = ', '.join(f'{k}={v}' for k, v in kwargs.items() if v is not None)
    return f'{label}({column}{", " + args if args else ""})'


def _feature(term, name):
    return Feature((term.column,), name, SmoothColumn(term))


def s(column, df=None, lam=None, name=None):
    """
    A smoothing spline term (see `SplineTerm`; `df=4` by default) as a
    ModelSpec feature, or in `GAM(terms=[...])`.
    """
    term = SplineTerm(column, df=df, lam=lam)
    return _feature(term, name or _name('s', column, df=df, lam=lam))


def lo(column, span=0.5, degree=1, name=None):
    """A loess term (see `LoessTerm`) as a ModelSpec feature, or in `GAM(terms=[...])`."""
    term = LoessTerm(column, span=span, degree=degree)
    # the name shows the arguments that differ from the defaults
    return _feature(term, name or _name('lo', column,
                                        span=None if span == 0.5 else span,
                                        degree=None if degree == 1 else degree))


def re(column, by=None, lam=None, df=None, name=None):
    """
    A random effect term (see `RandomEffectTerm`): a random intercept per
    group of `column`, or with `by`, a random slope of the column `by` per
    group; as a ModelSpec feature, or in `GAM(terms=[...])`.
    """
    term = RandomEffectTerm(column, by=by, lam=lam, df=df)
    variables = (column,) if by is None else (column, by)
    return Feature(variables, name or _name('re', column, by=by, lam=lam, df=df),
                   GroupColumns(term))


def smooth_term(feature):
    """The GAM term of a smooth feature (`s()`, `lo()`, `re()`), or None."""
    if isinstance(feature, Feature) and isinstance(feature.encoder, (SmoothColumn, GroupColumns)):
        return feature.encoder.term
    return None


def as_term(term):
    """A GAM term from a term or a smooth feature (in `GAM(terms=[...])`)."""
    if isinstance(term, Term):
        return term
    smooth = smooth_term(term)
    if smooth is None:
        raise ValueError(f'{term!r} is not a GAM term (s(), lo(), linear(), factor())')
    smooth = deepcopy(smooth)
    smooth.label = term.name
    return smooth


def terms_from_spec(spec, design_columns):
    """
    The GAM terms of a fitted ModelSpec, on the columns of its design: each
    smooth feature its term, on its column; every other term a block of its
    design columns (`ColumnsTerm`). The intercept is left to the GAM.
    """
    terms = []
    for term, term_ in zip(spec.terms, spec.terms_):
        cols = list(design_columns[spec.column_map_[term]])
        smooth = smooth_term(term_)
        if smooth is not None:
            smooth = deepcopy(smooth)
            smooth.column = cols[0]
            if isinstance(smooth, RandomEffectTerm):
                # the groups by their codes: their names from the encoder
                smooth.level_names = term_.encoder.levels_
                if len(cols) > 1:
                    smooth.by = cols[1]
            smooth.label = term_.name
            terms.append(smooth)
        else:
            terms.append(ColumnsTerm(cols, label=term_.name))
    return terms
