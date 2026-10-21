"""
`ISLP.gam.GAM`: glmstar's GAM (`glmnet.gam.GAM`), with a model given by a
ModelSpec.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from glmnet.gam import GAM as _GAM

from .model_spec import as_term, terms_from_spec


@dataclass
class GAM(_GAM):
    """
    glmstar's generalized additive model (`glmnet.gam.GAM`: local scoring
    as R's `gam`, or the joint solve and REML/ML as mgcv; see its
    documentation), with the model given either by a list of terms, as
    there, or by a ModelSpec (`model_spec`).

    A ModelSpec's smooth features (`ISLP.gam.s`, `lo`, `re`) are smoothed,
    and each of its other terms (columns, contrasts, `ns()`, `poly()`, ...)
    is fitted as one parametric block. The GAM has its own intercept, so a
    spec whose `intercept` is not given is fitted without the column of
    ones.

    Parameters
    ----------
    model_spec: ModelSpec
        The model (instead of `terms`): e.g.
        `MS([s('year', df=4), s('age', df=5), 'education'])`. It is fitted
        (a clone) on the `X` of `fit`, and transforms the `X` of `predict`
        and `partial_effect`.

    The other parameters and the attributes are those of `glmnet.gam.GAM`;
    also `model_spec_`, the fitted clone of `model_spec` (None without one).

    Examples
    --------
    >>> from ISLP import load_data
    >>> from ISLP.models import ModelSpec as MS
    >>> from ISLP.gam import GAM, s
    >>> Wage = load_data('Wage')
    >>> gam = GAM(model_spec=MS([s('year', df=4), s('age', df=5), 'education']))
    >>> gam.fit(Wage, Wage['wage']).plot()
    """
    model_spec: object = None

    def _model_terms(self, X):
        if self.model_spec is None:
            self.model_spec_ = None
            return [as_term(term) for term in self.terms], X
        if self.terms:
            raise ValueError('give one of terms and model_spec')
        spec = clone(self.model_spec)
        if spec.intercept is None:
            # (the GAM fits the intercept: no column of ones, and no warning)
            spec.set_params(intercept=False)
        self.model_spec_ = spec.fit(X)
        X = self.model_spec_.transform(X)
        columns = X.columns if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])
        return terms_from_spec(self.model_spec_, columns), X

    def _design(self, X):
        spec = getattr(self, 'model_spec_', None)
        return X if spec is None else spec.transform(X)
