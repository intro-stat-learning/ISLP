# The default `intercept=None` of ModelSpec: deprecated, it behaves as
# True but warns at fit, as the default will change to False

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from ISLP import load_data
from ISLP.models import ModelSpec as MS


@pytest.fixture(scope='module')
def Carseats():
    return load_data('Carseats')


TERMS = ['Price', 'ShelveLoc']


def test_default_warns_once_per_fit(Carseats):
    spec = MS(TERMS)
    assert spec.intercept is None
    with pytest.warns(FutureWarning, match='will change to False') as record:
        spec.fit(Carseats)
    assert len([w for w in record if issubclass(w.category, FutureWarning)]) == 1
    # transform and the other methods do not warn again
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        X = spec.transform(Carseats)
        names = spec.names
        list(spec.build_sequence(Carseats))
        spec.build_submodel(Carseats, spec.terms_[:1])
    assert spec.intercept is None


def test_default_behaves_as_True(Carseats):
    with pytest.warns(FutureWarning):
        spec = MS(TERMS).fit(Carseats)
    X = spec.transform(Carseats)
    X_true = MS(TERMS, intercept=True).fit_transform(Carseats)
    assert list(X.columns) == list(X_true.columns)
    assert X.columns[0] == 'intercept'
    np.testing.assert_allclose(X['intercept'], 1)
    np.testing.assert_allclose(X, X_true)
    assert spec.names == ['intercept'] + TERMS
    assert spec.column_map_['intercept'] == slice(0, 1)
    D = spec.build_submodel(Carseats, spec.terms_[:1])
    assert list(D.columns) == ['intercept', 'Price']
    seq = list(spec.build_sequence(Carseats))
    assert list(seq[0].columns) == ['intercept']


@pytest.mark.parametrize('intercept', [True, False])
def test_explicit_does_not_warn(Carseats, intercept):
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        spec = MS(TERMS, intercept=intercept)
        X = spec.fit_transform(Carseats)
        spec.names
        list(spec.build_sequence(Carseats))
    assert ('intercept' in X.columns) == intercept
    assert ('intercept' in spec.names) == intercept


@pytest.mark.parametrize('intercept', [True, False])
def test_clone_cross_val_score_explicit_does_not_warn(Carseats, intercept):
    spec = MS(TERMS, intercept=intercept)
    pipe = make_pipeline(spec, LinearRegression(fit_intercept=not intercept))
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        clone(spec).fit(Carseats)
        scores = cross_val_score(pipe, Carseats, Carseats['Sales'], cv=3,
                                 error_score='raise')
    assert np.all(np.isfinite(scores))


def test_clone_get_params_preserve_None(Carseats):
    spec = MS(TERMS)
    assert spec.get_params()['intercept'] is None
    with warnings.catch_warnings():
        # clone re-runs __init__: no warning until fit
        warnings.simplefilter('error', FutureWarning)
        cloned = clone(spec)
    assert cloned.get_params()['intercept'] is None
    with pytest.warns(FutureWarning):
        spec.fit(Carseats)
    # fitting does not resolve the parameter in place
    assert spec.get_params()['intercept'] is None
    assert clone(spec).get_params()['intercept'] is None
