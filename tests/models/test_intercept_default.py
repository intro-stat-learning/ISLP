# The default `intercept=None` of ModelSpec: deprecated, it behaves as
# True but warns at fit, as the default will change to False in ISLP 0.7,
# when ModelSpec moves to its own package. Every fit warns of the move;
# only a spec relying on the default gets the advice on `intercept`.

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from ISLP import load_data
from ISLP.models import ModelSpec as MS

ADVICE = 'relies on the default of `intercept`'


def _future_warnings(record):
    return [str(w.message) for w in record if issubclass(w.category, FutureWarning)]


@pytest.fixture(scope='module')
def Carseats():
    return load_data('Carseats')


TERMS = ['Price', 'ShelveLoc']


def test_default_warns_once_per_fit(Carseats):
    spec = MS(TERMS)
    assert spec.intercept is None
    with pytest.warns(FutureWarning, match='default to False') as record:
        spec.fit(Carseats)
    # one warning: of the move, with the advice on intercept
    messages = _future_warnings(record)
    assert len(messages) == 1
    assert messages[0].startswith('ModelSpec will move') and ADVICE in messages[0]
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
def test_explicit_warns_only_of_move(Carseats, intercept):
    spec = MS(TERMS, intercept=intercept)
    with pytest.warns(FutureWarning) as record:
        spec.fit(Carseats)
    messages = _future_warnings(record)
    assert len(messages) == 1
    assert messages[0].startswith('ModelSpec will move') and ADVICE not in messages[0]
    assert "filterwarnings('ignore', message='ModelSpec will move')" in messages[0]
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        X = spec.transform(Carseats)
        spec.names
        list(spec.build_sequence(Carseats))
    assert ('intercept' in X.columns) == intercept
    assert ('intercept' in spec.names) == intercept


@pytest.mark.parametrize('intercept', [True, False])
def test_clone_cross_val_score_explicit_no_advice(Carseats, intercept):
    # the move warning can be silenced; nothing else then warns
    spec = MS(TERMS, intercept=intercept)
    pipe = make_pipeline(spec, LinearRegression(fit_intercept=not intercept))
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        warnings.filterwarnings('ignore', message='ModelSpec will move')
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
