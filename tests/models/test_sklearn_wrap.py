
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import is_classifier, is_regressor
import pytest

from ISLP.models.sklearn_wrap import sklearn_sm, sklearn_selected
from ISLP.models.model_spec import ModelSpec
from ISLP.models.strategy import min_max

@pytest.fixture
def model_setup():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'X1': rng.random(10), 'X2': rng.random(10), 'X3': rng.random(10)})
    y = pd.Series(rng.integers(0, 2, 10)) # For classifier
    model_spec_dummy = ModelSpec(['X1', 'X2', 'X3']).fit(X)
    min_max_strategy_dummy = min_max(model_spec_dummy, min_terms=1, max_terms=2)
    return X, y, model_spec_dummy, min_max_strategy_dummy

def test_OLS_is_regressor():
    model = sklearn_sm(sm.OLS)
    assert model.__sklearn_tags__().estimator_type == 'regressor'
    assert is_regressor(model)

def test_GLM_binomial_is_classifier():
    model = sklearn_sm(sm.GLM, model_args={'family': sm.families.Binomial()})
    assert model.__sklearn_tags__().estimator_type == 'classifier'
    assert is_classifier(model)

def test_GLM_binomial_probit_is_classifier():
    model = sklearn_sm(sm.GLM, model_args={'family': sm.families.Binomial(link=sm.families.links.Probit())})
    assert model.__sklearn_tags__().estimator_type == 'classifier'
    assert is_classifier(model)


def test_selected_OLS_is_regressor(model_setup):
    X, y, model_spec_dummy, min_max_strategy_dummy = model_setup
    model = sklearn_selected(sm.OLS, strategy=min_max_strategy_dummy)
    assert model.__sklearn_tags__().estimator_type == 'regressor'
    assert is_regressor(model)

def test_selected_GLM_binomial_is_classifier(model_setup):
    X, y, model_spec_dummy, min_max_strategy_dummy = model_setup
    model = sklearn_selected(sm.GLM, strategy=min_max_strategy_dummy, model_args={'family': sm.families.Binomial()})
    assert model.__sklearn_tags__().estimator_type == 'classifier'
    assert is_classifier(model)

@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((60, 3)), columns=['X1', 'X2', 'X3'])
    y = 1 + 2 * X['X1'] + rng.standard_normal(60)
    return X, y

def test_fit_returns_self(regression_data):
    X, y = regression_data
    spec = ModelSpec(['X1', 'X2', 'X3'])
    model = sklearn_sm(sm.OLS, spec)
    assert model.fit(X, y) is model
    np.testing.assert_allclose(model.fit(X, y).predict(X),
                               model.results_.fittedvalues)

    strategy = min_max(spec.fit(X), min_terms=1, max_terms=2)
    selected = sklearn_selected(sm.OLS, strategy)
    assert selected.fit(X, y) is selected

def test_OLS_score_is_MSE(regression_data):
    X, y = regression_data
    model = sklearn_sm(sm.OLS, ModelSpec(['X1', 'X2', 'X3'])).fit(X, y)
    resid = y - model.predict(X)
    np.testing.assert_allclose(model.score(X, y), np.mean(resid**2))

    w = np.arange(1, 61)
    np.testing.assert_allclose(model.score(X, y, sample_weight=w),
                               np.sum(w * resid**2) / np.sum(w))

def test_GLM_score_is_average_deviance(regression_data):
    X, _ = regression_data
    rng = np.random.default_rng(1)
    y = rng.poisson(np.exp(0.5 * X['X1']))
    family = sm.families.Poisson()
    model = sklearn_sm(sm.GLM,
                       ModelSpec(['X1', 'X2', 'X3']),
                       model_args={'family': family}).fit(X, y)
    yhat = model.predict(X)
    np.testing.assert_allclose(model.score(X, y),
                               family.deviance(y, yhat) / len(y))

    # an average does not depend on the number of observations
    X2, y2 = pd.concat([X, X]), np.concatenate([y, y])
    np.testing.assert_allclose(model.score(X2, y2), model.score(X, y))

    # integer weights act like repeated observations
    w = np.full(len(y), 2)
    np.testing.assert_allclose(model.score(X, y, sample_weight=w),
                               model.score(X, y))

def test_score_unsupported_model(regression_data):
    X, y = regression_data
    Xd = sm.add_constant(X)
    model = sklearn_sm(sm.Logit).fit(Xd, (y > 1).astype(int))
    with pytest.raises(NotImplementedError):
        model.score(Xd, (y > 1).astype(int))

def test_cross_validate_MSE(regression_data):
    from sklearn.model_selection import cross_validate, KFold
    X, y = regression_data
    spec = ModelSpec(['X1', 'X2', 'X3'])
    cv = KFold(3)
    results = cross_validate(sklearn_sm(sm.OLS, spec), X, y, cv=cv)
    for score, (train, test) in zip(results['test_score'], cv.split(X)):
        fit = sm.OLS(y.iloc[train], spec.fit_transform(X.iloc[train])).fit()
        pred = fit.predict(spec.transform(X.iloc[test]))
        np.testing.assert_allclose(score, np.mean((y.iloc[test] - pred)**2))
