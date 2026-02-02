
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
    X = pd.DataFrame({'X1': np.random.rand(10), 'X2': np.random.rand(10), 'X3': np.random.rand(10)})
    y = pd.Series(np.random.randint(0, 2, 10)) # For classifier
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
