# ModelSpec is modelmatrix's: its imports from ISLP, and model selection
# with its terms (ModelSpec itself is tested in modelmatrix)

import numpy as np, pandas as pd
import pytest
import statsmodels.api as sm

import modelmatrix
from ISLP.models import (ModelSpec as MS,
                         Stepwise,
                         sklearn_selected,
                         poly,
                         wildcard)

bug = lambda reason: pytest.mark.xfail(strict=True, reason=reason)


@pytest.fixture
def D():
    rng = np.random.default_rng(0)
    n = 60
    D = pd.DataFrame({'x': rng.uniform(0, 10, n),
                      'z': rng.standard_normal(n),
                      'u': rng.standard_normal(n),
                      'g': pd.Categorical(rng.choice(['a', 'b', 'c'], n))})
    D['y'] = D['x'] + 2 * D['z'] + rng.standard_normal(n)
    return D


def test_imports():
    import ISLP.models
    from ISLP.models.model_spec import ModelSpec, Contrast, build_model
    from ISLP.models.columns import Column
    from ISLP.transforms import Poly, NaturalSpline
    for name in modelmatrix.__all__:
        if hasattr(ISLP.models, name):
            assert getattr(ISLP.models, name) is getattr(modelmatrix, name)
    assert ModelSpec is modelmatrix.ModelSpec
    assert Contrast is modelmatrix.Contrast
    assert Column is modelmatrix.Column
    assert Poly is modelmatrix.transforms.Poly
    assert callable(build_model) and NaturalSpline is modelmatrix.transforms.NaturalSpline


def test_intercept_default_is_False(D):
    assert MS(['x']).get_params()['intercept'] is False
    assert list(MS(['x']).fit_transform(D).columns) == ['x']


def _selected(model_spec, D, **kwargs):
    strategy = Stepwise.first_peak(model_spec, direction='forward',
                                   max_terms=len(model_spec.terms_), **kwargs)
    return sklearn_selected(sm.OLS, strategy).fit(D, D['y']).selected_state_


def test_stepwise_with_wildcard(D):
    # the wildcard stands for several terms: the same selection as with
    # the columns given one by one
    explicit = MS(['x', 'z', 'u', 'g'], intercept=True).fit(D)
    dot = MS([wildcard(exclude='y')], intercept=True).fit(D)
    assert len(dot.terms) == 1 and len(dot.terms_) == 4
    assert _selected(dot, D) == _selected(explicit, D)
    assert set(_selected(dot, D)) >= {'x', 'z'}


def test_stepwise_lower_terms_among_the_wildcard(D):
    # a column that the wildcard stands for can be a lower term
    dot = MS([wildcard(exclude='y')], intercept=True).fit(D)
    assert 'u' in _selected(dot, D, lower_terms=['u'])


def test_fixed_steps_backward_with_wildcard(D):
    dot = MS([wildcard(exclude='y')], intercept=True).fit(D)
    strategy = Stepwise.fixed_steps(dot, 1, direction='backward')
    selected = sklearn_selected(sm.OLS, strategy).fit(D, D['y']).selected_state_
    assert len(selected) == len(dot.terms_) - 1


@bug('_postprocess_best reduces state to names; build_submodel cannot resolve derived names')
@pytest.mark.parametrize('term', [poly('z', 2), ('x', 'z')])
def test_stepwise_with_derived_term(D, term):
    M = MS(['x', term, 'g'], intercept=True).fit(D)
    strategy = Stepwise.first_peak(M, direction='forward', max_terms=3)
    sklearn_selected(sm.OLS, strategy).fit(D, D.x + D.z)


def test_best_1sd_picks_smaller_model():
    from ISLP.models.strategy import _postprocess_best_1sd
    class T(str):
        name = property(str)
    results = [((T('a'),), 0, np.array([-1.0, -1.1, -0.9])),
               ((T('a'), T('b')), 1, np.array([-0.98, -0.99, -0.97]))]
    assert _postprocess_best_1sd(results)[0] == ('a',)
