from itertools import product

import pytest

import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

from ISLP.models import ModelSpec as MS
from ISLP.models.strategy import min_max, Stepwise, validator_from_constraints
from ISLP.models.generic_selector import FeatureSelector

def _check_bounds(selector, lower_terms=None, upper_terms=None):
    # fixed_steps returns a state of Features, the other strategies return names
    selected = set(getattr(t, 'name', t) for t in selector.selected_state_)
    if lower_terms is not None:
        assert set(lower_terms) <= selected
    if upper_terms is not None:
        assert selected <= set(upper_terms)
    assert selector.selected_state_ in selector.results_

def test_min_max():

    rng = np.random.default_rng(0)
    n, p = 100, 7
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:p])
    D['A'] = pd.Categorical(rng.choice(range(5), (n,), replace=True))

    model_spec = MS(list(D.columns))
    model_spec.fit(D)

    strategy = min_max(model_spec,
                       min_terms=1,
                       max_terms=len(model_spec.terms),
                       lower_terms=['B','C'],
                       upper_terms=['B','C', 'D', 'E'])

    min_max_selector = FeatureSelector(LinearRegression(),
                                       strategy,
                                       cv=3)

    min_max_selector.fit(D, Y)

    _check_bounds(min_max_selector,
                  lower_terms=['B','C'],
                  upper_terms=['B','C', 'D', 'E'])

def test_step():

    n, p = 100, 7
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:p])
    D['A'] = pd.Categorical(rng.choice(range(5), (n,), replace=True))

    model_spec = MS(list(D.columns))
    model_spec.fit(D)

    for (direction,
         upper_terms,
         lower_terms) in product(['forward',
                                  'backward',
                                  'both'],
                                 [None, ['B','C', 'D', 'E']],
                                 [None, ['B', 'C']]):

        strategy = Stepwise.first_peak(model_spec,
                                       direction=direction,
                                       min_terms=1,
                                       max_terms=len(model_spec.terms),
                                       initial_terms=['B','C'],
                                       upper_terms=upper_terms,
                                       lower_terms=lower_terms)
        step_selector = FeatureSelector(LinearRegression(),
                                        strategy,
                                        cv=3)
        step_selector.fit(D, Y)
        _check_bounds(step_selector, lower_terms, upper_terms)

        strategy = Stepwise.first_peak(model_spec,
                                       direction=direction,
                                       min_terms=1,
                                       max_terms=len(model_spec.terms),
                                       initial_terms=['B','C'],
                                       upper_terms=upper_terms,
                                       lower_terms=lower_terms)
        step_selector = FeatureSelector(LinearRegression(),
                                        strategy,
                                        cv=None)
        step_selector.fit(D, Y)
        _check_bounds(step_selector, lower_terms, upper_terms)

        # 4 backward steps from 7 terms targets 3 terms,
        # which cannot be reached starting from ['B','C']
        if direction != 'backward':

            strategy = Stepwise.fixed_steps(model_spec,
                                            4,
                                            direction=direction,
                                            initial_terms=['B','C'],
                                            upper_terms=upper_terms,
                                            lower_terms=lower_terms)

            step_selector = FeatureSelector(LinearRegression(),
                                            strategy,
                                            cv=3)
            step_selector.fit(D, Y)
            _check_bounds(step_selector, lower_terms, upper_terms)

            strategy = Stepwise.fixed_steps(model_spec,
                                            4,
                                            direction=direction,
                                            initial_terms=['B','C'],
                                            upper_terms=upper_terms)

            step_selector = FeatureSelector(LinearRegression(),
                                            strategy,
                                            cv=None)
            step_selector.fit(D, Y)
            _check_bounds(step_selector, None, upper_terms)

def test_fixed_steps_backward():

    n, p = 100, 7
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:p])
    D['A'] = pd.Categorical(rng.choice(range(5), (n,), replace=True))

    model_spec = MS(list(D.columns))
    model_spec.fit(D)

    num_steps = 4
    strategy = Stepwise.fixed_steps(model_spec,
                                        num_steps,
                                        direction='backward')

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    cv=3)
    step_selector.fit(D, Y)

    print("selected", [term.name for term in step_selector.selected_state_])
    assert len(step_selector.selected_state_) == (len(model_spec.terms) - num_steps)

    num_steps = 2
    strategy = Stepwise.fixed_steps(model_spec,
                                    num_steps,
                                    direction='backward')

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    cv=None)
    step_selector.fit(D, Y)

    print("selected", [term.name for term in step_selector.selected_state_])
    assert len(step_selector.selected_state_) == (len(model_spec.terms) - num_steps)

def test_constraint():

    rng = np.random.default_rng(3)
    n, p = 100, 7
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:p])
    D['A'] = pd.Categorical(rng.choice(range(5), (n,), replace=True))

    model_spec = MS(list(D.columns))
    model_spec.fit(D)

    constraints = np.zeros((len(model_spec.terms),)*2)
    constraints[3,4] = 1
    constraints[4,5] = 1
    validator = validator_from_constraints(model_spec,
                                           constraints)
                                           
    strategy = min_max(model_spec,
                       min_terms=1,
                       max_terms=len(model_spec.terms),
                       lower_terms=['B','C'],
                       validator=validator)

    min_max_selector = FeatureSelector(LinearRegression(),
                                       strategy,
                                       cv=3)

    min_max_selector.fit(D, Y)

    _check_bounds(min_max_selector, lower_terms=['B','C'])
    # E requires D and F requires E
    for state in min_max_selector.results_:
        if 'E' in state:
            assert 'D' in state
        if 'F' in state:
            assert 'E' in state

def test_forward_finds_signal():

    rng = np.random.default_rng(0)
    n = 100
    D = pd.DataFrame(rng.standard_normal((n, 7)), columns=list('ABCDEFG'))
    Y = 2 * D['D'] - 2 * D['E'] + rng.standard_normal(n)

    model_spec = MS(list(D.columns))
    model_spec.fit(D)

    states = {}
    for parsimonious in [False, True]:
        strategy = Stepwise.first_peak(model_spec,
                                       direction='forward',
                                       max_terms=len(model_spec.terms),
                                       parsimonious=parsimonious)
        selector = FeatureSelector(LinearRegression(),
                                   strategy,
                                   cv=3)
        selector.fit(D, Y)
        assert {'D', 'E'} <= set(selector.selected_state_)
        states[parsimonious] = selector.selected_state_

    assert len(states[True]) <= len(states[False])
