from itertools import product

import pytest

import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

from ISLP.models import ModelSpec as MS
from ISLP.models.model_spec import Feature
from ISLP.models.strategy import (min_max,
                                  Stepwise,
                                  validator_from_constraints,
                                  _postprocess_best_1sd)
from ISLP.models.generic_selector import FeatureSelector

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
                       upper_terms=['B','C', 'D', 'H', 'I'])

    min_max_selector = FeatureSelector(LinearRegression(),
                                       strategy,
                                       cv=3)

    min_max_selector.fit(D, Y)

    print(min_max_selector.results_)
    print('selected')
    print(min_max_selector.selected_state_)
    print(min_max_selector.results_[min_max_selector.selected_state_])

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
                                 [None, ['B','C', 'D', 'H', 'I']],
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

        print(step_selector.results_)
        print(step_selector.selected_state_)
        print('huh2')

        strategy = Stepwise.fixed_steps(model_spec,
                                        4,
                                        direction=direction,
                                        initial_terms=['B','C'],
                                        upper_terms=upper_terms)

        step_selector = FeatureSelector(LinearRegression(),
                                        strategy,
                                        cv=None)
        step_selector.fit(D, Y)


        print(step_selector.results_)
        print(step_selector.selected_state_)
        print('huh')

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

def test_first_peak_backward_default_start():

    # issue #33: with no initial_terms, backward search
    # should start from the full model, not the empty one

    rng = np.random.default_rng(0)
    n = 100
    D = pd.DataFrame(rng.standard_normal((n, 7)), columns=list('ABCDEFG'))
    Y = 2 * D['D'] - 2 * D['E'] + rng.standard_normal(n)

    model_spec = MS(list(D.columns))
    model_spec.fit(D)

    strategy = Stepwise.first_peak(model_spec,
                                   direction='backward',
                                   max_terms=len(model_spec.terms))
    selector = FeatureSelector(LinearRegression(),
                               strategy,
                               cv=3)
    selector.fit(D, Y)
    assert {'D', 'E'} <= set(selector.selected_state_)
    # the full model was scored
    assert tuple('ABCDEFG') in selector.results_

def test_postprocess_best_1sd():

    # ('A','B') has the best average score, but ('A',) is smaller
    # and within one standard error of it

    A, B = [Feature((c,), c, None) for c in 'AB']
    results = [((A,), 1, np.array([-1.0, -0.5, -1.5])),
               ((A, B), 2, np.array([-0.9, -0.9, -0.9]))]
    best_state, scores = _postprocess_best_1sd(results)
    assert best_state == ('A',)
    np.testing.assert_allclose(scores[('A', 'B')], -0.9)

    # ('A',) is not within one standard error so ('A','B') is chosen

    results = [((A,), 1, np.array([-1.0, -1.0, -1.0])),
               ((A, B), 2, np.array([-0.9, -0.9, -0.9]))]
    best_state, _ = _postprocess_best_1sd(results)
    assert best_state == ('A', 'B')

    

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

    print(min_max_selector.results_)
    print('selected')
    print(min_max_selector.selected_state_)
    print(min_max_selector.results_[min_max_selector.selected_state_])

    
