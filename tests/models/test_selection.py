from itertools import product

import pytest

import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

from ISLP.models import ModelSpec as MS
from ISLP.models.strategy import min_max, Stepwise, validator_from_constraints
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

    
