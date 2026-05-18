"""
Model selection strategies
==========================

This module defines search strategies to be used in generic
stepwise model selection.

"""

# Jonathan Taylor 2021
# mlxtend Machine Learning Library Extensions
#
# Objects describing search strategy
# Author: Jonathan Taylor <jonathan.taylor@stanford.edu>
# 

from typing import NamedTuple, Any, Callable
from itertools import chain, combinations
from functools import partial

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .columns import (_get_column_info,
                     Column,
                     _categorical_from_df,
                     _check_categories)

class Strategy(NamedTuple):

    """
    initial_state: object
        Initial state of feature selector.
    candidate_states: callable
        Callable taking single argument `state` and returning
        candidates for next batch of scores to be calculated.
    build_submodel: callable
        Callable taking two arguments `(X, state)` that returns
        model matrix represented by `state`.
    check_finished: callable
        Callable taking three arguments 
        `(results, best_state, batch_results)` which determines if
        the state generator should step. Often will just check
        if there is a better score than that at current best state
        but can use entire set of results if desired.
    postprocess: callable
        Callable to postprocess the results after selection
        procedure terminates.
    """

    initial_state: Any
    candidate_states: Callable
    build_submodel: Callable
    check_finished: Callable
    postprocess: Callable

      
class MinMaxCandidates(object):

    def __init__(self,
                 model_spec,
                 min_terms=0,
                 max_terms=0,
                 lower_terms=None,
                 upper_terms=None,
                 validator=None):
        """
        Parameters
        ----------
        model_spec: ModelSpec
            ModelSpec describing the terms in the model.
        min_terms: int (default: 0)
            Minumum number of terms to select
        max_terms: int (default: 0)
            Maximum number of terms to select
        lower_terms: [Feature]
            Subset of terms to keep: smallest model.
        upper_terms: [Feature]
            Largest possible model.
        validator: callable
            Callable taking a single argument: state,
            returning whether this is a valid state.
  
        """

        self.model_spec = model_spec
        nterms = len(self.model_spec.terms)

        if (not isinstance(max_terms, int) or
                (max_terms > nterms or max_terms < 0)):
            raise AttributeError('max_terms must be'
                                 ' smaller than %d and >= 0' %
                                 (nterms + 1))

        if (not isinstance(min_terms, int) or
                (min_terms > nterms or min_terms < 0)):
            raise AttributeError('min_terms must be'
                                 ' smaller than %d and >= 0'
                                 % (nterms + 1))

        if max_terms < min_terms:
            raise AttributeError('min_terms must be <= max_terms')

        self.min_terms, self.max_terms = min_terms, max_terms

        self._have_already_run = False

        if lower_terms:
            lower_terms_ = []
            for term in lower_terms:
                mm_terms = list(self.model_spec.terms)
                if term in mm_terms:
                    idx = mm_terms.index(term)
                    term = self.model_spec.terms_[idx]
                lower_terms_.append(term)
            self.lower_terms = set(lower_terms_)
        else:
            self.lower_terms = set([])

        if upper_terms:
            upper_terms_ = []
            for term in upper_terms:
                mm_terms = list(self.model_spec.terms)
                if term in mm_terms:
                    idx = mm_terms.index(term)
                    term = self.model_spec.terms_[idx]
                upper_terms_.append(term)
            self.upper_terms = set(upper_terms_)
        else:
            self.upper_terms = set(self.model_spec.terms_)

        if not self.lower_terms.issubset(self.upper_terms):
            raise ValueError('lower_terms should be a subset of upper_terms')
        
        self.validator = validator
            
    def candidate_states(self, state):
        """
        Produce candidates for fitting.

        Parameters
        ----------

        state: ignored

        Returns
        -------
        candidates: iterator
            A generator of (indices, label) where indices
            are columns of X and label is a name for the 
            given model. The iterator cycles through
            all combinations of columns of nfeature total
            of size ranging between min_terms and max_terms.
            If appropriate, restricts combinations to include
            a set of fixed terms.
            Models are labeled with a tuple of the feature names.
            The names of the columns default to strings of integers
            from range(nterms).

        """

        check_is_fitted(self.model_spec)
        terms = self.model_spec.terms_

        if self.validator is None:
            is_valid = lambda c: True
        else:
            is_valid = self.validator

        if not self._have_already_run:
            self._have_already_run = True # maybe could be done with a StopIteration on candidates?
            def chain_(i):
                return (c for c in combinations(terms, r=i)
                        if (self.lower_terms.issubset(c) and
                            self.upper_terms.issuperset(c) and
                            is_valid(c)))
            
            candidates = chain.from_iterable(chain_(i) for i in
                                             range(self.min_terms,
                                                   self.max_terms+1))
            return candidates
        
    def check_finished(self,
                       results,
                       path,
                       best,
                       batch_results):
        """
        Check if we should continue or not. 
        For exhaustive search we stop because
        all models are fit in a single batch.
        """
        new_best = (None, None, None)
        batch_best_score = -np.inf
        
        for (state, iteration, scores) in batch_results:
            avg_score = np.nanmean(scores)
            if avg_score > batch_best_score:
                new_best = (state, iteration, scores)
                batch_best_score = np.nanmean(scores)

        return new_best, True


class Stepwise(MinMaxCandidates):

    """
    Parameters
    ----------
    model_spec: ModelSpec
        ModelSpec describing the terms in the model.
    direction: str
        One of ['forward', 'backward', 'both']
    min_terms: int (default: 1)
        Minumum number of terms to select
    max_terms: int (default: 1)
        Maximum number of terms to select
    lower_terms: [Feature]
        Subset of terms to keep: smallest model.
    upper_terms: [Feature]
        Largest possible model.
    constraints: {array-like} (optional), shape [n_terms, n_terms]
        Boolean matrix decribing a dag with [i,j] nonzero implying that j is
        a child of i (i.e. there is an edge i->j). 
        All search candidates are checked for validity: i.e.
        the parent of each term in a candidate must be included
        in the set of terms.
    """

    def __init__(self,
                 model_spec,
                 direction='forward',
                 min_terms=1,
                 max_terms=1,
                 lower_terms=None,
                 upper_terms=None,
                 validator=None):

        self.direction = direction
        MinMaxCandidates.__init__(self,
                                  model_spec=model_spec,
                                  min_terms=min_terms,
                                  max_terms=max_terms,
                                  lower_terms=lower_terms,
                                  upper_terms=upper_terms,
                                  validator=validator)
            
    def candidate_states(self, state):
        """
        Produce candidates for fitting.
        For stepwise search this depends on the direction.

        If 'forward', all columns not in the current state
        are added (maintaining an upper limit on the number of columns 
        at *self.max_terms*).

        If 'backward', all columns not in the current state
        are dropped (maintaining a lower limit on the number of columns 
        at *self.min_terms*).

        All candidates include *self.lower_terms* if any.
        
        Parameters
        ----------

        state: ignored

        Returns
        -------
        candidates: iterator
            A generator of (indices, label) where indices
            are columns of X and label is a name for the 
            given model. The iterator cycles through
            all combinations of columns of nfeature total
            of size ranging between min_terms and max_terms.
            If appropriate, restricts combinations to include
            a set of fixed terms.
            Models are labeled with a tuple of the feature names.
            The names of the columns default to strings of integers
            from range(nterms).

        """

        state = set(state)
        terms = self.model_spec.terms_
        lower_terms = self.lower_terms
        upper_terms = self.upper_terms
        
        if self.validator is None:
            is_valid = lambda c: True
        else:
            is_valid = self.validator

        if len(state) < self.max_terms: # union
            forward = (tuple(sorted(state | set([c])))
                       for c in terms if (c not in state and
                                          lower_terms.issubset(state | set([c])) and
                                          upper_terms.issuperset(state | set([c])) and
                                          is_valid(state | set([c]))))
        else:
            forward = []

        if len(state) > self.min_terms: # symmetric difference
            backward = (tuple(sorted(state ^ set([c])))
                        for c in terms if (c in state and
                                           lower_terms.issubset(state ^ set([c])) and
                                           upper_terms.issuperset(state ^ set([c])) and
                                           is_valid(state ^ set([c]))))
        else:
            backward = []

        if self.direction == 'forward':
            return forward
        elif self.direction == 'backward':
            return backward
        else:
            return chain.from_iterable([forward, backward])

    @staticmethod
    def first_peak(model_spec,
                   direction='forward',
                   min_terms=1,
                   max_terms=1,
                   random_state=0,
                   lower_terms=[],
                   upper_terms=[],
                   initial_terms=[],
                   validator=None,
                   parsimonious=False):
        """
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        direction: str
            One of ['forward', 'backward', 'both']
        min_terms: int (default: 1)
            Minumum number of terms to select
        max_terms: int (default: 1)
            Maximum number of terms to select
        lower_terms: [Feature]
            Subset of terms to keep: smallest model.
        upper_terms: [Feature]
            Largest possible model.
        initial_terms: column identifiers, default=[]
            Subset of terms to be used to initialize when direction
            is `both`. If None defaults to behavior of `forward`.
            where `self.columns` will correspond to columns if X is a `pd.DataFrame`
            or an array of integers if X is an `np.ndarray`
        validator: callable
            Callable taking a single argument: state,
            returning whether this is a valid state.
        parsimonious: bool
            If True, use the 1sd rule: among the shortest models
            within one standard deviation of the best score
            pick the one with the best average score. 

        Returns
        -------

        initial_state: tuple
            (column_names, feature_idx)

        state_generator: callable
            Object that proposes candidates
            based on current state. Takes a single 
            argument `state`

        build_submodel: callable
            Candidate generator that enumerate
            all valid subsets of columns.

        check_finished: callable
            Check whether to stop. Takes two arguments:
            `best_result` a dict with keys of scores
            and `state`.

        """

        check_is_fitted(model_spec)

        step = Stepwise(model_spec,
                        direction=direction,
                        min_terms=min_terms,
                        max_terms=max_terms,
                        lower_terms=lower_terms,
                        upper_terms=upper_terms,
                        validator=validator)

        # pick an initial state

        if initial_terms is not None:
            initial_terms_ = []
            for term in initial_terms:
                mm_terms = list(model_spec.terms)
                if term in mm_terms:
                    idx = mm_terms.index(term)
                    term = model_spec.terms_[idx]
                initial_terms_.append(term)
            initial_state = tuple(initial_terms_)
        else:
            initial_state = () if direction == 'forward' else list(model_spec.terms)

        if not parsimonious:
            _postprocess = _postprocess_best
        else:
            _postprocess = _postprocess_best_1sd

        return Strategy(initial_state,
                        step.candidate_states,
                        model_spec.build_submodel,
                        first_peak,
                        _postprocess)

    @staticmethod
    def fixed_steps(model_spec,
                    n_steps,
                    direction='forward',
                    lower_terms=[],
                    upper_terms=[],
                    initial_terms=[],
                    validator=None):
        """
        Strategy that stops first time
        a given model size is reached.

        Parameters
        ----------
        model_spec: ModelSpec
            ModelSpec describing the terms in the model.
        n_steps: int
            How many steps to take in the search?
        direction: str
            One of ['forward', 'backward', 'both']
        min_terms: int (default: 0)
            Minumum number of terms to select
        max_terms: int (default: None)
            Maximum number of terms to select.
            If None defaults to number of terms in *model_spec*.
        lower_terms: [Feature]
            Subset of terms to keep: smallest model.
        upper_terms: [Feature]
            Largest possible model.
        initial_terms: column identifiers, default=[]
            Subset of terms to be used to initialize.

        Returns
        -------

        strategy : NamedTuple

        """

        n_terms = n_steps if direction == 'forward' else len(list(model_spec.terms)) - n_steps
        step = Stepwise(model_spec,
                        direction=direction,
                        min_terms=n_terms,
                        max_terms=n_terms,
                        lower_terms=lower_terms,
                        upper_terms=upper_terms,
                        validator=validator)

        # pick an initial state

        if initial_terms is not None and initial_terms != []:
            initial_terms_ = []
            mm_terms = list(model_spec.terms)
            for term in initial_terms:
                if term in mm_terms:
                    idx = mm_terms.index(term)
                    term = model_spec.terms_[idx]
                initial_terms_.append(term)
            initial_state = tuple(initial_terms_)
        else:
            initial_state = () if direction == 'forward' else list(model_spec.terms_)

        if not step.lower_terms.issubset(initial_state):
            raise ValueError('initial_state should contain %s' % str(step.lower_terms))

        if not step.upper_terms.issuperset(initial_state):
            raise ValueError('initial_state should be contained in %s' % str(step.upper_terms))

        return Strategy(initial_state,
                        step.candidate_states,
                        model_spec.build_submodel,
                        partial(fixed_steps, n_terms),
                        partial(_postprocess_fixed_steps, n_terms))
    

def min_max(model_spec,
            min_terms=1,
            max_terms=1,
            lower_terms=None,
            upper_terms=None,
            validator=None,
            parsimonious=False):
    """
    Parameters
    ----------
    model_spec: ModelSpec
        ModelSpec describing the terms in the model.
    min_terms: int (default: 1)
        Minumum number of terms to select
    max_terms: int (default: 1)
        Maximum number of terms to select
    lower_terms: [Feature]
        Subset of terms to keep: smallest model.
    upper_terms: [Feature]
        Largest possible model.
    validator: callable
        Callable taking a single argument: state,
        returning whether this is a valid state.
    parsimonious: bool
        If True, use the 1sd rule: among the shortest models
        within one standard deviation of the best score
        pick the one with the best average score. 

    Returns
    -------

    initial_state: tuple
        (column_names, feature_idx)

    state_generator: callable
        Object that proposes candidates
        based on current state. Takes a single 
        argument `state`

    build_submodel: callable
        Candidate generator that enumerate
        all valid subsets of columns.

    check_finished: callable
        Check whether to stop. Takes two arguments:
        `best_result` a dict with keys of scores.
        and `state`.

    """

    strategy = MinMaxCandidates(model_spec,
                                min_terms=min_terms,
                                max_terms=max_terms,
                                lower_terms=lower_terms,
                                upper_terms=upper_terms,
                                validator=validator)
    
    # if any categorical terms or an intercept
    # is included then we must
    # create a new design matrix

    initial_state = tuple(strategy.lower_terms)

    if not parsimonious:
        _postprocess = _postprocess_best
    else:
        _postprocess = _postprocess_best_1sd

    return Strategy(initial_state,
                    strategy.candidate_states,
                    model_spec.build_submodel,
                    strategy.check_finished,
                    _postprocess)


def validator_from_constraints(model_spec,
                               constraints):

    def is_valid(model_spec,
                 constraints,
                 state):

        check_is_fitted(model_spec)
        terms_ = model_spec.terms_

        if constraints.shape != (len(terms_),)*2:
            raise ValueError('constraint should have shape (nterms, nterms)')

        parents_included = []
        for term in state:
            idx = terms_.index(term)
            parents = np.nonzero(constraints[:,idx])[0]
            parents_included.append(np.all([terms_[j] in state for j in parents]))
        return np.all(parents_included)

    return partial(is_valid, model_spec, constraints)


def first_peak(results,
               path,
               best,
               batch_results):
    """
    Check if we should continue or not. 

    For first_peak search we stop if we cannot improve
    over our current best score.

    """
    new_best = (None, None, None)
    batch_best_score = -np.inf

    for state, iteration, scores in batch_results:
        avg_score = np.nanmean(scores)
        if avg_score > batch_best_score:
            new_best = (state, iteration, scores)
            batch_best_score = avg_score

    any_better = batch_best_score > np.nanmean(best[2])
    return new_best, not any_better

def fixed_steps(n_steps,
                results,
                path,
                best,
                batch_results):
    """
    Check if we should continue or not. 

    For first_peak search we stop if we cannot improve
    over our current best score.

    """
    new_best = (None, None, None)
    batch_best_score = -np.inf

    for state, iteration, scores in batch_results:
        avg_score = np.nanmean(scores)
        if avg_score > batch_best_score:
            new_best = (state, iteration, scores)
            batch_best_score = avg_score

    any_better = batch_best_score > np.nanmean(best[2])
    return new_best, len(new_best[0]) == n_steps

# private functions


def _build_submodel(column_info, X, cols):
    return np.column_stack([column_info[col].get_columns(X, fit=True)[0] for col in cols])

    
def _postprocess_fixed_steps(n_steps, results):
    """
    Find the best state from `results`
    based on `avg_score`.

    Return best state and results
    """

    best_state = None
    best_score = -np.inf

    new_results = {}
    for (state, iteration, scores) in results:
        new_state = tuple(state) # [v.name for v in state])
        avg_score = np.nanmean(scores)
        if avg_score > best_score and len(new_state) == n_steps:
            best_state = new_state
            best_score = avg_score
        new_results[new_state] = avg_score
    return best_state, new_results
    

def _postprocess_best(results):
    """
    Find the best state from `results`
    based on `avg_score`.

    Return best state and results
    """

    best_state = None
    best_score = -np.inf

    new_results = {}
    for (state, iteration, scores) in results:
        new_state = tuple([v.name for v in state])
        avg_score = np.nanmean(scores)
        if avg_score > best_score:
            best_state = new_state
            best_score = avg_score
        new_results[new_state] = avg_score
    
    return best_state, new_results

def _postprocess_best_1sd(results):
    """
    Find the best state from `results`
    based on np.nanmean(scores)

    Find models satisfying the 1sd rule
    and choose the state with best score
    among the smallest such states.

    Return best state and results

    Models are compared by length of state
    """

    best_state = None
    best_score = -np.inf

    for state, iteration, scores in results:
        avg_score = np.nanmean(scores)
        if avg_score > best_score:
            best_state = state
            best_score = avg_score

    states_1sd = []

    for (state, iteration, scores) in results:
        if len(state) >= len(best_state):
            continue
        _limit = (np.nanmean(scores) + 
                  np.nanstd(scores) / np.sqrt(scores.shape[0]))
        if _limit >= best_score:
            states_1sd.append((state, iteration, scores))

    shortest_1sd = np.inf

    for (state, iteration, scores) in states_1sd:
        if len(state) < shortest_1sd:
            shortest_1sd = len(state)
            
    best_state_1sd = None
    best_score_1sd = -np.inf

    for (state, iteration, scores) in states_1sd:
        avg_score = np.nanmean(scores)
        if ((len(state) == shortest_1sd)
            and (avg_score <=
                 best_score_1sd)):
            best_state_1sd = state
            best_score_1sd = avg_score
            
    new_results = {}
    for (state, iteration, scores) in results:
        new_state = tuple([v.name for v in state])
        new_results[new_state] = np.nanmean(scores)
    if best_state_1sd:
        best_state_1sd = tuple([v.name for v in best_state_1sd])
        return best_state_1sd, new_results
    else:
        best_state = tuple([v.name for v in best_state])
        return best_state, new_results
