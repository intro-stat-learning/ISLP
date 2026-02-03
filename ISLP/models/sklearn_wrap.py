"""

Wrappers for *statsmodels*
==========================

"""

import numpy as np, pandas as pd

from sklearn.base import (BaseEstimator,
                          RegressorMixin,
                          TransformerMixin)
from sklearn.utils.validation import check_is_fitted

import statsmodels.api as sm
from .generic_selector import FeatureSelector

class sklearn_sm(BaseEstimator,
                 RegressorMixin): 

    """
    Parameters
    ----------

    model_type: class
        A model type from statsmodels, e.g. sm.OLS or sm.GLM

    model_spec: ModelSpec
        Specify the design matrix.

    model_args: dict (optional)
        Arguments passed to the statsmodels model.

    Notes
    -----

    If model_str is present, then X and Y are presumed
    to be pandas objects that are placed
    into a dataframe before formula is evaluated.
    This affects `fit` and `predict` methods.

    """

    def __init__(self,
                 model_type,
                 model_spec=None,
                 model_args={}):

        self.model_type = model_type
        self.model_spec = model_spec
        self.model_args = model_args

    def __sklearn_tags__(self):    
        tags = super().__sklearn_tags__()
        if self.model_type == sm.OLS:
            tags.estimator_type = 'regressor'
        elif (issubclass(self.model_type, sm.GLM) and
              'family' in self.model_args and
              isinstance(self.model_args.get('family', None), sm.families.Binomial)):
            tags.estimator_type = 'classifier'
        return tags

    def fit(self, X, y, **fit_params):
        """
        Fit a statsmodel model
        with design matrix 
        determined from X and response y.

        Parameters
        ----------

        X : array-like
            Design matrix.

        y : array-like
            Response vector.
        """

        if self.model_spec is not None:
            self.model_spec_ = self.model_spec.fit(X)
            X = self.model_spec_.transform(X)
        self.model_ = self.model_type(y, X, **self.model_args)
        self.results_ = self.model_.fit()

    def predict(self, X):
        """
        Compute predictions
        for design matrix X.

        Parameters
        ----------

        X : array-like
            Design matrix.

        """
        if self.model_spec is not None:
            X = self.model_spec_.transform(X)
        return self.results_.predict(exog=X)
 
    def score(self, X, y, sample_weight=None):
        """
        Score a statsmodel model
        with test design matrix X and 
        test response y.

        If model_type is OLS, use MSE. For
        a GLM this computes (average) deviance.

        Parameters
        ----------

        X : array-like
            Design matrix.

        y : array-like
            Response vector.

        sample_weight : None
            Optional sample weights.
        """

        yhat = self.predict(X)
        if isinstance(self.model_, sm.OLS):
            if sample_weight is None:
                return np.mean((y-yhat)**2)
            else:
                return (np.mean((y-yhat)**2*sample_weight) /
                        np.mean(sample_weight))
                
        elif isinstance(self.model_, sm.GLM):
            if sample_weight is None:
                return self.model_.family.deviance(y,
                                                   yhat).mean()
            else:
                value = self.model_.family.deviance(y,
                                                    yhat,
                                                    freq_weights=sample_weight).mean()

                return value / np.mean(sample_weight)

           
class sklearn_selected(sklearn_sm):

    """
    Parameters
    ----------

    model_type : class
        A model type from statsmodels, e.g. sm.OLS or sm.GLM

    strategy : Strategy
        A search strategy

    model_args : dict (optional)
        Arguments passed to the statsmodels model.

    scoring : str or callable, default=None

        A str (see model evaluation documentation) or a scorer
        callable object / function with signature `scorer(estimator, X,
        y)` which should return only a single value.
        
    cv: int, cross-validation generator or an iterable, default=None

        Determines the cross-validation splitting strategy.

    """

    def __init__(self,
                 model_type,
                 strategy,
                 model_args={},
                 scoring=None,
                 cv=None):

        self.model_type = model_type
        self.model_args = model_args

        self.strategy = strategy
        self.cv = cv
        self.scoring = scoring

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
                                     
    def fit(self, X, y, **fit_params):
        """
        First, select a model
        with design matrix 
        determined from X and response y.
        Then, fit selected model.

        Parameters
        ----------

        X : array-like
            Design matrix.

        y : array-like
            Response vector.
        """

        # first run the selection process

        self.sm_ = sklearn_sm(self.model_type,
                              model_args=self.model_args)
        self.selector_ = FeatureSelector(self.sm_,
                                         self.strategy,
                                         cv=self.cv,
                                         scoring=self.scoring)
        self.selector_.fit(X, y)
        self.selected_state_ = self.selector_.selected_state_

        # now refit the model
        
        Xsel = self.selector_.fit_transform(X, y)
        self.model_ = self.model_type(y, Xsel, **self.model_args)
        self.results_ = self.model_.fit()

    def predict(self, X):
        """
        Compute predictions
        for design matrix X in selected model.

        Parameters
        ----------

        X : array-like
            Design matrix.

        """
        Xsel = self.selector_.transform(X)
        return self.results_.predict(exog=Xsel)


class sklearn_selection_path(sklearn_sm):

    """
    Parameters
    ----------

    model_type : class
        A model type from statsmodels, e.g. sm.OLS or sm.GLM

    strategy : Strategy
        A search strategy

    model_args : dict (optional)
        Arguments passed to the statsmodels model.

    scoring : str or callable, default=None

        A str (see model evaluation documentation) or a scorer
        callable object / function with signature `scorer(estimator, X,
        y)` which should return only a single value.
        
    cv: int, cross-validation generator or an iterable, default=None

        Determines the cross-validation splitting strategy.

    """

    def __init__(self,
                 model_type,
                 strategy,
                 model_args={},
                 scoring=None,
                 cv=None):
        self.model_type = model_type
        self.model_args = model_args

        self.strategy = strategy
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y, **fit_params):

        """
        First, select a model
        with design matrix 
        determined from X and response y.
        Then, fit selected model.

        Parameters
        ----------

        X : array-like
            Design matrix.

        y : array-like
            Response vector.
        """

        # first run the selection process

        self.sm_ = sklearn_sm(self.model_type,
                              model_args=self.model_args)
        self.selector_ = FeatureSelector(self.sm_,
                                         self.strategy,
                                         cv=self.cv,
                                         scoring=self.scoring)
        self.selector_.fit(X, y)
        build_submodel = self.selector_.strategy.build_submodel
        
        self.models_ = []

        for (state, _, _) in self.selector_.path_:
            if state is not None:  # last state could be (None,)*3
                Xstate = build_submodel(X, state)
                model_ = self.model_type(y, Xstate, **self.model_args)
                results_ = model_.fit()
                self.models_.append((state, model_, results_))

    def predict(self, X):
        """
        Compute predictions along selection path
        for design matrix X.

        Parameters
        ----------

        X : array-like
            Design matrix.

        """

        build_submodel = self.selector_.strategy.build_submodel
        predictions = []
        for (state, _, _results) in self.models_:
            Xstate = build_submodel(X, state)
            predictions.append(_results.predict(exog=Xstate))

        return np.array(predictions).T


