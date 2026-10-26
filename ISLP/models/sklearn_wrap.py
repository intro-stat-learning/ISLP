"""

Wrappers for *statsmodels*
==========================

`sklearn_selected` and `sklearn_selection_path` also accept
a scikit-learn estimator, such as `glmnet.glm.GLM()`, in place of
a statsmodels model type.

"""

import numpy as np, pandas as pd

from sklearn.base import (BaseEstimator,
                          RegressorMixin,
                          TransformerMixin,
                          clone)
from sklearn.utils.validation import check_is_fitted

import statsmodels.api as sm
from .generic_selector import FeatureSelector


def _is_estimator(model_type):
    """
    Is `model_type` a scikit-learn estimator (an instance with
    `fit` and `get_params`, e.g. `GLM()`) rather than a statsmodels model type?
    """
    return (not isinstance(model_type, type) and
            hasattr(model_type, 'fit') and
            hasattr(model_type, 'get_params'))


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
        if _is_estimator(self.model_type):
            # the estimator's type if it sets one, else a regressor
            estimator_type = self.model_type.__sklearn_tags__().estimator_type
            tags.estimator_type = estimator_type or 'regressor'
        elif self.model_type == sm.OLS:
            tags.estimator_type = 'regressor'
        elif (issubclass(self.model_type, sm.GLM) and
              'family' in self.model_args and
              isinstance(self.model_args.get('family', None), sm.families.Binomial)):
            tags.estimator_type = 'classifier'
        return tags

    def fit(self, X, y):
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

        Returns
        -------

        self : sklearn_sm
            The fitted estimator.
        """

        if self.model_spec is not None:
            self.model_spec_ = self.model_spec.fit(X)
            X = self.model_spec_.transform(X)
        self.model_ = self.model_type(y, X, **self.model_args)
        self.results_ = self.model_.fit()
        return self

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
        a GLM this computes average deviance
        per observation. Note that, unlike the
        usual `sklearn` convention, lower is better.

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
            # `family.deviance` returns the total deviance
            if sample_weight is None:
                return self.model_.family.deviance(y,
                                                   yhat) / len(y)
            else:
                value = self.model_.family.deviance(y,
                                                    yhat,
                                                    freq_weights=sample_weight)

                return value / np.sum(sample_weight)
        else:
            raise NotImplementedError('score is only implemented for '
                                      'model_type sm.OLS or sm.GLM, '
                                      f'got {self.model_type}')

           
class sklearn_selected(sklearn_sm):

    """
    Parameters
    ----------

    model_type : class or estimator
        A model type from statsmodels, e.g. sm.OLS or sm.GLM,
        or a scikit-learn estimator, e.g. `glmnet.glm.GLM()`,
        which is cloned and fit with `fit(X, y)`.

    strategy : Strategy
        A search strategy

    model_args : dict (optional)
        Arguments passed to the statsmodels model; not used
        with an estimator, whose arguments are its parameters.

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
                                     
    def fit(self, X, y):
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

        self.sm_ = _selection_estimator(self.model_type,
                                        self.model_args)
        self.selector_ = FeatureSelector(self.sm_,
                                         self.strategy,
                                         cv=self.cv,
                                         scoring=self.scoring)
        self.selector_.fit(X, y)
        self.selected_state_ = self.selector_.selected_state_

        # now refit the model
        
        Xsel = self.selector_.fit_transform(X, y)
        self.model_, self.results_ = _refit(self.model_type,
                                            self.model_args,
                                            Xsel,
                                            y)
        return self

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
        return _predict(self.results_, Xsel)


class sklearn_selection_path(sklearn_sm):

    """
    Parameters
    ----------

    model_type : class or estimator
        A model type from statsmodels, e.g. sm.OLS or sm.GLM,
        or a scikit-learn estimator, e.g. `glmnet.glm.GLM()`,
        which is cloned and fit with `fit(X, y)`.

    strategy : Strategy
        A search strategy

    model_args : dict (optional)
        Arguments passed to the statsmodels model; not used
        with an estimator, whose arguments are its parameters.

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

    def fit(self, X, y):

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

        self.sm_ = _selection_estimator(self.model_type,
                                        self.model_args)
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
                model_, results_ = _refit(self.model_type,
                                          self.model_args,
                                          Xstate,
                                          y)
                self.models_.append((state, model_, results_))
        return self

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
            predictions.append(_predict(_results, Xstate))

        return np.array(predictions).T


def _selection_estimator(model_type, model_args):
    """
    The estimator used in the selection: a clone of an estimator,
    or `sklearn_sm` wrapping a statsmodels model type.
    """
    if _is_estimator(model_type):
        if model_args:
            raise ValueError('model_args is not used with an estimator: '
                             'set its parameters instead')
        return clone(model_type)
    return sklearn_sm(model_type,
                      model_args=model_args)


def _refit(model_type, model_args, X, y):
    """
    Fit the model on the selected columns X, returning
    `(model, results)`: both are the fitted estimator for an estimator.
    """
    if _is_estimator(model_type):
        estimator = clone(model_type).fit(X, y)
        return estimator, estimator
    model = model_type(y, X, **model_args)
    return model, model.fit()


def _predict(results, X):
    """
    Predictions of statsmodels results or of a fitted estimator.
    """
    if _is_estimator(results):
        return results.predict(X)
    return results.predict(exog=X)
