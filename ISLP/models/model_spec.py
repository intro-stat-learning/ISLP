"""
Model specification
===================

This module defines the basic object to represent regression
formula: *ModelSpec*. 
"""

from collections import namedtuple
from itertools import product
from typing import NamedTuple, Any
from copy import copy

import numpy as np, pandas as pd
from scipy import sparse as sp_sparse

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler)
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.exceptions import NotFittedError
from joblib import hash as joblib_hash

from .columns import (_get_column_info,
                      Column,
                      _categorical_from_df,
                      _check_categories)

from ..transforms import (Poly,
                          BSpline,
                          NaturalSpline,
                          Interaction)

DOCACHE = False

class Feature(NamedTuple):

    """
    An element in a model matrix that will build
    columns from an array-like X.
    """

    variables: tuple
    name: str
    encoder: Any
    use_transform: bool=True   # if False use the predict method
    pure_columns: bool=False
    override_encoder_colnames: bool=False
    

#### contrast specific code

def _as_series(X):
    """
    A single categorical feature as a `pd.Series`.
    """
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError(f'expecting a single column, got {X.shape[1]}')
        X = X.iloc[:,0]
    if not isinstance(X, pd.Series):
        X = np.asarray(X)
        if X.ndim > 2 or (X.ndim == 2 and X.shape[1] != 1):
            raise ValueError(f'expecting a single column, got array of shape {X.shape}')
        X = pd.Series(X.reshape(-1))
    return X

def _observed_levels(X):
    """
    Levels of `X` that are observed (missing values excluded),
    in the order of `X.dtype.categories` for a categorical,
    sorted otherwise.
    """
    observed = X.dropna().unique()
    if isinstance(X.dtype, pd.CategoricalDtype):
        observed = set(observed)
        return [c for c in X.dtype.categories if c in observed]
    return sorted(observed)

def _level_codes(X, levels, during='transform'):
    """
    Position of each entry of `X` in `levels`, and a mask
    of missing entries (whose codes are -1). Raises a `ValueError`
    for entries that are neither missing nor in `levels`.
    """
    X = _as_series(X)
    missing = np.asarray(X.isna())
    codes = pd.Index(levels).get_indexer(np.asarray(X, dtype=object))
    codes[missing] = -1
    unknown = (codes < 0) & ~missing
    if unknown.any():
        raise ValueError('Found unknown categories {0} during {1}'.format(
                         sorted(set(np.asarray(X)[unknown]), key=str), during))
    return codes, missing

class Contrast(TransformerMixin, BaseEstimator):

    def __init__(self,
                 method='drop',
                 drop_level=None):
        """
        Contrast encoding for categorical variables.

        Parameters
        ----------
        method : ['drop', 'sum', None, callable]
            If 'drop', then a column of the one-hot
            encoding will be dropped. If 'sum', then the sum of
            coefficients is constrained to sum to 1.
            If `None`, the full one-hot encoding is returned.
            Finally, if callable, then it should take the number of
            levels of the category as a single argument and return
            an appropriate contrast of the full one-hot encoding.

        drop_level : level identifier (optional)
            If not None, this level of the category
            will be dropped if `method` is 'drop' (or given
            coefficient minus the sum of the others if 'sum').
            Matched against the levels by value, then by `str`.

        Levels are those observed when fit, in the order of
        the categories for a categorical dtype and sorted
        otherwise; unused categories are dropped. Missing
        values give a row of `np.nan`, and a level not seen
        when fit raises a `ValueError` on transform.

        """

        self.method = method
        self.drop_level = drop_level
        
    def fit(self, X, y=None):

        """
        Construct contrast of categorical variable
        for use in building a design matrix.

        Parameters
        ----------
        X : array-like
            X on which model matrix will be evaluated.
            If a :py:class:`pd.DataFrame` or :py:class:`pd.Series`, variables that are of
            categorical dtype will be treated as categorical.

        Returns
        -------
        F : array-like
            Columns of design matrix implied by the
            categorical variable.

        """

        levels = _observed_levels(_as_series(X))
        column_names = [str(l) for l in levels]
        nlevel = len(levels)

        if self.method in ['drop', 'sum']:
            if self.drop_level is None: # these defaults consistent with R
                drop_idx = 0 if self.method == 'drop' else nlevel - 1
            else:
                drop_idx = _level_index(levels, self.drop_level)
            keep = np.ones(nlevel, bool)
            keep[drop_idx] = 0
            columns = [n for n, k in zip(column_names, keep) if k]
            contrast_matrix = np.identity(nlevel)[:,keep]
            if self.method == 'sum':
                contrast_matrix[drop_idx] = -1
        elif callable(self.method):
            contrast_matrix = self.method(nlevel)
            columns = ['C({})'.format(i) for i in range(contrast_matrix.shape[1])]
        elif self.method is None:
            contrast_matrix = np.identity(nlevel)
            columns = column_names
        else:
            raise ValueError('method must be one of ["drop", "sum", None] or a callable' +
                             'that returns a contrast matrix and column names given the number' +
                             ' of levels')

        self.levels_ = levels
        self.n_features_in_ = 1
        self.columns_ = columns
        self.contrast_matrix_ = contrast_matrix
        return self

    def transform(self, X):
        check_is_fitted(self)
        codes, missing = _level_codes(X, self.levels_)
        D = np.zeros((codes.shape[0], len(self.levels_)))
        D[np.nonzero(~missing)[0], codes[~missing]] = 1
        value = D.dot(self.contrast_matrix_)
        value[missing] = np.nan

        if isinstance(X, (pd.DataFrame, pd.Series)):
            df = pd.DataFrame(value, columns=self.columns_)
            df.index = X.index
            return df
        return value

def _level_index(levels, level):
    """
    Index of `level` in `levels`, matched by value then by `str`.
    """
    for match in [lambda l: l == level,
                  lambda l: str(l) == str(level)]:
        idx = [i for i, l in enumerate(levels) if match(l)]
        if idx:
            return idx[0]
    raise ValueError('level {0!r} is not among the observed levels {1}'.format(
                     level, [str(l) for l in levels]))

class Ordinal(TransformerMixin, BaseEstimator):

    def __init__(self, levels=None):
        """
        Ordinal encoding for categorical variables: each entry
        is replaced by the position of its level.

        Parameters
        ----------
        levels : sequence (optional)
            The levels in order. If None, the order must be
            unambiguous: the categories of an ordered categorical
            dtype (unused categories keep their positions), or
            the sorted values of a numeric or boolean column.
            Other columns, e.g. strings or an unordered categorical,
            raise a `ValueError` since their sorted (alphabetical)
            order is rarely the intended one.

        Missing values are encoded as `np.nan`, and a level not
        among the levels raises a `ValueError`.
        """
        self.levels = levels

    def fit(self, X, y=None):

        """
        Find the levels of the categorical variable.

        Parameters
        ----------
        X : array-like
            Single column on which the encoding will be evaluated.

        """

        X = _as_series(X)
        if self.levels is not None:
            levels = list(self.levels)
            if len(set(levels)) != len(levels):
                raise ValueError('levels should not contain duplicates')
            _level_codes(X, levels, during='fit')   # raises if X has other levels
        elif isinstance(X.dtype, pd.CategoricalDtype) and X.dtype.ordered:
            levels = list(X.dtype.categories)
        elif (pd.api.types.is_numeric_dtype(X.dtype) or
              pd.api.types.is_bool_dtype(X.dtype)):
            levels = _observed_levels(X)
        else:
            raise ValueError(
                'the order of the levels {0} is ambiguous: give it '
                'with `levels=[...]`, or use an ordered pd.Categorical, '
                'e.g. `X[col].cat.as_ordered()` or '
                '`pd.Categorical(..., categories=[...], ordered=True)`'.format(
                    [str(l) for l in _observed_levels(X)]))
        self.levels_ = levels
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        check_is_fitted(self)
        codes, missing = _level_codes(X, self.levels_)
        value = codes.astype(float).reshape((-1,1))
        value[missing] = np.nan
        return value

class ModelSpec(TransformerMixin, BaseEstimator):

    '''Parameters
    ----------

    terms : sequence (optional)

        Sequence of sets whose
        elements are columns of *X* when fit.
        For :py:class:`pd.DataFrame` these can be column
        names.

    intercept : bool (optional)

        Include a column for intercept?

    categorical_features : array-like of {bool, int} of shape (n_features) 
            or shape (n_categorical_features,), default=None.

        Indicates the categorical features. Will be ignored if *X* is a :py:class:`pd.DataFrame`
        or :py:class:`pd.Series`.

        - None : no feature will be considered categorical for :py:class:`np.ndarray`.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.

    categorical_encoders : dict (optional)

        Dictionary whose keys are columns of *X* that represent
        **categorical variables**. Its values are (unfitted) transforms
        used to encode those columns in the model matrix: each is
        cloned and fit when *fit* is called, so the values in the
        dictionary are not modified. The key 'categorical' sets the
        encoder for all other categorical columns, ordered or not,
        defaulting to `Contrast(method='drop')`. If the key 'ordinal'
        is present, its encoder is used instead for ordered
        `pd.Categorical` columns, e.g. `{'ordinal': Ordinal()}`
        to score them by the position of their level; a single
        column can be scored with `ordinal()` in the terms.

    '''

    def __init__(self,
                 terms=(),
                 intercept=True,
                 categorical_features=None,
                 categorical_encoders=None
                 ):

        self.intercept = intercept
        self.terms = terms
        self.categorical_features = categorical_features
        self.categorical_encoders = categorical_encoders

    def _check_X(self, X, reset):
        """
        Validate `X`, returning a `pd.DataFrame` or 2-d `np.ndarray`.

        If `reset`, record `n_features_in_` (and `feature_names_in_`
        for a `pd.DataFrame`), else check an `np.ndarray` has
        `n_features_in_` columns. Columns of a `pd.DataFrame` are
        matched by name, so it may have other columns, in any order.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame()
        if sp_sparse.issparse(X):
            raise TypeError('Sparse data is not supported by ModelSpec; '
                            'convert it with X.toarray()')
        if not isinstance(X, pd.DataFrame):
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError('Expected a 2-dimensional array or a pd.DataFrame, '
                                 f'got an array with {X.ndim} dimension(s). '
                                 'Reshape your data either using array.reshape(-1, 1) '
                                 'if your data has a single feature or array.reshape(1, -1) '
                                 'if it contains a single sample.')
            if np.iscomplexobj(X):
                raise ValueError('Complex data not supported')
        if X.shape[0] == 0:
            raise ValueError(f'Found array with 0 sample(s) (shape={X.shape}) '
                             'while a minimum of 1 is required.')
        if X.shape[1] == 0:
            raise ValueError(f'Found array with 0 feature(s) (shape={X.shape}) '
                             'while a minimum of 1 is required.')
        if reset:
            self.n_features_in_ = X.shape[1]
            if (isinstance(X, pd.DataFrame) and
                all(isinstance(c, str) for c in X.columns)):
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            elif hasattr(self, 'feature_names_in_'):
                del self.feature_names_in_
        elif (not isinstance(X, pd.DataFrame) and
              X.shape[1] != self.n_features_in_):
            raise ValueError(f'X has {X.shape[1]} features, but ModelSpec '
                             f'is expecting {self.n_features_in_} features as input.')
        return X

    # missing values in X propagate to the model matrix

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):   # scikit-learn < 1.6
        return {'allow_nan': True}

    def fit(self, X, y=None):

        """
        Fit the encoders of each term, determining
        the columns of the model matrix.

        Parameters
        ----------
        X : array-like
            X on which model matrix will be evaluated.
            If a :py:class:`pd.DataFrame` or :py:class:`pd.Series`, variables that are of
            categorical dtype will be treated as categorical.

        y : None
            Ignored. This parameter exists only for compatibility with
            :py:class:`sklearn.pipeline.Pipeline`.

        """

        X = self._check_X(X, reset=True)

        self.categorical_encoders_ = {'categorical': Contrast(method='drop')}
        if self.categorical_encoders is not None:
            self.categorical_encoders_.update(self.categorical_encoders)

        if isinstance(X, (pd.DataFrame, pd.Series)):
            (categorical_features,
             self.is_ordinal_) = _categorical_from_df(X)
            (self.is_categorical_,
             self.known_categories_) = _check_categories(categorical_features,
                                                         X)
            self.columns_ = X.columns
            if self.is_categorical_ is None:
                self.is_categorical_ = np.zeros(X.shape[1], bool)
            self.is_ordinal_ = pd.Series(self.is_ordinal_,
                                         index=self.columns_)
            self.is_categorical_ = pd.Series(self.is_categorical_,
                                             index=self.columns_)
        else:
            categorical_features = self.categorical_features
            (self.is_categorical_,
             self.known_categories_) = _check_categories(categorical_features,
                                                         X)
            if self.is_categorical_ is None:
                self.is_categorical_ = np.zeros(X.shape[1], bool)
            self.is_ordinal_ = np.zeros(self.is_categorical_.shape,
                                        bool)
            self.columns_ = np.arange(X.shape[1])

        self.features_ = {}
        self.encoders_ = {}

        # encoders are fit on clones so that `self.terms` is never
        # mutated and the spec can be refit (or cloned and refit)
        clone_memo = {}

        self.column_info_ = _get_column_info(X,
                                             self.columns_,
                                             np.asarray(self.is_categorical_),
                                             np.asarray(self.is_ordinal_),
                                             categorical_encoders=self.categorical_encoders_)

        # include each column as a Feature
        # so that their columns are built if needed

        for col_ in self.columns_:
            self.features_[col_] = Feature((col_,), str(col_), None, pure_columns=True) 

        # find possible interactions and other features

        tmp_cache = {}

        for term in self.terms:
            if isinstance(term, Feature):
                term_ = _clone_term(term, clone_memo)
                self.features_[term] = term_
                build_columns(self.column_info_,
                              X,
                              term_,
                              encoders=self.encoders_,
                              col_cache=tmp_cache,
                              fit=True) # these encoders won't have been fit yet
                for var, var_ in zip(term.variables, term_.variables):
                    if var not in self.features_ and isinstance(var, Feature):
                            self.features_[var] = var_
            elif term not in self.column_info_:
                # a tuple of features represents an interaction
                if type(term) == type((1,)): 
                    names = []
                    column_map = {}
                    column_names = {}
                    idx = 0
                    variables_ = []
                    for var in term:
                        if var in self.features_:
                            var_ = self.features_[var]
                        else:
                            var_ = _clone_term(var, clone_memo)
                        # column identifiers are kept as is, Features and
                        # Columns are replaced by their fitted copies
                        variables_.append(var_ if isinstance(var, (Feature, Column)) else var)
                        var = var_
                        cols, cur_names = build_columns(self.column_info_,
                                                        X,
                                                        var,
                                                        encoders=self.encoders_,
                                                        col_cache=tmp_cache,
                                                        fit=True) # these encoders won't have been fit yet
                        column_map[var.name] = range(idx, idx + cols.shape[1])
                        column_names[var.name] = cur_names
                        idx += cols.shape[1]                 
                        names.append(var.name)
                    encoder_ = Interaction(names, column_map, column_names)
                    self.features_[term] = Feature(tuple(variables_), ':'.join(n for n in names), encoder_)
                elif isinstance(term, Column):
                    term_ = _clone_term(term, clone_memo)
                    self.features_[term] = Feature((term_,), term.name, None, pure_columns=True)
                    build_columns(self.column_info_,
                                  X,
                                  term_,
                                  col_cache=tmp_cache,
                                  fit=True)
                else:
                    raise ValueError('each element in a term should be a Feature, Column or identify a column')
                
        # build the mapping of terms to columns and column names

        self.column_names_ = {}
        self.column_map_ = {}
        self.terms_ = [self.features_[t] for t in self.terms]
        
        idx = 0
        if self.intercept:
            self.column_map_['intercept'] = slice(0, 1)
            idx += 1 # intercept will be first column
        
        for term, term_ in zip(self.terms, self.terms_):
            term_df, term_names = build_columns(self.column_info_,
                                                X,
                                                term_,
                                                encoders=self.encoders_)
            self.column_names_[term] = term_names
            self.column_map_[term] = slice(idx, idx + term_df.shape[1])
            idx += term_df.shape[1]
    
        return self
    
    def transform(self, X, y=None):
        """
        Build design on X after fitting.

        Parameters
        ----------
        X : array-like

        y : None
            Ignored. This parameter exists only for compatibility with
            :py:class:`sklearn.pipeline.Pipeline`.
        """
        check_is_fitted(self)
        X = self._check_X(X, reset=False)
        return build_model(self.column_info_,
                           X,
                           self.terms_,
                           intercept=self.intercept,
                           encoders=self.encoders_)

    def get_feature_names_out(self, input_features=None):
        """
        Names of the columns of the model matrix.

        Parameters
        ----------
        input_features : None
            Ignored. This parameter exists only for compatibility with
            :py:class:`sklearn.pipeline.Pipeline`.

        Returns
        -------
        names : np.ndarray of str
        """
        check_is_fitted(self)
        names = ['intercept'] if self.intercept else []
        for term in self.terms:
            names.extend(str(n) for n in self.column_names_[term])
        if not names:   # build_model returns a 'zero' column
            names = ['zero']
        return np.asarray(names, dtype=object)

    # ModelSpec specific methods

    @property
    def names(self, help='Name for each term in model specification.'):
        names = []
        if self.intercept:
            names = ['intercept']
        return names + [t.name for t in self.terms_]
        

    def build_submodel(self,
                       X,
                       terms):
        """
        Build design on X after fitting.

        Parameters
        ----------
        X : array-like
            X on which columns are evaluated.

        terms : [Feature]
            Sequence of features. Elements of `self.terms`
            are replaced by their fitted versions in `self.terms_`.

        Returns
        -------
        D : array-like
            Design matrix created with `terms`
        """

        check_is_fitted(self)
        X = self._check_X(X, reset=False)
        terms = [self.features_.get(t, t) for t in terms]
        return build_model(self.column_info_,
                           X,
                           terms,
                           intercept=self.intercept,
                           encoders=self.encoders_)

    def build_sequence(self,
                       X,
                       anova_type='sequential'):
        """
        Build implied sequence of submodels 
        based on successively including more terms.

        Parameters
        ----------
        X : array-like
            X on which columns are evaluated.

        anova_type: str
            One of "sequential" or "drop".

        Returns
        -------

        models : generator
            Generator for sequence of models for ANOVA.

        """

        check_is_fitted(self)
        X = self._check_X(X, reset=False)

        col_cache = {}  # avoid recomputing the same columns

        dfs = []

        if self.intercept:
            df_int = pd.DataFrame({'intercept':np.ones(X.shape[0])})
            if isinstance(X, (pd.Series, pd.DataFrame)):
                df_int.index = X.index
            dfs.append(df_int)
        else:
            df_int = pd.DataFrame({'zero':np.zeros(X.shape[0])})
            if isinstance(X, (pd.Series, pd.DataFrame)):
                df_int.index = X.index
            dfs.append(df_int)

        for term_ in self.terms_:
            term_df, _  = build_columns(self.column_info_,
                                        X,
                                        term_,
                                        col_cache=col_cache,
                                        encoders=self.encoders_,
                                        fit=False)
            if isinstance(X, (pd.Series, pd.DataFrame)):
                term_df.index = X.index
            # as in build_model, make boolean columns float
            for i, col in enumerate(term_df.columns):
                if term_df.iloc[:,i].dtype == bool:
                    term_df[col] = term_df.iloc[:,i].astype(float)

            dfs.append(term_df)

        if anova_type == 'sequential':
            if isinstance(X, (pd.Series, pd.DataFrame)):
                return (pd.concat(dfs[:i], axis=1) for i in range(1, len(dfs)+1))
            else:
                return (np.column_stack(dfs[:i]) for i in range(1, len(dfs)+1))
        elif anova_type == 'drop':
            if isinstance(X, (pd.Series, pd.DataFrame)):
                return (pd.concat([dfs[j] for j in range(len(dfs)) if j != i], axis=1) for i in range(len(dfs)))
            else:
                return (np.column_stack([dfs[j] for j in range(len(dfs)) if j != i]) for i in range(len(dfs)))
        else:
            raise ValueError('anova_type must be one of ["sequential", "drop"]')

def _clone_term(term, memo):
    """
    Copy a Feature or Column, replacing each encoder
    (recursively) with an unfitted clone.

    `memo` maps `id()` of terms and encoders already copied
    to their copies, so a Feature or encoder that appears
    in several places is copied (and later fit) only once.
    """

    if not isinstance(term, (Feature, Column)):
        return term   # a column identifier
    if id(term) in memo:
        return memo[id(term)]

    encoder = term.encoder
    if encoder is not None:
        if id(encoder) not in memo:
            memo[id(encoder)] = clone(encoder, safe=False)
        encoder = memo[id(encoder)]

    if isinstance(term, Feature):
        variables = tuple(_clone_term(v, memo) for v in term.variables)
        term_ = term._replace(variables=variables, encoder=encoder)
    else:
        term_ = term._replace(encoder=encoder)
    memo[id(term)] = term_
    return term_

def fit_encoder(encoders, var, X):
    """
    Fit an encoder if not already registered
    in `encoders`.

    Parameters
    ----------

    encoders : dict
        Dictionary of encoders for each feature.

    var : Feature
        Feature whose encoder will be fit.

    X : array-like
        X on which encoder will be fit.

    """

    if var.encoder not in encoders:
        encoders[var] = var.encoder.fit(X)
            
def build_columns(column_info, X, var, encoders={}, col_cache={}, fit=False):
    """
    Build columns for a Feature from X.

    Parameters
    ----------

    column_info: dict
        Dictionary with values specifying sets of columns to
        be concatenated into a design matrix.

    X : array-like
        X on which columns are evaluated.

    var : Feature
        Feature whose columns will be built, typically a key in `column_info`.

    encoders : dict
        Dict that stores encoder of each Feature.
    
    col_cache: dict
        Dict where columns will be stored --
        if `var.name` in `col_cache` then just
        returns those columns.

    fit : bool (optional)
        If True, then try to fit encoder.
        Will raise an error if encoder has already been fit.

    """

    if var in column_info:
        var = column_info[var]

    if DOCACHE and joblib_hash([var, X]) in col_cache:
        return col_cache[joblib_hash([var, X])]

    if isinstance(var, Column):
        if DOCACHE:
            if joblib_hash([var, X]) not in col_cache:
                cols, names = var.get_columns(X, fit=fit)
                col_cache[joblib_hash([var, X])] = cols, names
            cols, name = col_cache[joblib_hash([var, X])]
        else:
            cols, names = var.get_columns(X, fit=fit)
    elif isinstance(var, Feature):
        cols = []
        names = []
        for v in var.variables:
            cur, cur_names = build_columns(column_info,
                                           X,
                                           v,
                                           encoders=encoders,
                                           col_cache=col_cache,
                                           fit=fit)
            cols.append(cur)
            names.extend(cur_names)
        cols = np.column_stack(cols)
        if len(names) != cols.shape[1]:
            names = ['{0}[{1}]'.format(var.name, j) for j in range(cols.shape[1])]
        if var.encoder:
            df_cols = pd.DataFrame(np.asarray(cols),
                                   columns=names)
            try:
                check_is_fitted(var.encoder)
                # ModelSpec.fit clones encoders, so one that is already fit
                # was kept by clone(), e.g. a sklearn.frozen.FrozenEstimator:
                # use it as is
                if fit and var not in encoders:
                    encoders[var] = var.encoder
            except NotFittedError as e:
                if fit:
                    fit_encoder(encoders,
                                var,
                                df_cols)
                # known issue with Pipeline
                # https://github.com/scikit-learn/scikit-learn/issues/18648
                elif isinstance(var.encoder, Pipeline):  
                    check_is_fitted(var.encoder, 'n_features_in_') 
                else:
                    raise(e)
            except Exception as e:  # was not the NotFitted
                raise ValueError(e)
            if var.use_transform:
                cols = var.encoder.transform(df_cols)
            else:
                cols = var.encoder.predict(df_cols)
            if hasattr(var.encoder, 'columns_') and not var.override_encoder_colnames:
                names = var.encoder.columns_
            else:
                if cols.ndim > 1 and cols.shape[1] > 1:
                    names = ['{0}[{1}]'.format(var.name, j) for j in range(cols.shape[1])]
                else:
                    names = [var.name]


    else:
        raise ValueError('expecting either a column or a Feature')
    val = pd.DataFrame(np.asarray(cols), columns=names)

    if isinstance(X, (pd.DataFrame, pd.Series)):
        val.index = X.index

    if DOCACHE:
        col_cache[joblib_hash([var.name, X])] = (val, names)
    return val, names

def build_model(column_info,
                X,
                terms,
                intercept=True,
                encoders={}):

    """
    Construct design matrix on a
    sequence of terms and X after 
    fitting.

    Parameters
    ----------
    column_info: dict
        Dictionary with values specifying sets of columns to
        be concatenated into a design matrix.

    X : array-like
        X on which columns are evaluated.

    terms : [Feature]
        Sequence of features

    encoders : dict
        Dict that stores encoder of each Feature.

    Returns
    -------
    df : np.ndarray or pd.DataFrame
        Design matrix.
    """

    dfs = []

    col_cache = {}  # avoid recomputing the same columns

    if intercept:
        df = pd.DataFrame({'intercept':np.ones(X.shape[0])})
        if isinstance(X, (pd.Series, pd.DataFrame)):
            df.index = X.index
        dfs.append(df)

    for term_ in terms:
        term_df = build_columns(column_info,
                                X,
                                term_,
                                col_cache=col_cache,
                                encoders=encoders,
                                fit=False)[0]
        dfs.append(term_df)

    if len(dfs):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            df = pd.concat(dfs, axis='columns')
            df.index = X.index
        else:
            return np.column_stack(dfs).astype(float)
    else:  # return a 0 design
        zero = np.zeros(X.shape[0])
        if isinstance(X, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame({'zero': zero})
            df.index = X.index
        else:
            return zero

    # if we reach here, we will be returning a DataFrame
    # make sure all columns are floats
    
    for i, col in enumerate(df.columns):
        if df.iloc[:,i].dtype == bool:
            df[col] = df.iloc[:,i].astype(float)
    return df

def derived_feature(variables, encoder=None, name=None, use_transform=True):
    """
    Create a Feature, optionally
    applying an encoder to the stacked columns.
    
    Parameters
    ----------

    variables : [column identifier, Column, Feature]
        Variables to apply transform to. Could be
        column identifiers or variables: all columns
        will be stacked before encoding.

    name : str (optional)
        Defaults to `str(encoder)`.

    encoder :  transform-like (optional)
        Transform obeying sklearn fit/transform convention.

    Returns
    -------

    var : Feature
    """

    if name is None:
        name = str(encoder)
    var = Feature(tuple([v for v in variables]),
                   name,
                   encoder,
                   use_transform=use_transform,
                   override_encoder_colnames=True)
    return var

def contrast(col,
             method='drop',
             drop_level=None):
    """
    Create encoding of categorical feature.
    
    Parameters
    ----------

    col: column identifier

    method: 'drop', 'sum', None or callable

    drop_level: level identifier

    Returns
    -------

    var : Feature

    """

    if isinstance(col, Column):
        name = col.name
        col = col.idx
    else:
        name = str(col)
    encoder = Contrast(method,
                       drop_level=drop_level)
    return Column(col,
                  name,
                  is_categorical=True,
                  encoder=encoder)

def ordinal(col, name=None, levels=None):
    """
    Create ordinal encoding of categorical feature: a single
    column holding the position of each entry's level (see `Ordinal`).

    Parameters
    ----------

    col: column identifier

    name: str (optional)
        Defaults to one derived from col.

    levels: sequence (optional)
        The levels in order. Needed unless `col` is an ordered
        categorical, or numeric.

    Returns
    -------

    var : Column

    """

    if isinstance(col, Column):
        if name is None:
            name = col.name
        col = col.idx
    elif name is None:
        name = str(col)
    return Column(col,
                  name,
                  is_categorical=True,
                  is_ordinal=True,
                  columns=(name,),
                  encoder=Ordinal(levels=levels))

def poly(col,
         degree=1,
         intercept=False,
         raw=False,
         name=None):

    """
    Create a polynomial Feature
    for a given column.
    
    Additional `args` and `kwargs`
    are passed to `Poly`.

    Parameters
    ----------

    col : column identifier or Column
        Column to transform.

    degree : int, default=1
        Degree of polynomial.

    intercept : bool, default=False
        Include a column for intercept?

    raw : bool, default=False
        If False, perform a QR decomposition on the resulting
        matrix of powers of centered and / or scaled features.

    name : str (optional)
        Defaults to one derived from col.

    Returns
    -------

    var : Feature
    """
    shortname, klass = 'poly', Poly
    encoder = klass(degree=degree,
                    raw=raw,
                    intercept=intercept) 
    if name is None:
        if isinstance(col, Column):
            name = col.name
        else:
            name = str(col)

        kwargs = {}
        if intercept:
            kwargs['intercept'] = True
        if raw:
            kwargs['raw'] = True

        _args = _argstring(degree=degree,
                           **kwargs)
        if _args:
            name = ', '.join([name, _args])

        name = f'{shortname}({name})'

    return derived_feature([col],
                            name=name,
                            encoder=encoder)

def ns(col, intercept=False, name=None, **spline_args):
    """
    Create a natural spline Feature
    for a given column.
    
    Additional *spline_args*
    are passed to :py:class:`NaturalSpline` along with *intercept*.

    Parameters
    ----------

    col : column identifier or Column

    intercept : bool
        Include an intercept column.

    name : str (optional)
        Defaults to one derived from col.

    Returns
    -------

    var : Feature

    """
    shortname, klass = 'ns', NaturalSpline
    if name is None:
        if isinstance(col, Column):
            name = col.name
        else:
            name = str(col)

        _args = _argstring(**spline_args)
        if _args:
            name = ', '.join([name, _args])

        name = f'{shortname}({name})'
    encoder = klass(intercept=intercept,
                    **spline_args) 
    return derived_feature([col],
                            name=name,
                            encoder=encoder)

def bs(col, intercept=False, name=None, **spline_args):
    """
    Create a B-spline Feature
    for a given column.
    
    Additional args and *spline_args*
    are passed to :py:class:`ISLP.transforms.BSpline`
    along with *intercept*.

    Parameters
    ----------

    col : column identifier or Column

    intercept : bool
        Include an intercept column.

    name : str (optional)
        Defaults to one derived from col.

    Returns
    -------

    var : Feature

    """
    shortname, klass = 'bs', BSpline
    if name is None:
        if isinstance(col, Column):
            name = col.name
        else:
            name = str(col)

        _args = _argstring(**spline_args)
        if _args:
            name = ', '.join([name, _args])

        name = f'{shortname}({name})'
    encoder = klass(intercept=intercept,
                    **spline_args) 
    return derived_feature([col],
                            name=name,
                            encoder=encoder)

def pca(variables, name, scale=False, **pca_args):
    """
    Create PCA encoding of features
    from a sequence of variables.
    
    Additional args and *pca_args*
    are passed to :py:class:`ISLP.transforms.PCA`.

    Parameters
    ----------

    variables : [column identifier, Column or Feature]
        Sequence whose columns will be encoded by PCA.

    Returns
    -------

    var : Feature

    """
    shortname, klass = 'pca', PCA
    encoder = klass(**pca_args) 
    if scale:
        scaler = StandardScaler(with_mean=True,
                                with_std=True)
        encoder = make_pipeline(scaler, encoder)

    _args = _argstring(**pca_args)

    if _args:
        name = ', '.join([name, _args])

    return derived_feature(variables,
                            name=f'{shortname}({name})',
                            encoder=encoder)


def _argstring(*args, **kwargs):
    _args = ', '.join([str(a) for a in args])
    _kwargs = ', '.join([f'{k}={v}' for k, v in kwargs.items()])

    if args and kwargs:
        return ', '.join([_args, _kwargs])
    elif args:
        return _args
    elif kwargs:
        return _kwargs
    else:
        return ''
