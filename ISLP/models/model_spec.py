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

        drop_level : str (optional)
            If not None, this level of the category
            will be dropped if `method=='drop'`.

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

        Xa = np.asarray(X).reshape((-1,1))
        self.encoder_ = OneHotEncoder(drop=None,
                                      sparse_output=False).fit(Xa)
        cats = self.encoder_.categories_[0]
        column_names = [str(n) for n in cats]

        if isinstance(X, pd.DataFrame): # expecting a column, we take .iloc[:,0]
            X = X.iloc[:,0]

        if X.dtype == 'category':
            Xcats = list(X.dtype.categories)
        else:
            Xcats = copy(column_names)
        if self.drop_level is None:
            if self.method == 'drop': # these defaults consistent with R
                drop_level = Xcats[0]
            elif self.method == 'sum':
                drop_level = Xcats[-1]
        else:
            drop_level = self.drop_level

        if self.method in ['drop', 'sum']:
            drop_idx = column_names.index(str(drop_level))
            Xcats.remove(drop_level)
            column_names.pop(drop_idx)

        colmap = [column_names.index(str(j)) for j in Xcats]
        cols = self.encoder_.transform(Xa)

        if self.method == 'drop':
            self.columns_ = [column_names[j] for j in colmap]
            self.contrast_matrix_ = np.identity(len(cats))
            keep = np.ones(len(cats), bool)
            keep[drop_idx] = 0
            self.contrast_matrix_ = self.contrast_matrix_[:,keep]
            self.contrast_matrix_ = self.contrast_matrix_[:,colmap]            
        elif self.method == 'sum':
            self.columns_ = [column_names[j] for j in colmap]
            self.contrast_matrix_ = np.zeros((len(cats), len(cats)-1))
            self.contrast_matrix_[:drop_idx][:,:drop_idx] = np.identity(drop_idx)
            self.contrast_matrix_[drop_idx+1:,drop_idx:] = np.identity(len(cats) - drop_idx - 1)
            self.contrast_matrix_[drop_idx] = -1
            self.contrast_matrix_ = self.contrast_matrix_[:,colmap]
        elif callable(self.method):
            self.contrast_matrix_ = self.method(len(cats))
            self.columns_ = ['C({})'.format(i) for i in range(self.contrast_matrix_.shape[1])]
        elif self.method is None:
            self.contrast_matrix_ = np.identity(len(cats))
            self.columns_ = column_names
        else:
            raise ValueError('method must be one of ["drop", "sum", None] or a callable' +
                             'that returns a contrast matrix and column names given the number' +
                             ' of levels')
        return self

    def transform(self, X):
        if not hasattr(self, 'encoder_'):
            self.fit(X)
        Xa = np.asarray(X).reshape((-1,1))
        D = self.encoder_.transform(Xa)
        value = D.dot(self.contrast_matrix_)

        if isinstance(X, (pd.DataFrame, pd.Series)):
            df = pd.DataFrame(value, columns=self.columns_)
            df.index = X.index
            return df
        return value

class ModelSpec(TransformerMixin, BaseEstimator):

    '''

    Parameters
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

    default_encoders : dict
        Dictionary whose keys are elements of *terms* and values
        are transforms to be applied to the associate columns in the model matrix
        by running the *fit_transform* method when *fit* is called and overwriting
        these values in the dictionary.
    '''

    def __init__(self,
                 terms=[],
                 intercept=True,
                 categorical_features=None,
                 default_encoders={'categorical': Contrast(method='drop'),
                                   'ordinal': OrdinalEncoder()}
                 ):
       
        self.intercept = intercept
        self.terms = terms
        self.categorical_features = categorical_features
        self.default_encoders = default_encoders
        
    def fit(self, X, y=None):

        """
        Construct parameters for orthogonal
        polynomials in the feature X.

        Parameters
        ----------
        X : array-like
            X on which model matrix will be evaluated.
            If a :py:class:`pd.DataFrame` or :py:class:`pd.Series`, variables that are of
            categorical dtype will be treated as categorical.

        """
        
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

        self.column_info_ = _get_column_info(X,
                                             self.columns_,
                                             self.is_categorical_,
                                             self.is_ordinal_,
                                             default_encoders=self.default_encoders)
        # include each column as a Feature
        # so that their columns are built if needed

        for col_ in self.columns_:
            self.features_[col_] = Feature((col_,), str(col_), None, pure_columns=True) 

        # find possible interactions and other features

        tmp_cache = {}

        for term in self.terms:
            if isinstance(term, Feature):
                self.features_[term] = term
                build_columns(self.column_info_,
                              X,
                              term,
                              encoders=self.encoders_,
                              col_cache=tmp_cache,
                              fit=True) # these encoders won't have been fit yet
                for var in term.variables:
                    if var not in self.features_ and isinstance(var, Feature):
                            self.features_[var] = var
            elif term not in self.column_info_:
                # a tuple of features represents an interaction
                if type(term) == type((1,)): 
                    names = []
                    column_map = {}
                    column_names = {}
                    idx = 0
                    for var in term:
                        if var in self.features_:
                            var = self.features_[var]
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
                    self.features_[term] = Feature(term, ':'.join(n for n in names), encoder_)
                elif isinstance(term, Column):
                    self.features_[term] = Feature((term,), term.name, None, pure_columns=True)
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
        return build_model(self.column_info_,
                           X,
                           self.terms_,
                           intercept=self.intercept,
                           encoders=self.encoders_)
    
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
            Sequence of features

        Returns
        -------
        D : array-like
            Design matrix created with `terms`
        """

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
                if fit and var not in encoders:
                    raise ValueError('encoder has already been fit previously')
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
    
    for i, _ in enumerate(df.columns):
        if df.iloc[:,i].dtype == bool:
            df.iloc[:,i] = df.iloc[:,i].astype(float)
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
        col = col.idx
        name = col.name
    else:
        name = str(col)
    encoder = Contrast(method,
                       drop_level=drop_level)
    return Column(col,
                  name,
                  is_categorical=True,
                  encoder=encoder)

def ordinal(col, name=None, *args, **kwargs):
    """
    Create ordinal encoding of categorical feature.
    
    Parameters
    ----------

    col: column identifier

    Returns
    -------

    var : Feature

    """

    shortname, klass = 'ordinal', OrdinalEncoder
    encoder = klass(*args,
                    **kwargs) 
    if name is None:
        if isinstance(col, Column):
            name = col.name
        else:
            name = str(col)

        _args = _argstring(*args, **kwargs)
        if _args:
            name = ', '.join([name, _args])

        name = f'{shortname}({name})'

    return derived_feature([col],
                            name=name,
                            encoder=encoder)

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
