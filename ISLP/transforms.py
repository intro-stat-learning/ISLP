"""
Transformers
============

This module defines some sklearn transformers useful for defining
flexible regression models for single features, as well as interactions
between sets of columns.

    - Poly: for orthogonalized polynomial regression

    - Interaction: computing pairwise product for interactions

    - BSpline: arbitrary degree B-splines

    - NaturalSpline: natural cubic splines
"""

from itertools import product
import numpy as np, pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from scipy.interpolate import splev
from scipy import sparse as sp_sparse

class Poly(TransformerMixin, BaseEstimator):

    '''

    Parameters
    ----------

    degree : int, default=1
        Degree of polynomial.

    intercept : bool, default=False
        Include a column for intercept?

    raw : bool, default=False
        If False, perform a QR decomposition on the resulting
        matrix of powers of centered and / or scaled features.
    '''

    def __init__(self,
                 degree=1,
                 intercept=False,
                 raw=False):
     
        self.degree = degree
        self.raw = raw
        self.intercept = intercept

    def fit(self,
            X,
            y=None):

        """
        Construct parameters for orthogonal
        polynomials in the feature X.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Features used in fitting `svm`. Assumed to have at least 2 columns.

        y : default=None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        """
        
        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        self.mean_ = X.mean()
        
        if not self.raw:
            powX = np.power.outer(X - self.mean_, np.arange(0, self.degree+1))
            # Following R's poly construction
            Q, R = np.linalg.qr(powX)
            Z = Q * np.diag(R)[None,:]
            self.norm2_ = (Z**2).sum(0)
            self.alpha_ = ((X[:,None] * Z**2).sum(0) / self.norm2_)[:self.degree]
            self.norm2_ = np.hstack([1, self.norm2_])

        # for pandas 

        self.columns_ = range(self.degree+1)
        if not self.intercept:
            self.columns_ = self.columns_[:-1]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            if isinstance(X_orig, pd.Series):
                name = X_orig.name
            else:
                name = X_orig.columns[0]
            self.columns_ = ['{0}[{1}]'.format(self, d)
                             for d in self.columns_]

        return self
    
    def transform(self, X):

        """
        Construct parameters for orthogonal
        polynomials in the feature X.

        Parameters
        ----------
        X : array-like
            X on which features will be evaluated.

        Returns
        -------
        XP : np.ndarray
            Evaluated polynomial features.
        """
        check_is_fitted(self)

        X_orig = X
        
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        if not self.raw:
            Z = np.ones((n, self.degree+1))
            Z[:,1] = X - self.alpha_[0]

            if self.degree > 1:
                for i in range(1, self.degree):
                    Z[:,i+1] = ((X - self.alpha_[i]) * Z[:,i] -
                                self.norm2_[i+1] / self.norm2_[i] * Z[:,i-1])
            Z /= np.sqrt(self.norm2_[1:])
            powX = Z
        else:
            powX = np.power.outer(X, np.arange(0, self.degree+1))

        if not self.intercept:
            powX = powX[:,1:]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame(powX,
                              columns=self.columns_,
                              copy=False)
            df.index = X_orig.index
            return df
        else:
            return powX

#### Interaction transform

class Interaction(TransformerMixin, BaseEstimator):

    '''

    Form the tensor product interaction
    from a group of columns.

    Parameters
    ----------

    variables : sequence
        Variables in the interactions.

    columns : dict
        Mapping from variable names to columns.

    column_names : dict
        Mapping from variable names to lists of column names.

    '''

    def __init__(self,
                 variables,
                 columns,
                 column_names):
       
        self.variables = variables
        self.columns = columns
        self.column_names = column_names
        self.column_names_ = {} 
        self.columns_ = []

        variable_names = []
        for variable in self.variables:
            cols = self.columns[variable]
            col_names = ['{0}[{1}]'.format(variable, i) for i in range(len(self.columns[variable]))]
            if variable in column_names:
                col_names = [str(c) for c in column_names[variable]]
            if len(cols) > 1:
                variable_names.append(col_names)
            else:
                variable_names.append(['{0}'.format(variable)])

        for names in product(*variable_names):
            self.columns_.append(':'.join(names))

    def fit(self,
            X,
            y=None):

        """
        Nothing to be computed for the fit.

        Parameters
        ----------
        X : array-like

        y : default=None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        """
        
        return self
    
    def transform(self, X):

        """
        Construct columns representing interactions
        of relevant variables.

        Parameters
        ----------
        X : array-like
            X on which features will be evaluated.

        Returns
        -------
        XI : np.ndarray
            Evaluated interaction features.
        """
        check_is_fitted(self)

        if sp_sparse.issparse(X):   # products of sparse columns stay sparse
            X = sp_sparse.csc_matrix(X)
            R = None
            for variable in self.variables:
                V = X[:, list(self.columns[variable])]
                if R is None:
                    R = V
                    continue
                # column i*q + j of the product is R[:,i] * V[:,j], as for
                # the dense columns below: scale the rows of R by each column
                # of V, then reorder
                r, q = R.shape[1], V.shape[1]
                M = sp_sparse.hstack([sp_sparse.diags(V[:, [j]].toarray().ravel()) @ R
                                      for j in range(q)], format='csc')
                R = M[:, (np.arange(q)[None, :] * r + np.arange(r)[:, None]).ravel()]
            return sp_sparse.csr_matrix(R)

        X_orig = X
        X = np.asarray(X)

        # each column of the product is filled in place
        idx = list(product(*[self.columns[v] for v in self.variables]))
        value = np.empty((X.shape[0], len(idx)))
        for k, cols in enumerate(idx):
            value[:, k] = X[:, cols[0]]
            for j in cols[1:]:
                value[:, k] *= X[:, j]

        df = pd.DataFrame(value,
                          columns=self.columns_,
                          copy=False)
        if isinstance(X_orig, (pd.DataFrame, pd.Series)):
            df.index = X_orig.index

        return df

#### Spline specific code

def _onehot(p, j):
    v = np.zeros(p)
    v[j] = 1
    return v

def _splevf(x,
            tk,
            der=0,
            ext=0):
    """
    Full B-spline matrix at x
    """
    x = np.asarray(x)
    knots, degree = tk
    nbasis = len(knots) - (degree + 1)
    tcks = [(knots, _onehot(nbasis, j), degree) for
            j in range(nbasis)]
    return np.column_stack([splev(x, tck, ext=ext, der=der)
                            for tck in tcks])

def _splev_taylor(x,
                  basept,
                  tk,
                  order):
    """
    Taylor expansion of B-spline at x around basept
    """
    x = np.asarray(x)
    derivs = np.array([_splevf([basept],
                               tk,
                               der=o,
                               ext=0).reshape(-1)
                       for o in range(order+1)])
    polys = np.power.outer(x-basept,
                           np.arange(order+1))
    fact = np.concatenate([[1], np.cumprod(np.arange(1, order+1))])
    polys /= fact[None,:]

    return polys.dot(derivs)

class BSpline(TransformerMixin, BaseEstimator):

    '''

    Parameters
    ----------

    degree : int, default=3
        Degree of polynomial.

    intercept : bool, default=False
        If False, a column of basis is dropped so that by
        adding an intercept column design stays full rank.

    lower_bound : float, default=None
        Lower boundary not. Will be set to minimal value if not supplied.

    upper_bound : float, default=None
        Upper boundary not. Will be set to maximal value if not supplied.

    internal_knots : array-like (optional)
        Optional internal knots of B-spline. Will be set to
        appropriate quantiles based on `df`.

    df : int, default=None
        Degrees of freedom for spline. Defaults to `degree + intercept`.

    ext : int
        How B-splines are to be extended beyond the boundary using
        `scipy.interpolate.splev`.

    '''

    def __init__(self,
                 degree=3,
                 intercept=False,
                 lower_bound=None,
                 upper_bound=None,
                 internal_knots=None,
                 df=None,
                 ext=0):

        self.degree = degree
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.internal_knots = internal_knots
        self.df = df
        self.ext = ext
        self.intercept = intercept
        
    def fit(self,
            X,
            y=None):

        """
        Compute knots for B-spline representation.

        Parameters
        ----------
        X : array-like
            Single feature on which B-spline will be evaluated.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        """
        
        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        order = self.degree + 1

        lower_bound = self.lower_bound
        if lower_bound is None:
            lower_bound = X.min()

        upper_bound = self.upper_bound
        if upper_bound is None:
            upper_bound = X.max()

        if self.df is not None:
            if self.df < order - 1 + self.intercept:
                raise ValueError('df must be greater than or equal to %d' % (order - 1 + self.intercept))
            ninternal = self.df - (order - 1 + self.intercept)
            percs = 100*np.linspace(0, 1, ninternal+2)[1:-1]
            internal_knots = np.percentile(X, percs)
            if self.internal_knots is not None:
                raise ValueError('only one of df or internal_knots should be specified')
        else:
            internal_knots = np.asarray(sorted(self.internal_knots))
            if self.internal_knots is None:
                raise ValueError('if df not specified then need internal_knots')

        if lower_bound >= upper_bound:
            raise ValueError('lower_bound must be smaller than upper_bound')

        self.internal_knots_ = internal_knots

        self.knots_ = np.sort(np.concatenate([[lower_bound]*order,
                                              [upper_bound]*order,
                                              internal_knots]))
        if self.knots_[0] < lower_bound:
            raise ValueError('internal_knots should be greater than our equal to lower_bound')
        if self.knots_[-1] > upper_bound:
            raise ValueError('internal_knots should be less than our equal to upper_bound')

        self.boundary_knots_ = [lower_bound, upper_bound]

        # for pandas 

        self.nbasis_ = len(self.knots_) - (self.degree + 1)
        self.columns_ = range(self.nbasis_)

        if not self.intercept:
            self.columns_ = self.columns_[:-1]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            if isinstance(X_orig, pd.Series):
                name = X_orig.name
            else:
                name = X_orig.columns[0] # a pd.DataFrame
            self.columns_ = ['{0}[{1}]'.format(self, d)
                             for d in self.columns_]
        return self
    
    def transform(self, X):

        """
        Construct design for B-splines
        using features X.

        Parameters
        ----------
        X : array-like
            X on which splines will be evaluated.

        Returns
        -------
        XS : np.ndarray
            Evaluated splines.
        """
        check_is_fitted(self)

        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        value = _splevf(X,
                        (self.knots_, self.degree),
                        der=0,
                        ext=self.ext)
        if not self.intercept:
            value = value[:,1:]
        columns_ = self.columns_

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame(value,
                              columns=columns_,
                              copy=False)
            df.index = X_orig.index
            return df
        else:
            return value

class NaturalSpline(TransformerMixin, BaseEstimator):

    '''

    Natural cubic spline.

    Parameters
    ----------

    intercept : bool, default=False
        If False, a column of basis is dropped so that by
        adding an intercept column design stays full rank.

    lower_bound : float, default=None
        Lower boundary not. Will be set to minimal value if not supplied.

    upper_bound : float, default=None
        Upper boundary not. Will be set to maximal value if not supplied.

    internal_knots : array-like (optional)
        Optional internal knots of B-spline. Will be set to
        appropriate quantiles based on `df`.

    df : int, default=None
        Degrees of freedom for spline. Defaults to `3 + intercept`.

    ext : int, default=0
        How B-splines are to be extended beyond the boundary using
        `scipy.interpolate.splev`.

    '''

    def __init__(self,
                 intercept=False,
                 lower_bound=None,
                 upper_bound=None,
                 internal_knots=None,
                 df=None,
                 ext=0):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.internal_knots = internal_knots
        self.df = df
        self.intercept = intercept
        self.ext = ext
        
    def fit(self,
            X,
            y=None):

        """
        Compute knots for natural spline representation.

        Parameters
        ----------
        X : array-like

            Single feature on which B-spline will be evaluated.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        """
        
        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        order = 4

        lower_bound = self.lower_bound
        if lower_bound is None:
            lower_bound = X.min()

        upper_bound = self.upper_bound
        if upper_bound is None:
            upper_bound = X.max()
            
        if self.df is not None:
            if self.df < order - 1 + self.intercept - 2:
                raise ValueError('df must be greater than or equal to %d' % (order - 1 + self.intercept - 2))
            ninternal = self.df - (order - 1 + self.intercept - 2) # -2 for constraints
            percs = 100*np.linspace(0, 1, ninternal+2)[1:-1]
            internal_knots = np.percentile(X, percs)
            if self.internal_knots is not None:
                raise ValueError('only one of df or internal_knots should be specified')
        else:
            internal_knots = np.asarray(sorted(self.internal_knots))
            if self.internal_knots is None:
                raise ValueError('if df not specified then need internal_knots')

        if lower_bound >= upper_bound:
            raise ValueError('lower_bound must be smaller than upper_bound')
        
        self.internal_knots_ = internal_knots
        self.knots_ = np.sort(np.concatenate([[lower_bound]*order,
                                              [upper_bound]*order,
                                              internal_knots]))
        if self.knots_[0] < lower_bound:
            raise ValueError('internal_knots should be greater than our equal to lower_bound')
        if self.knots_[-1] > upper_bound:
            raise ValueError('internal_knots should be less than our equal to upper_bound')
        
        self.boundary_knots_ = [lower_bound, upper_bound]
        self.nbasis_ = self.knots_.shape[0] - order

        # now enforce constraint that 2nd derivative at two boundary knots is 0
        # this depends on the X that is passed to fit!

        boundary_vals = _splevf(self.boundary_knots_,
                                (self.knots_, 3),
                                der=2,
                                ext=0)
        if not self.intercept:
            boundary_vals = boundary_vals[:,1:]
        Q_, R_ = np.linalg.qr(boundary_vals.T,
                              mode='complete')
        self.Qproj_ = Q_[:,2:]

        # for pandas 

        self.columns_ = range(self.nbasis_-2) # two constraints

        if not self.intercept:
            self.columns_ = self.columns_[:-1]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            if isinstance(X_orig, pd.Series):
                name = X_orig.name
            else:
                name = X_orig.columns[0] # a pd.DataFrame
            self.columns_ = ['{0}[{1}]'.format(self, d)
                             for d in self.columns_]
        return self
    
    def transform(self, X):

        """
        Construct design for natural cubic splines
        for features X.

        Parameters
        ----------
        X : array-like
            X on which natural cubic splines will be evaluated.

        Returns
        -------
        XN : np.ndarray
            Evaluated natural cubic spline features.
        """

        check_is_fitted(self)

        lower_bound, upper_bound = self.boundary_knots_

        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        value = np.empty((X.shape[0], self.nbasis_))
        inside = (X >= lower_bound) * (X <= upper_bound)
        if np.any(inside):
            value[inside] = _splevf(X[inside],
                                    (self.knots_, 3),
                                    der=0,
                                    ext=self.ext)

        to_the_right = (X > upper_bound)
        if np.any(to_the_right):
            value[to_the_right] = _splev_taylor(X[to_the_right],
                                                upper_bound,
                                                (self.knots_, 3),
                                                order=1)

        to_the_left = (X < lower_bound)
        if np.any(to_the_left):
            value[to_the_left] = _splev_taylor(X[to_the_left],
                                                lower_bound,
                                                (self.knots_, 3),
                                                order=1)

        # possibly drop column for intercept

        if not self.intercept:
            value = value[:,1:]

        # map onto the natural splines

        value = value.dot(self.Qproj_)
        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame(value,
                              columns=self.columns_,
                              copy=False)
            df.index = X_orig.index
            return df
        else:
            return value

