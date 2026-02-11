from dataclasses import dataclass, field
from scipy.interpolate import (make_smoothing_spline,
                               BSpline)
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import time
from sklearn.base import BaseEstimator
from scipy.optimize import bisect

@dataclass
class SmoothingSpline(BaseEstimator):
    """
    Smoothing spline estimator.

    Fits a smoothing spline to data, allowing for either a smoothing
    parameter `lamval` or degrees of freedom `df` to be specified.
    """
    lamval: float = None
    df: int = None
    degrees_of_freedom: int = None
    _penalized_spline_engine: "PenalizedSpline" = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.degrees_of_freedom is not None:
            if self.df is not None:
                raise ValueError("Only one of `df` or `degrees_of_freedom` should be provided.")
            self.df = self.degrees_of_freedom
        
        if self.lamval is None and self.df is None:
            raise ValueError("Either `lamval` or `df` must be provided.")
        if self.lamval is not None and self.df is not None:
            raise ValueError("Only one of `lamval` or `df` can be provided.")

    def _preprocess(self, x, y, w=None):
        """
        Sort and unique x values, and average y and sum w at those unique x's.
        """
        x_unique, inverse = np.unique(x, return_inverse=True)
        y_unique = np.bincount(inverse, weights=y) / np.bincount(inverse)
        if w is not None:
            w_unique = np.bincount(inverse, weights=w)
        else:
            w_unique = np.bincount(inverse)
        return x_unique, y_unique, w_unique

    def _df_to_lam_reinsch(self, df, x, w, CUTOFF=1e12):
        """
        Find lamval for a given df using the exact Reinsch formulation.
        """
        def objective(lam):
            return compute_edf_reinsch(x, lam, w) - df

        if objective(0) <= 0:
            return 0
        if objective(CUTOFF) >= 0:
            return CUTOFF

        return bisect(objective, 0, CUTOFF)

    def fit(self, x, y, w=None):
        """
        Fit the smoothing spline.
        """
        x_unique, y_unique, w_unique = self._preprocess(x, y, w)

        if self.df is not None:
            self.lamval = self._df_to_lam_reinsch(self.df, x_unique, w_unique)
        
        # Use PenalizedSpline as the engine
        if (self._penalized_spline_engine is None or
            self.lamval != self._penalized_spline_engine.lam or
            not np.array_equal(x_unique, self._penalized_spline_engine.knots)):
            
            self._penalized_spline_engine = PenalizedSpline(lam=self.lamval, knots=x_unique)
            self._penalized_spline_engine.fit(x, y, w=w) # Full fit
        else:
            # Engine exists and is compatible, just refit y
            self._penalized_spline_engine._solve_alpha_and_set_spline(y, w=w)

        self.spline_ = self._penalized_spline_engine.spline_
        return self

    def predict(self, x):
        """
        Predict using the fitted smoothing spline.
        """
        return self.spline_(x)

@dataclass
class PenalizedSpline(BaseEstimator):
    """
    Penalized B-spline estimator.
    """
    lam: float = 0
    knots: np.ndarray = None
    n_knots: int = None
    spline_: BSpline = field(init=False, repr=False)
    _N_mat: sparse.csc_matrix = field(init=False, repr=False)
    _t: np.ndarray = field(init=False, repr=False)
    _Omega: np.ndarray = field(init=False, repr=False)
    _XTWX_cached: np.ndarray = field(init=False, repr=False)
    _W_diag_cached: sparse.spmatrix = field(init=False, repr=False)

    def fit(self, x, y, w=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)

        if w is None:
            w = np.ones(n)
        else:
            w = np.asarray(w, dtype=float)

        # Determine knots (cached)
        if self.knots is None:
            n_knots = self.n_knots or max(10, n // 10)
            idx = np.linspace(0, n - 1, n_knots, dtype=int)
            knots = np.unique(x[idx])
        else:
            knots = np.asarray(self.knots)
            knots.sort()
        n_k = len(knots)

        # B-spline basis (cached)
        k = 3
        self._t = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))
        self._N_mat = BSpline.design_matrix(x, self._t, k)
        B_knots = BSpline.design_matrix(knots, self._t, k)

        # Penalty matrix Omega (cached)
        hk = np.diff(knots)
        inv_hk = 1.0 / hk
        R_k = sparse.diags([hk[1:-1]/6.0, (hk[:-1]+hk[1:])/3.0, hk[1:-1]/6.0],
                           [-1, 0, 1], shape=(n_k-2, n_k-2))
        Q_k = sparse.diags([inv_hk[:-1], -inv_hk[:-1]-inv_hk[1:], inv_hk[1:]],
                           [0, -1, -2], shape=(n_k, n_k-2))

        if sparse.issparse(R_k):
            R_inv_QT = splinalg.spsolve(R_k.tocsc(), Q_k.T.tocsc()).toarray()
        else:
            R_inv_QT = np.linalg.solve(R_k, Q_k.T)
        K_reinsch = Q_k @ R_inv_QT
        self._Omega = B_knots.T @ K_reinsch @ B_knots

        # Weighted design matrix product (cached)
        self._W_diag_cached = sparse.diags(w) # cache the diagonal matrix, not just vector
        self._XTWX_cached = self._N_mat.T @ self._W_diag_cached @ self._N_mat

        self._solve_alpha_and_set_spline(y, w)
        return self

    def _solve_alpha_and_set_spline(self, y, w=None):
        # Solve Ridge Regression
        # (N.T W N + lam Omega) alpha = N.T W y

        LHS = self._XTWX_cached + self.lam * self._Omega
        LHS += 1e-8 * np.eye(LHS.shape[0]) # Small regularization
        
        if w is None:
            RHS = self._N_mat.T @ (self._W_diag_cached @ y)
        else:
            w = np.asarray(w, dtype=float)
            RHS = self._N_mat.T @ (sparse.diags(w) @ y)

        if sparse.issparse(LHS):
            LHS = LHS.toarray()
        
        alpha = np.linalg.solve(LHS, RHS)
        self.spline_ = BSpline(self._t, alpha, 3)

    def predict(self, x):
        return self.spline_(x)

def compute_edf_reinsch(x, lamval, weights=None):
    x = np.array(x, dtype=float)
    n = len(x)
    if weights is None:
        w_inv_vec = np.ones(n)
    else:
        weights = np.array(weights, dtype=float)
        w_inv_vec = 1.0 / (weights + 1e-12)
    W_inv = sparse.diags(w_inv_vec)
    h = np.diff(x)
    inv_h = 1.0 / h
    main_diag_R = (h[:-1] + h[1:]) / 3.0
    off_diag_R = h[1:-1] / 6.0
    R = sparse.diags([off_diag_R, main_diag_R, off_diag_R],
                     [-1, 0, 1], shape=(n-2, n-2))
    d0 = inv_h[:-1]
    d1 = -inv_h[:-1] - inv_h[1:]
    d2 = inv_h[1:]
    Q = sparse.diags([d0, d1, d2], [0, -1, -2], shape=(n, n-2))
    B = Q.T @ W_inv @ Q
    M_trace = R + lamval * B
    M_trace = M_trace.tocsc()
    B = B.tocsc()
    X = splinalg.spsolve(M_trace, B)
    if sparse.issparse(X):
        tr_val = X.diagonal().sum()
    else:
        tr_val = np.trace(X)
    return n - lamval * tr_val

def estimate_edf_hutchinson(x, lamval, weights=None, n_vectors=50):
    x = np.array(x, dtype=float)
    n = len(x)
    if weights is None: w_inv_vec = np.ones(n)
    else: w_inv_vec = 1.0 / (np.array(weights, dtype=float) + 1e-12)
    W_inv = sparse.diags(w_inv_vec)
    h = np.diff(x)
    inv_h = 1.0 / h
    R = sparse.diags([h[1:-1]/6.0, (h[:-1]+h[1:])/3.0, h[1:-1]/6.0], [-1, 0, 1], shape=(n-2, n-2))
    Q = sparse.diags([inv_h[:-1], -inv_h[:-1]-inv_hk[1:], inv_hk[1:]], [0, -1, -2], shape=(n, n-2))
    B = Q.T @ W_inv @ Q
    M_trace = R + lamval * B
    M_trace = M_trace.tocsc()
    solve_M = splinalg.factorized(M_trace)
    trace_est = 0.0
    dim = n - 2
    for i in range(n_vectors):
        z = np.random.choice([-1, 1], size=dim)
        v = B @ z
        u = solve_M(v)
        trace_est += np.dot(z, u)
    mean_trace = trace_est / n_vectors
    return n - lamval * mean_trace
