from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import time
from sklearn.base import BaseEstimator
from scipy.optimize import bisect

def compute_edf_pspline(N_mat, W_diag, Omega, lam):
    """
    Computes the Effective Degrees of Freedom (EDF) of a penalized B-spline.
    EDF = tr(S) = tr(N (N^T W N + lambda * Omega)^-1 N^T W)
    """
    XTWX = N_mat.T @ W_diag @ N_mat
    LHS = XTWX + lam * Omega
    LHS_inv = np.linalg.inv(LHS)
    
    # S = N @ LHS_inv @ N.T @ W
    # trace(S) = trace(LHS_inv @ N.T @ W @ N) = trace(LHS_inv @ XTWX)
    trace = np.trace(LHS_inv @ XTWX)
    return trace

def compute_edf_natural_spline(x, lam, w=None, knots=None, n_knots=None):
    """
    Computes the Effective Degrees of Freedom (EDF) of a penalized natural spline.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if w is None:
        w = np.ones(n)
    else:
        w = np.asarray(w, dtype=float)

    # Determine knots
    if knots is None:
        _n_knots = n_knots or max(10, n // 10)
        idx = np.linspace(0, n - 1, _n_knots, dtype=int)
        knots = np.unique(x[idx])
    else:
        knots = np.asarray(knots)
        knots.sort()
    n_k = len(knots)

    # Construct Design Matrix N (Cardinal Splines)
    cs_basis = CubicSpline(knots, np.eye(n_k), bc_type='natural')
    N_mat = cs_basis(x)

    # Apply linear extrapolation to the basis matrix for points outside the boundaries
    mask_lo = x < knots[0]
    mask_hi = x > knots[-1]
    
    if np.any(mask_lo) or np.any(mask_hi):
        cs_basis_d1 = cs_basis.derivative(nu=1)
        d1_left = cs_basis_d1(knots[0])
        d1_right = cs_basis_d1(knots[-1])
        
        vals_at_left_boundary = np.zeros(n_k); vals_at_left_boundary[0] = 1
        vals_at_right_boundary = np.zeros(n_k); vals_at_right_boundary[-1] = 1

        if np.any(mask_lo):
            x_lo = x[mask_lo]
            N_mat[mask_lo, :] = vals_at_left_boundary[None, :] + (x_lo - knots[0])[:, None] * d1_left[None, :]
        
        if np.any(mask_hi):
            x_hi = x[mask_hi]
            N_mat[mask_hi, :] = vals_at_right_boundary[None, :] + (x_hi - knots[-1])[:, None] * d1_right[None, :]

    # Construct Exact Penalty Matrix Omega
    hk = np.diff(knots)
    inv_hk = 1.0 / hk
    R_k = sparse.diags([hk[1:-1]/6.0, (hk[:-1]+hk[1:])/3.0, hk[1:-1]/6.0],
                       [-1, 0, 1], shape=(n_k-2, n_k-2))
    Q_k = sparse.diags([inv_hk[:-1], -inv_hk[:-1]-inv_hk[1:], inv_hk[1:]],
                       [0, -1, -2], shape=(n_k, n_k-2))

    if sparse.issparse(R_k):
        try:
            R_inv_QT = splinalg.spsolve(R_k.tocsc(), Q_k.T.tocsc())
        except RuntimeError: # singular
             R_inv_QT = (np.linalg.pinv(R_k.toarray()) @ Q_k.T).toarray()
    else:
        R_inv_QT = np.linalg.solve(R_k, Q_k.T)
    Omega = Q_k @ R_inv_QT

    # Weighted matrices
    W_diag = sparse.diags(w)

    return compute_edf_pspline(N_mat, W_diag, Omega, lam)

@dataclass
class SmoothingSpline(BaseEstimator):
    """
    Penalized natural spline estimator based on a cardinal basis representation.
    The coefficients `alpha_` represent the fitted values at the knots.
    This version uses a cardinal basis for fitting and manual linear
    extrapolation for prediction.
    """
    lamval: float = None
    df: int = None
    degrees_of_freedom: int = None
    knots: np.ndarray = None
    n_knots: int = None
    spline_: CubicSpline = field(init=False, repr=False)
    alpha_: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.degrees_of_freedom is not None:
            if self.df is not None:
                raise ValueError("Only one of `df` or `degrees_of_freedom` should be provided.")
            self.df = self.degrees_of_freedom
        
        if self.lamval is None and self.df is None:
            self.lamval = 0
        
        if self.lamval is not None and self.df is not None:
            raise ValueError("Only one of `lamval` or `df` can be provided.")

    def fit(self, x, y, w=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)

        if w is None:
            w = np.ones(n)
        else:
            w = np.asarray(w, dtype=float)

        # Determine knots
        knots = self.knots
        if knots is None:
            if self.n_knots is not None:
                percs = np.linspace(0, 100, self.n_knots)
                knots = np.percentile(x, percs)
            else:
                knots = np.sort(np.unique(x))
        else:
            knots = np.sort(knots)

        n_k = len(knots)
        self.knots = knots

        # Construct Design Matrix N (Cardinal Splines)
        cs_basis = CubicSpline(knots, np.eye(n_k), bc_type='natural')
        N_mat = cs_basis(x)

        # Apply linear extrapolation to the basis matrix for points outside the boundaries
        mask_lo = x < knots[0]
        mask_hi = x > knots[-1]
        
        if np.any(mask_lo) or np.any(mask_hi):
            cs_basis_d1 = cs_basis.derivative(nu=1)
            d1_left = cs_basis_d1(knots[0])
            d1_right = cs_basis_d1(knots[-1])
            
            vals_at_left_boundary = np.zeros(n_k); vals_at_left_boundary[0] = 1
            vals_at_right_boundary = np.zeros(n_k); vals_at_right_boundary[-1] = 1

            if np.any(mask_lo):
                x_lo = x[mask_lo]
                N_mat[mask_lo, :] = vals_at_left_boundary[None, :] + (x_lo - knots[0])[:, None] * d1_left[None, :]
            
            if np.any(mask_hi):
                x_hi = x[mask_hi]
                N_mat[mask_hi, :] = vals_at_right_boundary[None, :] + (x_hi - knots[-1])[:, None] * d1_right[None, :]

        # Construct Exact Penalty Matrix Omega
        hk = np.diff(knots)
        inv_hk = 1.0 / hk
        R_k = sparse.diags([hk[1:-1]/6.0, (hk[:-1]+hk[1:])/3.0, hk[1:-1]/6.0],
                           [-1, 0, 1], shape=(n_k-2, n_k-2))
        Q_k = sparse.diags([inv_hk[:-1], -inv_hk[:-1]-inv_hk[1:], inv_hk[1:]],
                           [0, -1, -2], shape=(n_k, n_k-2))

        if sparse.issparse(R_k):
            try:
                R_inv_QT = splinalg.spsolve(R_k.tocsc(), Q_k.T.tocsc())
            except RuntimeError: # singular
                 R_inv_QT = (np.linalg.pinv(R_k.toarray()) @ Q_k.T).toarray()
        else:
            R_inv_QT = np.linalg.solve(R_k, Q_k.T)
        Omega = Q_k @ R_inv_QT

        # Weighted matrices
        W_diag = sparse.diags(w)
        NTW = N_mat.T * w # Broadcasting

        if self.df is not None:
            self.lamval = self._df_to_lam(self.df, N_mat, W_diag, Omega)
            
        # Solve Ridge Regression
        LHS = NTW @ N_mat + self.lamval * Omega
        RHS = NTW @ y
        self.alpha_ = np.linalg.solve(LHS, RHS)
        
        # Store final spline
        self.spline_ = CubicSpline(self.knots, self.alpha_, bc_type='natural')
        return self

    def _df_to_lam(self, df, N_mat, W_diag, Omega, CUTOFF=1e12):
        def objective(lamval):
            return compute_edf_pspline(N_mat, W_diag, Omega, lamval) - df
        if objective(0) <= 0:
            return 0
        if objective(CUTOFF) >= 0:
            return CUTOFF
        
        val = bisect(objective, 0, CUTOFF)
        print(objective(val), 'huh')
        return val

    def predict(self, x):
        x = np.asarray(x)
        y_pred = np.zeros_like(x, dtype=float)
        
        mask_in = (x >= self.knots[0]) & (x <= self.knots[-1])
        mask_lo = x < self.knots[0]
        mask_hi = x > self.knots[-1]

        y_pred[mask_in] = self.spline_(x[mask_in])

        # Linear extrapolation for points outside the knots
        if np.any(mask_lo):
            deriv = self.spline_.derivative(1)(self.knots[0])
            y_pred[mask_lo] = self.alpha_[0] + (x[mask_lo] - self.knots[0]) * deriv
        
        if np.any(mask_hi):
            deriv = self.spline_.derivative(1)(self.knots[-1])
            y_pred[mask_hi] = self.alpha_[-1] + (x[mask_hi] - self.knots[-1]) * deriv

        return y_pred

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
