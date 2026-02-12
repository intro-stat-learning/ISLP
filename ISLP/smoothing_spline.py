from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import time
from sklearn.base import BaseEstimator
from scipy.optimize import bisect, brentq

def _prepare_natural_spline_matrices(x, weights=None, knots=None, n_knots=None):
    """
    Internal helper to compute the scaled matrices required for both
    fitting and EDF calculation of the Natural Spline Ridge.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if weights is None: weights = np.ones(n)
    else: weights = np.asarray(weights, dtype=float)

    if knots is None:
        if n_knots is not None:
            percs = np.linspace(0, 100, n_knots)
            knots = np.percentile(x, percs)
        else:
            knots = np.sort(np.unique(x))
    else:
        knots = np.asarray(knots)
        knots.sort()

    n_k = len(knots)

    # --- Standardization / Scaling ---
    x_min, x_max = x.min(), x.max()
    scale = x_max - x_min if x_max > x_min else 1.0

    x_scaled = (x - x_min) / scale
    knots_scaled = (knots - x_min) / scale

    # --- Design Matrix N ---
    cs_basis = CubicSpline(knots_scaled, np.eye(n_k), bc_type='natural')
    N_mat = cs_basis(x_scaled)

    # Apply linear extrapolation to the basis matrix for points outside the boundaries
    mask_lo = x_scaled < knots_scaled[0]
    mask_hi = x_scaled > knots_scaled[-1]
    
    if np.any(mask_lo) or np.any(mask_hi):
        cs_basis_d1 = cs_basis.derivative(nu=1)
        d1_left = cs_basis_d1(knots_scaled[0])
        d1_right = cs_basis_d1(knots_scaled[-1])
        
        vals_at_left_boundary = np.zeros(n_k); vals_at_left_boundary[0] = 1
        vals_at_right_boundary = np.zeros(n_k); vals_at_right_boundary[-1] = 1

        if np.any(mask_lo):
            x_lo = x_scaled[mask_lo]
            N_mat[mask_lo, :] = vals_at_left_boundary[None, :] + (x_lo - knots_scaled[0])[:, None] * d1_left[None, :]
        
        if np.any(mask_hi):
            x_hi = x_scaled[mask_hi]
            N_mat[mask_hi, :] = vals_at_right_boundary[None, :] + (x_hi - knots_scaled[-1])[:, None] * d1_right[None, :]

    # --- Penalty Matrix Omega ---
    hk = np.diff(knots_scaled)
    inv_hk = 1.0 / hk
    R_k = sparse.diags([hk[1:-1]/6.0, (hk[:-1]+hk[1:])/3.0, hk[1:-1]/6.0], [-1, 0, 1], shape=(n_k-2, n_k-2))
    Q_k = sparse.diags([inv_hk[:-1], -inv_hk[:-1]-inv_hk[1:], inv_hk[1:]], [0, -1, -2], shape=(n_k, n_k-2))

    if sparse.issparse(R_k):
        R_inv_QT = splinalg.spsolve(R_k.tocsc(), Q_k.T.tocsc()).toarray()
    else:
        R_inv_QT = np.linalg.solve(R_k, Q_k.T)

    Omega_scaled = Q_k @ R_inv_QT

    # NTW is N.T * weights (precomputed for efficiency)
    NTW = N_mat.T * weights

    return knots, N_mat, NTW, Omega_scaled, scale, n_k


def find_lamval_for_df(target_df, x, weights=None, knots=None, n_knots=None,
                       log10_lam_bounds=(-12, 12)):
    """
    Finds the exact lambda value that yields the target degrees of freedom.
    This is highly stable *because* of the internal [0,1] scaling.
    """
    _, N_mat, NTW, Omega_scaled, scale, n_k = _prepare_natural_spline_matrices(
        x, weights, knots, n_knots
    )

    # Validate target DF
    # Max DF is the number of knots (interpolation)
    # Min DF is 2 (linear fit, as lambda -> infinity)
    if target_df >= n_k - 0.01:
        raise ValueError(f"Target DF ({target_df}) too high. Max is roughly {n_k}.")
    if target_df <= 2.01:
        raise ValueError(f"Target DF ({target_df}) too low. Min is 2 (linear).")

    XTWX = NTW @ N_mat

    def df_error_func(log_lam_scaled):
        """Returns (Current DF - Target DF)"""
        lam_scaled = 10 ** log_lam_scaled
        LHS = XTWX + lam_scaled * Omega_scaled
        # Trace of (LHS^-1 * XTWX)
        S_matrix = np.linalg.solve(LHS, XTWX)
        current_df = np.trace(S_matrix)
        return current_df - target_df

    # Use Brent's method to find the root
    try:
        # We search in the scaled lambda space for maximum numerical stability
        log_lam_scaled_opt = brentq(df_error_func, log10_lam_bounds[0], log10_lam_bounds[1])
    except ValueError as e:
        raise RuntimeError(
            "Could not find root in the given bounds. This usually means "
            "the target DF is effectively unreachable or bounds need expanding."
        ) from e

    lam_scaled_opt = 10 ** log_lam_scaled_opt

    # Convert scaled lambda back to the original data scale
    lam_opt = lam_scaled_opt * (scale ** 3)

    return lam_opt

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
    x_min_: float = field(init=False, repr=False)
    x_scale_: float = field(init=False, repr=False)

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

        (self.knots,
         N_mat,
         NTW,
         Omega,
         self.x_scale_,
         n_k) = _prepare_natural_spline_matrices(x,
                                                 w,
                                                 knots=self.knots,
                                                 n_knots=self.n_knots)

        self.x_min_ = x.min()
        knots_scaled = (self.knots - self.x_min_) / self.x_scale_

        if self.df is not None:
            self.lamval = find_lamval_for_df(self.df,
                                             x,
                                             w,
                                             knots=self.knots,
                                             n_knots=self.n_knots)
            
        # Solve Ridge Regression
        lam_scaled = self.lamval / self.x_scale_**3

        LHS = NTW @ N_mat + lam_scaled * Omega
        RHS = NTW @ y
        self.alpha_ = np.linalg.solve(LHS, RHS)
        
        # Store final spline
        self.spline_ = CubicSpline(knots_scaled, self.alpha_, bc_type='natural')
        return self

    def predict(self, x):
        x_scaled = (x - self.x_min_) / self.x_scale_
        knots_scaled = (self.knots - self.x_min_) / self.x_scale_
        
        y_pred = np.zeros_like(x_scaled, dtype=float)
        
        mask_in = (x_scaled >= knots_scaled[0]) & (x_scaled <= knots_scaled[-1])
        mask_lo = x_scaled < knots_scaled[0]
        mask_hi = x_scaled > knots_scaled[-1]

        y_pred[mask_in] = self.spline_(x_scaled[mask_in])

        # Linear extrapolation for points outside the knots
        if np.any(mask_lo):
            deriv = self.spline_.derivative(1)(knots_scaled[0])
            y_pred[mask_lo] = self.alpha_[0] + (x_scaled[mask_lo] - knots_scaled[0]) * deriv
        
        if np.any(mask_hi):
            deriv = self.spline_.derivative(1)(knots_scaled[-1])
            y_pred[mask_hi] = self.alpha_[-1] + (x_scaled[mask_hi] - knots_scaled[-1]) * deriv

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
