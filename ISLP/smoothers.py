from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import time
from sklearn.base import BaseEstimator
from scipy.optimize import bisect, brentq

@dataclass
class SplineFitter:
    """
    Internal class to handle the fitting logic for the smoothing spline.
    Parameters
    ----------
    x : np.ndarray
        Predictor variable.
    w : np.ndarray, optional
        Weights for the observations. Defaults to None.
    lamval : float, optional
        The penalty parameter. It is generally recommended to specify the
        degrees of freedom `df` instead of `lamval`. Defaults to None.
    df : int, optional
        The desired degrees of freedom. This is used to automatically
        determine the penalty parameter `lam`. Defaults to None.
    knots : np.ndarray, optional
        The interior knots to use for the spline. Defaults to None.
    n_knots : int, optional
        The number of knots to use. If `knots` is not specified, `n_knots`
        are chosen uniformly from the percentiles of `x`. Defaults to None.
    Attributes
    ----------
    knots_ : np.ndarray
        The knots used for the spline.
    n_k_ : int
        The number of knots.
    x_min_ : float
        The minimum value of `x`.
    x_scale_ : float
        The scaling factor for `x`.
    knots_scaled_ : np.ndarray
        The scaled knots.
    Omega_ : np.ndarray
        The penalty matrix.
    N_ : np.ndarray
        The basis matrix.
    NTW_ : np.ndarray
        The weighted basis matrix transposed.
    alpha_ : np.ndarray
        The fitted spline coefficients.
    spline_ : CubicSpline
        The fitted cubic spline object.
    intercept_ : float
        The intercept of the linear component of the fitted spline.
    coef_ : float
        The coefficient of the linear component of the fitted spline.
    """
    x: np.ndarray
    w: np.ndarray = None
    lamval: float = None
    df: int = None
    knots: np.ndarray = None
    n_knots: int = None
    
    def _prepare_matrices(self):
        """
        Compute the scaled matrices required for both
        fitting and EDF calculation of the Natural Spline Ridge.
        """
        x = self.x
        weights = self.w
        knots = self.knots
        n_knots = self.n_knots
        
        n = len(x)
        if weights is None: weights = np.ones(n)
        
        if knots is None:
            if n_knots is not None:
                percs = np.linspace(0, 100, n_knots)
                knots = np.percentile(x, percs)
            else:
                knots = np.sort(np.unique(x))
        else:
            knots = np.asarray(knots)
            knots.sort()
            
        self.knots = knots
        n_k = len(knots)
        self.n_k_ = n_k

        # --- Standardization / Scaling ---
        x_min, x_max = x.min(), x.max()
        scale = x_max - x_min if x_max > x_min else 1.0
        self.x_min_ = x_min
        self.x_scale_ = scale

        x_scaled = (x - x_min) / scale
        knots_scaled = (knots - x_min) / scale
        self.knots_scaled_ = knots_scaled

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

        self.Omega_ = Q_k @ R_inv_QT
        self.N_ = N_mat
        self.NTW_ = self.N_.T * weights

    def _find_lamval_for_df(self, target_df, log10_lam_bounds=(-12, 12)):
        """
        Finds the exact lambda value that yields the target degrees of freedom.
        Parameters
        ----------
        target_df : float
            The target degrees of freedom.
        log10_lam_bounds : tuple, optional
            The bounds for the search of the log10 of the penalty parameter.
            Defaults to (-12, 12).
        Returns
        -------
        float
            The penalty parameter `lam` that corresponds to the target `df`.
        """
        # Validate target DF
        if target_df >= self.n_k_ - 0.01:
            raise ValueError(f"Target DF ({target_df}) too high. Max is roughly {self.n_k_}.")
        if target_df <= 2.01:
            raise ValueError(f"Target DF ({target_df}) too low. Min is 2 (linear).")

        XTWX = self.NTW_ @ self.N_

        def df_error_func(log_lam_scaled):
            lam_scaled = 10 ** log_lam_scaled
            LHS = XTWX + lam_scaled * self.Omega_
            S_matrix = np.linalg.solve(LHS, XTWX)
            current_df = np.trace(S_matrix)
            return current_df - target_df

        try:
            log_lam_scaled_opt = brentq(df_error_func, log10_lam_bounds[0], log10_lam_bounds[1])
        except ValueError as e:
            raise RuntimeError(
                "Could not find root in the given bounds. This usually means "
                "the target DF is effectively unreachable or bounds need expanding."
            ) from e

        lam_scaled_opt = 10 ** log_lam_scaled_opt
        return lam_scaled_opt * (self.x_scale_ ** 3)

    def __post_init__(self):
        self._prepare_matrices()
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)

    def fit(self, y):
        """
        Fit the smoothing spline. This method prepares the matrices,
        finds the penalty parameter if `df` is specified, and solves
        for the spline coefficients.
        Parameters
        ----------
        y : np.ndarray
            The response variable.
        """
        self.y = y
        # Solve Ridge Regression
        lam_scaled = self.lamval / self.x_scale_**3

        LHS = self.NTW_ @ self.N_ + lam_scaled * self.Omega_
        RHS = self.NTW_ @ self.y
        self.alpha_ = np.linalg.solve(LHS, RHS)
        
        # Store final spline
        self.spline_ = CubicSpline(self.knots_scaled_, self.alpha_, bc_type='natural')

        # Fit linear component to the predictions
        y_hat = self.predict(self.x)
        if self.w is not None:
            X = np.vander(self.x, 2)
            Xw = X * self.w[:, None]
            yw = y_hat * self.w
            beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        else:
            beta = np.polyfit(self.x, y_hat, 1)

        self.intercept_ = beta[1]
        self.coef_ = beta[0]
        
    def update_weights(self, w):
        """
        Update the weights and refit the model.
        Parameters
        ----------
        w : np.ndarray
            New weights for the observations.
        """
        self.w = w
        self._prepare_matrices()
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)
        if hasattr(self, 'y'):
            self.fit(self.y)

    def predict(self, x):
        """
        Predict the response for a new set of predictor variables.
        Parameters
        ----------
        x : np.ndarray
            The predictor variables.
        Returns
        -------
        np.ndarray
            The predicted response.
        """
        x_scaled = (x - self.x_min_) / self.x_scale_
        
        y_pred = np.zeros_like(x_scaled, dtype=float)
        
        mask_in = (x_scaled >= self.knots_scaled_[0]) & (x_scaled <= self.knots_scaled_[-1])
        mask_lo = x_scaled < self.knots_scaled_[0]
        mask_hi = x_scaled > self.knots_scaled_[-1]

        y_pred[mask_in] = self.spline_(x_scaled[mask_in])

        # Linear extrapolation for points outside the knots
        if np.any(mask_lo):
            deriv = self.spline_.derivative(1)(self.knots_scaled_[0])
            y_pred[mask_lo] = self.alpha_[0] + (x_scaled[mask_lo] - self.knots_scaled_[0]) * deriv
        
        if np.any(mask_hi):
            deriv = self.spline_.derivative(1)(self.knots_scaled_[-1])
            y_pred[mask_hi] = self.alpha_[-1] + (x_scaled[mask_hi] - self.knots_scaled_[-1]) * deriv

        return y_pred

    @property
    def nonlinear_(self):
        """
        The non-linear component of the fitted spline.
        """
        linear_part = self.coef_ * self.x + self.intercept_
        return self.predict(self.x) - linear_part


@dataclass
class SmoothingSpline(BaseEstimator):
    """
    Penalized natural spline estimator.
    This estimator fits a smoothing spline to the data, a popular method for
    non-parametric regression. The smoothness of the spline is controlled by
    a penalty parameter, which can be chosen automatically by specifying the
    desired degrees of freedom.
    Parameters
    ----------
    lamval : float, optional
        The penalty parameter `lambda`. It is generally recommended to specify
        the degrees of freedom `df` instead of `lamval`. If neither is
        provided, `lamval` defaults to 0, which corresponds to interpolation.
    df : int, optional
        The desired degrees of freedom. This is used to automatically
        determine the penalty parameter `lamval`.
    degrees_of_freedom : int, optional
        An alias for `df`.
    knots : np.ndarray, optional
        The interior knots to use for the spline. If not specified, the unique
        values of `x` are used.
    n_knots : int, optional
        The number of knots to use. If `knots` is not specified, `n_knots`
        are chosen uniformly from the percentiles of `x`.
    Attributes
    ----------
    fitter_ : SplineFitter
        The internal fitter object that contains the detailed results of the
        fit, including the fitted spline, coefficients, and linear components.
    """
    lamval: float = None
    df: int = None
    degrees_of_freedom: int = None
    knots: np.ndarray = None
    n_knots: int = None
    fitter_: SplineFitter = field(init=False, repr=False)

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
        """
        Fit the smoothing spline to the data.
        Parameters
        ----------
        x : np.ndarray
            The predictor variable.
        y : np.ndarray
            The response variable.
        w : np.ndarray, optional
            Weights for the observations. Defaults to None.
        Returns
        -------
        self : SmoothingSpline
            The fitted estimator.
        """
        self.fitter_ = SplineFitter(x,
                                     w=w,
                                     lamval=self.lamval,
                                     df=self.df,
                                     knots=self.knots,
                                     n_knots=self.n_knots)
        self.fitter_.fit(y)
        return self

    def predict(self, x):
        """
        Predict the response for a new set of predictor variables.
        Parameters
        ----------
        x : np.ndarray
            The predictor variables.
        Returns
        -------
        np.ndarray
            The predicted response.
        """
        return self.fitter_.predict(x)

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
