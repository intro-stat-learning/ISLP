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
    spline_: BSpline = field(init=False, repr=False)

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

        # The objective function is monotonically decreasing in lam.
        # If df is high (low smoothing), lam is small.
        # If df is low (high smoothing), lam is large.

        # Check boundaries
        if objective(0) <= 0:  # Target df is >= n, lam=0 is the best we can do
            return 0
        if objective(CUTOFF) >= 0: # Target df is very small, lam=CUTOFF is best
            return CUTOFF

        # Find the root using bisection
        return bisect(objective, 0, CUTOFF)

    def fit(self, x, y, w=None):
        """
        Fit the smoothing spline.

        Parameters
        ----------
        x : array-like
            Predictor variable.
        y : array-like
            Response variable.
        w : array-like, optional
            Weights for each observation.
        """
        x_unique, y_unique, w_unique = self._preprocess(x, y, w)

        if self.df is not None:
            self.lamval = self._df_to_lam_reinsch(self.df, x_unique, w_unique)

        self.spline_ = make_smoothing_spline(x_unique, y_unique, w=w_unique, lam=self.lamval)
        return self

    def predict(self, x):
        """
        Predict using the fitted smoothing spline.

        Parameters
        ----------
        x : array-like
            Values at which to evaluate the spline.
        """
        return self.spline_(x)



def compute_edf_reinsch(x, lamval, weights=None):
    """
    Computes the Effective Degrees of Freedom (EDF) of a cubic smoothing spline
    using the Reinsch (dual) formulation and sparse matrices.

    Formula: EDF = n - lambda * tr( (R + lambda*B)^-1 * B )

    This method is exact and efficient for N up to ~20,000.
    """
    x = np.array(x, dtype=float)
    n = len(x)

    # 1. Setup Weights
    if weights is None:
        w_inv_vec = np.ones(n)
    else:
        weights = np.array(weights, dtype=float)
        w_inv_vec = 1.0 / (weights + 1e-12)

    W_inv = sparse.diags(w_inv_vec)

    # 2. Compute differences h
    h = np.diff(x)
    inv_h = 1.0 / h

    # 3. Construct Sparse R (n-2 x n-2) - Tridiagonal
    main_diag_R = (h[:-1] + h[1:]) / 3.0
    off_diag_R = h[1:-1] / 6.0

    R = sparse.diags([off_diag_R, main_diag_R, off_diag_R],
                     [-1, 0, 1], shape=(n-2, n-2))

    # 4. Construct Sparse Q (n x n-2)
    d0 = inv_h[:-1]
    d1 = -inv_h[:-1] - inv_h[1:]
    d2 = inv_h[1:]

    Q = sparse.diags([d0, d1, d2], [0, -1, -2], shape=(n, n-2))

    # 5. Form the Matrices B and M
    # B = Q^T W^-1 Q
    B = Q.T @ W_inv @ Q

    # CORRECT TRACE FORMULA uses M_trace = R + lambda * B
    # (Note: This is different from the fitting system B + lambda * R)
    M_trace = R + lamval * B

    # 6. Compute Trace
    # tr(S) = n - lambda * tr( M_trace^-1 * B )

    # Convert to CSC for efficient solving
    M_trace = M_trace.tocsc()
    B = B.tocsc()

    # Solve M * X = B
    X = splinalg.spsolve(M_trace, B)

    # Compute trace of X
    if sparse.issparse(X):
        tr_val = X.diagonal().sum()
    else:
        tr_val = np.trace(X)

    return n - lamval * tr_val

def estimate_edf_hutchinson(x, lamval, weights=None, n_vectors=50):
    """
    Estimates the EDF using Hutchinson's Trace Estimator.
    Uses the correct matrix formulation M = R + lambda * B.
    """
    x = np.array(x, dtype=float)
    n = len(x)

    if weights is None: w_inv_vec = np.ones(n)
    else: w_inv_vec = 1.0 / (np.array(weights, dtype=float) + 1e-12)
    W_inv = sparse.diags(w_inv_vec)

    h = np.diff(x)
    inv_h = 1.0 / h

    R = sparse.diags([h[1:-1]/6.0, (h[:-1]+h[1:])/3.0, h[1:-1]/6.0], [-1, 0, 1], shape=(n-2, n-2))
    Q = sparse.diags([inv_h[:-1], -inv_h[:-1]-inv_h[1:], inv_h[1:]], [0, -1, -2], shape=(n, n-2))

    B = Q.T @ W_inv @ Q

    # Correct Matrix for Trace: M = R + lambda * B
    M_trace = R + lamval * B
    M_trace = M_trace.tocsc()

    solve_M = splinalg.factorized(M_trace)

    # tr(S) = n - lambda * tr(M^-1 B)
    trace_est = 0.0
    dim = n - 2

    for i in range(n_vectors):
        z = np.random.choice([-1, 1], size=dim)

        # v = B z
        v = B @ z

        # u = M^-1 v = M^-1 B z
        u = solve_M(v)

        trace_est += np.dot(z, u)

    mean_trace = trace_est / n_vectors
    return n - lamval * mean_trace

# --- Verification & Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # --- 1. Small Explicit Verification ---
    print("\n[Verification vs Dense Matrix Definition]")

    x_small = np.array([0.0, 0.5, 1.2, 1.8, 2.5, 3.0, 3.8, 4.2, 5.0,
                        5.5, 6.2, 7.0, 7.5, 8.2, 9.0])
    weights_small = np.array([1.0, 1.2, 0.8, 1.0, 1.5, 0.5, 1.0, 1.0,
                              2.0, 1.0, 0.9, 1.1, 1.0, 0.8, 1.0])
    y_small = np.sin(x_small) # Dummy y for Scipy check
    lam_small = 0.5

    # A. Optimized Sparse Method
    edf_reinsch = compute_edf_reinsch(x_small, lam_small, weights_small)

    # B. Dense Matrix Construction (S = (W + lam*K)^-1 W)
    n = len(x_small)
    h = np.diff(x_small)

    # Manual R
    R_dense = np.zeros((n-2, n-2))
    for j in range(n-2):
        R_dense[j, j] = (h[j] + h[j+1]) / 3.0
        if j < n-3:
            R_dense[j, j+1] = h[j+1] / 6.0
            R_dense[j+1, j] = h[j+1] / 6.0

    # Manual Q
    Q_dense = np.zeros((n, n-2))
    for j in range(n-2):
        Q_dense[j, j] = 1.0/h[j]
        Q_dense[j+1, j] = -1.0/h[j] - 1.0/h[j+1]
        Q_dense[j+2, j] = 1.0/h[j+1]

    R_inv = np.linalg.inv(R_dense)
    K_dense = Q_dense @ R_inv @ Q_dense.T

    W_diag = np.diag(weights_small)
    LHS = W_diag + lam_small * K_dense
    S_matrix = np.linalg.solve(LHS, W_diag)
    edf_dense = np.trace(S_matrix)

    print(f"Reinsch (Sparse) EDF: {edf_reinsch:.10f}")
    print(f"Explicit (Dense) EDF: {edf_dense:.10f}")
    diff = abs(edf_reinsch - edf_dense)
    print(f"Difference:           {diff:.2e}")
    if diff < 1e-9:
        print(">> MATCH SUCCESSFUL")
    else:
        print(">> MATCH FAILED")

    # C. Scipy Cross-Validation
    # We fit Scipy's spline and check if fitted values match S_matrix @ y
    print("\n[Cross-Check with Scipy make_smoothing_spline]")
    try:
        # Note: Scipy minimizes sum w(y-f)^2 + lam * int(f'')^2
        spl = make_smoothing_spline(x_small, y_small, w=weights_small, lam=lam_small)
        y_scipy = spl(x_small)

        # Our Dense fitted values
        y_our_matrix = S_matrix @ y_small

        max_diff = np.max(np.abs(y_scipy - y_our_matrix))
        print(f"Max Diff (Scipy Fit vs S @ y): {max_diff:.2e}")
        if max_diff < 1e-9:
            print(">> MATRIX VALIDATED AGAINST SCIPY")
        else:
            print(">> SCIPY MISMATCH (Check lambda scaling definitions)")

    except ImportError:
        print("Scipy 1.10+ required for make_smoothing_spline")

    # --- 2. Performance Demo ---
    print("\n[Performance Test]")
    N_demo = 2000
    x_demo = np.sort(np.random.rand(N_demo) * 10)
    w_demo = np.random.uniform(0.5, 1.5, N_demo)

    t0 = time.time()
    edf_val = compute_edf_reinsch(x_demo, 0.1, w_demo)
    print(f"N={N_demo}: EDF={edf_val:.4f} (Time: {time.time()-t0:.4f}s)")

    # Hutchinson
    t0 = time.time()
    edf_est = estimate_edf_hutchinson(x_demo, 0.1, w_demo, n_vectors=50)
    print(f"Hutchinson Est: {edf_est:.4f} (Time: {time.time()-t0:.4f}s)")
    
