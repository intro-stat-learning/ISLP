import numpy as np
import pytest
from ISLP.smoothing_spline import SmoothingSpline, compute_edf_reinsch
from scipy.interpolate import make_smoothing_spline

# Setup for R comparison
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    R_ENABLED = True
except ImportError:
    R_ENABLED = False

def test_smoothing_spline_lamval():
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    spline = SmoothingSpline(lamval=0.1)
    spline.fit(x, y)
    y_pred = spline.predict(x)
    
    assert y_pred.shape == x.shape

def test_smoothing_spline_df():
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1
    
    spline = SmoothingSpline(df=5)
    spline.fit(x, y)
    y_pred = spline.predict(x)
    
    assert y_pred.shape == x.shape

@pytest.mark.parametrize(
    "use_weights, use_df",
    [(False, True), (True, True), (False, False), (True, False)]
)
def test_comparison_with_R(use_weights, use_df):
    """
    Compare the output of SmoothingSpline with R's smooth.spline,
    parameterized for weights and df/lambda.
    """
    # Ensure Rpy2 is enabled for this test
    if not R_ENABLED:
        pytest.skip("R or rpy2 is not installed.")

    # Generate some data
    np.random.seed(10)
    x = np.linspace(0, 1, 100)
    x = np.sort(np.append(x, x[5])) # introduce a duplicate
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.3, x.shape[0])
    weights = np.random.uniform(0.5, 1.5, size=x.shape) if use_weights else None

    # ISLP fitting
    if use_df:
        islp_spline = SmoothingSpline(degrees_of_freedom=8)
    else:
        islp_spline = SmoothingSpline(lamval=0.0001) # A small lambda for testing

    islp_spline.fit(x, y, w=weights)
    x_unique, y_unique, w_unique = islp_spline._preprocess(x, y, w=weights)
    islp_pred_unique = islp_spline.predict(x_unique)

    # R fitting
    with ro.conversion.localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['x_r'] = x_unique
        ro.globalenv['y_r'] = y_unique
        ro.globalenv['w_r'] = w_unique if use_weights else ro.NULL
        
        r_code = f"""
        set.seed(10)
        # R's lambda is different from scipy's, so we use df for a fair comparison
        # or match on lambda if that is what is being tested.
        r_smooth_spline <- smooth.spline(x=x_r, y=y_r, w=w_r, {'df=8' if use_df else 'lambda=0.0001'})
        r_pred_object <- predict(r_smooth_spline, x_r)
        r_pred_unique <- r_pred_object$y
        """
        ro.r(r_code)
        r_pred_unique = np.array(ro.globalenv['r_pred_unique'])

    # Comparison
    np.testing.assert_allclose(islp_pred_unique, r_pred_unique, rtol=0.1, atol=0.1)

    # Test prediction on new data
    x_pred_new = np.linspace(x_unique.min(), x_unique.max(), 200)
    islp_pred_new = islp_spline.predict(x_pred_new)
    
    with ro.conversion.localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['x_pred_r'] = x_pred_new
        ro.r('r_pred_new <- predict(r_smooth_spline, x_pred_r)$y')
        r_pred_new = np.array(ro.globalenv['r_pred_new'])
        
    np.testing.assert_allclose(islp_pred_new, r_pred_new, rtol=0.1, atol=0.1)

def test_reinsch_form_verification():
    """
    Verify the sparse Reinsch form EDF calculation against a dense matrix
    formulation and cross-check fitted values with scipy.
    """
    np.random.seed(0)
    x_small = np.array([0.0, 0.5, 1.2, 1.8, 2.5, 3.0, 3.8, 4.2, 5.0,
                        5.5, 6.2, 7.0, 7.5, 8.2, 9.0])
    weights_small = np.array([1.0, 1.2, 0.8, 1.0, 1.5, 0.5, 1.0, 1.0,
                              2.0, 1.0, 0.9, 1.1, 1.0, 0.8, 1.0])
    y_small = np.sin(x_small) + np.random.normal(0, 0.1, size=x_small.shape)
    lam_small = 0.5

    # A. Optimized Sparse Method
    edf_reinsch = compute_edf_reinsch(x_small, lam_small, weights_small)

    # B. Dense Matrix Construction
    n = len(x_small)
    h = np.diff(x_small)
    R_dense = np.zeros((n - 2, n - 2))
    for j in range(n - 2):
        R_dense[j, j] = (h[j] + h[j + 1]) / 3.0
        if j < n - 3:
            R_dense[j, j + 1] = h[j + 1] / 6.0
            R_dense[j + 1, j] = h[j + 1] / 6.0
    Q_dense = np.zeros((n, n - 2))
    for j in range(n - 2):
        Q_dense[j, j] = 1.0 / h[j]
        Q_dense[j + 1, j] = -1.0 / h[j] - 1.0 / h[j + 1]
        Q_dense[j + 2, j] = 1.0 / h[j + 1]
    R_inv = np.linalg.inv(R_dense)
    K_dense = Q_dense @ R_inv @ Q_dense.T
    W_diag = np.diag(weights_small)
    LHS = W_diag + lam_small * K_dense
    S_matrix = np.linalg.solve(LHS, W_diag)
    edf_dense = np.trace(S_matrix)

    np.testing.assert_allclose(edf_reinsch, edf_dense)

    # C. Scipy Cross-Validation
    spl = make_smoothing_spline(x_small, y_small, w=weights_small, lam=lam_small)
    y_scipy = spl(x_small)
    y_our_matrix = S_matrix @ y_small
    np.testing.assert_allclose(y_scipy, y_our_matrix)

    # D. Test SmoothingSpline estimator directly
    islp_spline = SmoothingSpline(lamval=lam_small)
    islp_spline.fit(x_small, y_small, w=weights_small)
    islp_pred = islp_spline.predict(x_small)
    np.testing.assert_allclose(islp_pred, y_our_matrix, rtol=1e-6, atol=1e-6)

    # E. Test prediction on new data
    x_pred_new = np.linspace(x_small.min(), x_small.max(), 200)
    islp_pred_new = islp_spline.predict(x_pred_new)
    scipy_pred_new = spl(x_pred_new)
    np.testing.assert_allclose(islp_pred_new, scipy_pred_new, rtol=1e-6, atol=1e-6)

