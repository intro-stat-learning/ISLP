import numpy as np
from sklearn.linear_model import LinearRegression

from ISLP.smoothers import SmoothingFitter, LoessFitter

def test_SmoothingFitter():
    rng = np.random.default_rng(0)
    X = np.linspace(0, 1, 100)
    Y = np.sin(4 * np.pi * X) + rng.normal(0, 0.2, 100)

    # Test with df
    smoother = SmoothingFitter(df=5)
    smoother.fit(X, Y)
    Y_hat = smoother.predict(X)
    assert Y_hat.shape == (100,)

    # Test with lamval
    smoother = SmoothingFitter(lamval=0.1)
    smoother.fit(X, Y)
    Y_hat2 = smoother.predict(X)
    assert Y_hat2.shape == (100,)
    assert not np.allclose(Y_hat, Y_hat2)

    # Test with knots
    smoother = SmoothingFitter(df=5, knots=np.array([0.2, 0.5, 0.8]))
    smoother.fit(X, Y)
    Y_hat3 = smoother.predict(X)
    assert Y_hat3.shape == (100,)
    assert not np.allclose(Y_hat, Y_hat3)

    # Test with n_knots
    smoother = SmoothingFitter(df=5, n_knots=10)
    smoother.fit(X, Y)
    Y_hat4 = smoother.predict(X)
    assert Y_hat4.shape == (100,)
    assert not np.allclose(Y_hat, Y_hat4)

def test_LoessFitter():
    rng = np.random.default_rng(0)
    X = np.linspace(0, 1, 100)
    Y = np.sin(4 * np.pi * X) + rng.normal(0, 0.2, 100)

    # Test with span
    smoother = LoessFitter(span=0.5)
    smoother.fit(X, Y)
    Y_hat = smoother.predict(X)
    assert Y_hat.shape == (100,)

    # Test with degree
    smoother = LoessFitter(span=0.5, degree=1)
    smoother.fit(X, Y)
    Y_hat2 = smoother.predict(X)
    assert Y_hat2.shape == (100,)
    assert not np.allclose(Y_hat, Y_hat2)
