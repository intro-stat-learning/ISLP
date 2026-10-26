import numpy as np

from ISLP.bart import BART
from sklearn.base import clone

def test_bart():

    n, p = 200, 10

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n,p))
    X2 = rng.standard_normal((n,p))
    Y = 3 * X[:,0] + rng.standard_normal(n)
    Y2 = 3 * X2[:,0] + rng.standard_normal(n)

    B = BART(random_state=0)
    B.fit(X, Y)
    pred = B.predict(X2)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))
    # BART should recover much of the signal on new data
    assert np.corrcoef(pred, Y2)[0,1] > 0.5

    # staged predictions are running averages over the draws
    staged = B.staged_predict(X2)
    assert staged.shape == (len(B.trees_sample_), n)
    np.testing.assert_allclose(staged[-1], pred)

    clone(B)

if __name__ == "__main__":

    test_bart()
