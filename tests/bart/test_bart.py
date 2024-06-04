import numpy as np

from ISLP.bart import BART
from sklearn.base import clone

def test_bart():
    # a smoke test
    
    n, p = 1000, 50

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n,p))
    X2 = rng.standard_normal((n,p))
    Y = rng.standard_normal(n)*5

    B = BART()
    B.fit(X, Y)
    print(B.predict(X2))

    clone(B)

if __name__ == "__main__":

    test_bart()
