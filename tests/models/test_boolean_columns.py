import pandas as pd
import statsmodels.api as sm
import numpy as np
from itertools import combinations

from ISLP.models import ModelSpec as MS

rng = np.random.default_rng(0)

df = pd.DataFrame({'A':rng.standard_normal(10),
                  'B':np.array([1,2,3,2,1,1,1,3,2,1], int),
                  'C':np.array([True,False,False,True,True]*2, bool),
                  'D':rng.standard_normal(10)})
Y = rng.standard_normal(10)

def test_all():

    for i in range(1, 5):
        for comb in combinations(['A','B','C','D'], i):

            X = MS(comb).fit_transform(df)
            sm.OLS(Y, X).fit() 

