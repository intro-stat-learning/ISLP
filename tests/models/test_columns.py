import numpy as np, pandas as pd
from sklearn.base import clone

from pandas.api.types import CategoricalDtype
from ISLP.models.columns import _get_column_info
from ISLP.models.model_spec import Contrast

def test_column_info():

    rng = np.random.default_rng(0)
    cat_type = CategoricalDtype(categories=list("abcd"), ordered=True)
    df = pd.DataFrame(rng.standard_normal((50, 5)), columns=['Aa', 'B', 'Ccc', 'D', 'E'])
    df['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    df['D'] = pd.Categorical(rng.choice(['a','b','c','d'], 50, replace=True))
    df['D'].astype(cat_type)
    print(_get_column_info(df,
                           df.columns,
                           [False]*4+[True],
                           [False]*5,
                           categorical_encoders={'categorical':Contrast(method='drop')}))


