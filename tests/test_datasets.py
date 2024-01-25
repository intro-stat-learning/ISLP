# test that all datasets import

from ISLP import load_data
import numpy as np
import pytest

datasets = ['Auto',
            'Bikeshare',
            'Boston',
            'BrainCancer',
            'Caravan',
            'Carseats',
            'College',
            'Credit',
            'Default',
            'Fund',
            'Hitters',
            'NYSE',
            'OJ',
            'Portfolio',
            'Publication',
            'Smarket',
            'Wage',
            'Weekly']

@pytest.mark.parametrize('dataset', datasets)
def test_load(dataset):
    df = load_data(dataset)
    for col in df.columns:
        assert df[col].dtype != np.dtype(object)
