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

def test_NCI60():
    NCI60 = load_data('NCI60')
    assert NCI60['data'].shape == (64, 6830)
    assert NCI60['labels'].shape == (64, 1)
    assert NCI60['labels']['label'].nunique() == 14

@pytest.mark.parametrize('dataset', datasets)
def test_load(dataset):
    df = load_data(dataset)
    for col in df.columns:
        assert df[col].dtype != np.dtype(object)

def test_Khan():
    Khan = load_data('Khan')
    assert Khan['xtrain'].shape == (63, 2308)
    assert Khan['xtest'].shape == (20, 2308)
    assert Khan['ytrain'].shape == (63,)
    assert Khan['ytest'].shape == (20,)
    expected = ['G%04d' % d for d in range(1, 2309)]
    assert list(Khan['xtrain'].columns) == expected
    assert list(Khan['xtest'].columns) == expected
    assert set(Khan['ytrain']) == set(Khan['ytest']) == {1, 2, 3, 4}
