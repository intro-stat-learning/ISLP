# ModelSpec as used in the labs: on the ISLP datasets, and against
# statsmodels' formulas where their codings agree

import numpy as np, pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         contrast,
                         ordinal,
                         poly,
                         ns,
                         bs,
                         pca)
from ISLP.transforms import Poly, NaturalSpline, BSpline

@pytest.fixture(scope='module')
def Carseats():
    return load_data('Carseats')

@pytest.fixture(scope='module')
def Wage():
    return load_data('Wage')

def test_matches_formula(Carseats):

    # treatment coding of categoricals and products of numeric variables
    # agree with statsmodels' formulas (patsy)

    spec = MS(['Price', 'ShelveLoc', 'US', ('Income', 'Advertising')])
    X = spec.fit_transform(Carseats)
    assert list(X.columns) == ['intercept', 'Price',
                               'ShelveLoc[Good]', 'ShelveLoc[Medium]',
                               'US[Yes]', 'Income:Advertising']

    fit = sm.OLS(Carseats['Sales'], X).fit()
    formula = smf.ols('Sales ~ Price + ShelveLoc + US + Income:Advertising',
                      data=Carseats).fit()
    params = formula.params.rename({'Intercept': 'intercept',
                                    'ShelveLoc[T.Good]': 'ShelveLoc[Good]',
                                    'ShelveLoc[T.Medium]': 'ShelveLoc[Medium]',
                                    'US[T.Yes]': 'US[Yes]'})
    np.testing.assert_allclose(fit.params, params[fit.params.index])
    np.testing.assert_allclose(fit.fittedvalues, formula.fittedvalues)

def test_index_kept(Carseats):

    # statsmodels aligns on the index, so it must be that of the data

    D = Carseats.copy()
    D.index = np.arange(1000, 1000 + D.shape[0])
    X = MS(['Price', poly('Income', degree=2), 'ShelveLoc']).fit_transform(D)
    assert (X.index == D.index).all()

def test_names(Carseats):

    spec = MS(['Price', poly('Income', degree=2), ('Price', 'Age')]).fit(Carseats)
    assert spec.names == ['intercept', 'Price', 'poly(Income, degree=2)', 'Price:Age']

    spec = MS(['Price'], intercept=False).fit(Carseats)
    assert spec.names == ['Price']
    assert list(spec.transform(Carseats).columns) == ['Price']

def test_transform_uses_fit(Carseats):

    # the transforms are fit on the training data only, and
    # then applied as is to new data

    train, test = Carseats.iloc[:300], Carseats.iloc[300:]
    spec = MS([poly('Price', degree=3),
               ns('Income', df=4),
               bs('Age', df=5)]).fit(train)
    X = np.asarray(spec.transform(test))

    expected = np.column_stack([
        Poly(degree=3).fit(train[['Price']]).transform(test[['Price']]),
        NaturalSpline(df=4).fit(train[['Income']]).transform(test[['Income']]),
        BSpline(df=5).fit(train[['Age']]).transform(test[['Age']])])
    np.testing.assert_allclose(X[:,1:], expected)

    # fitting on the test data would differ
    other = MS([poly('Price', degree=3),
                ns('Income', df=4),
                bs('Age', df=5)]).fit(test)
    assert not np.allclose(X, np.asarray(other.transform(test)))

def test_categorical_levels_from_fit(Carseats):

    # levels missing from new data keep their columns;
    # levels not seen at fit raise an error

    train, test = Carseats.iloc[:300], Carseats.iloc[300:]
    spec = MS(['Price', 'ShelveLoc']).fit(train)

    good = test[test['ShelveLoc'] == 'Good']
    X = spec.transform(good)
    assert list(X.columns) == ['intercept', 'Price', 'ShelveLoc[Good]', 'ShelveLoc[Medium]']
    assert (X['ShelveLoc[Good]'] == 1).all()
    assert (X['ShelveLoc[Medium]'] == 0).all()

    new = test.copy()
    new['ShelveLoc'] = new['ShelveLoc'].cat.add_categories(['Other'])
    new.iloc[0, new.columns.get_loc('ShelveLoc')] = 'Other'
    with pytest.raises(ValueError, match='unknown categories'):
        spec.transform(new)

@pytest.mark.parametrize('method, drop_level, columns', [
    ('drop', None, ['ShelveLoc[Good]', 'ShelveLoc[Medium]']),
    ('drop', 'Medium', ['ShelveLoc[Bad]', 'ShelveLoc[Good]']),
    ('sum', None, ['ShelveLoc[Bad]', 'ShelveLoc[Good]']),
    ('sum', 'Bad', ['ShelveLoc[Good]', 'ShelveLoc[Medium]']),
    (None, None, ['ShelveLoc[Bad]', 'ShelveLoc[Good]', 'ShelveLoc[Medium]'])])
def test_contrast_methods(Carseats, method, drop_level, columns):

    X = MS([contrast('ShelveLoc', method, drop_level)]).fit_transform(Carseats)
    assert list(X.columns) == ['intercept'] + columns
    level = np.asarray(Carseats['ShelveLoc'])
    C = np.asarray(X[columns])
    if method is None:
        for j, l in enumerate(['Bad', 'Good', 'Medium']):
            np.testing.assert_allclose(C[:,j], level == l)
        return
    if drop_level is None:
        # as R: 'drop' drops the first level, 'sum' codes the last as -1
        drop_level = 'Bad' if method == 'drop' else 'Medium'
    for j, c in enumerate(columns):
        l = c[len('ShelveLoc['):-1]
        np.testing.assert_allclose(C[level == l, j], 1)
    np.testing.assert_allclose(C[level == drop_level], 0 if method == 'drop' else -1)

def test_contrast_of_column(Carseats):

    # contrast() accepts a Column, e.g. from column_info_
    col = MS(['ShelveLoc']).fit(Carseats).column_info_['ShelveLoc']
    X = MS([contrast(col, 'sum')]).fit_transform(Carseats)
    assert list(X.columns) == ['intercept', 'ShelveLoc[Bad]', 'ShelveLoc[Good]']

def test_contrast_bad_method(Carseats):

    with pytest.raises(ValueError, match='method must be one of'):
        MS([contrast('ShelveLoc', 'bogus')]).fit(Carseats)

def test_ordered_categorical(Wage):

    # an ordered categorical gets its integer codes, as ordinal()

    assert Wage['education'].dtype.ordered
    X = MS(['education']).fit_transform(Wage)
    assert list(X.columns) == ['intercept', 'education']
    np.testing.assert_allclose(X['education'], Wage['education'].cat.codes)

    X = MS([ordinal('education')]).fit_transform(Wage)
    assert list(X.columns) == ['intercept', 'ordinal(education)']
    np.testing.assert_allclose(X['ordinal(education)'], Wage['education'].cat.codes)

    # an unordered categorical is dummy coded
    X = MS(['jobclass']).fit_transform(Wage)
    assert list(X.columns) == ['intercept', 'jobclass[2. Information]']

def test_pca_scale(Carseats):

    cols = ['Price', 'Income', 'Advertising', 'Age']
    X = MS([pca(cols, 'P', n_components=2, scale=True)],
           intercept=False).fit_transform(Carseats)
    Z = StandardScaler().fit_transform(Carseats[cols])
    expected = PCA(n_components=2).fit_transform(Z)
    np.testing.assert_allclose(np.asarray(X), expected)

def test_build_sequence_drop(Carseats):

    spec = MS(['Price', 'Income', 'Age']).fit(Carseats)
    X = spec.transform(Carseats)
    seq = list(spec.build_sequence(Carseats, anova_type='drop'))
    # each design drops one term (the intercept first)
    assert [list(D.columns) for D in seq] == [['Price', 'Income', 'Age'],
                                              ['intercept', 'Income', 'Age'],
                                              ['intercept', 'Price', 'Age'],
                                              ['intercept', 'Price', 'Income']]
    for D in seq:
        np.testing.assert_allclose(D, X[D.columns])

    with pytest.raises(ValueError, match='anova_type'):
        list(spec.build_sequence(Carseats, anova_type='bogus'))

def test_ndarray(Carseats):

    # an ndarray gives ndarray designs
    X = np.asarray(Carseats[['Price', 'Income']])
    spec = MS([0, 1, (0, 1)]).fit(X)
    M = spec.transform(X)
    assert isinstance(M, np.ndarray)
    np.testing.assert_allclose(M, np.column_stack([np.ones(X.shape[0]),
                                                   X,
                                                   X[:,0] * X[:,1]]))
    assert [D.shape[1] for D in spec.build_sequence(X)] == [1, 2, 3, 4]

def test_clone_is_unfitted(Carseats):

    spec = MS([poly('Price', degree=2)])
    fitted = clone(spec).fit(Carseats)
    assert hasattr(fitted, 'column_info_')
    assert not hasattr(clone(spec), 'column_info_')
    # clones of an unfitted spec are fit independently
    before = np.asarray(fitted.transform(Carseats))
    clone(spec).fit(Carseats.iloc[:100])
    np.testing.assert_allclose(fitted.transform(Carseats), before)

# fit fits the encoders of the terms (e.g. the Poly of poly()) in place,
# so a fitted spec, or a clone of it, cannot be fit again

@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='ModelSpec.fit fits the encoders of its terms in place')
def test_refit(Carseats):
    spec = MS([poly('Price', degree=2)]).fit(Carseats.iloc[:200])
    spec.fit(Carseats.iloc[200:])

@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='ModelSpec.fit fits the encoders of its terms in place')
def test_clone_fitted_then_fit(Carseats):
    spec = MS([poly('Price', degree=2)]).fit(Carseats.iloc[:200])
    clone(spec).fit(Carseats.iloc[200:])

@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='ModelSpec.fit fits the encoders of its terms in place')
def test_refit_contrast(Carseats):
    # the Contrast of contrast() is kept from the first fit, so a level
    # that only the second fit sees is unknown at transform
    spec = MS([contrast('ShelveLoc', None)]).fit(Carseats[Carseats['ShelveLoc'] != 'Bad'].assign(
        ShelveLoc=lambda df: df['ShelveLoc'].cat.remove_unused_categories()))
    X = spec.fit(Carseats).transform(Carseats)
    assert X.shape[1] == 1 + 3
