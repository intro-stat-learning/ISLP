import numpy as np, pandas as pd
from sklearn.base import clone

from ISLP.transforms import Poly, NaturalSpline, BSpline, Interaction
from ISLP.models.model_spec import ModelSpec, Feature, ns, bs, poly, pca, contrast, Contrast, build_model

from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder)
from sklearn.decomposition import PCA

default_encoders = {'categorical': Contrast(method=None),
                    'ordinal': OrdinalEncoder()}


def test_interaction():

    rng = np.random.default_rng(0)
    I = Interaction(['V', 'U'],
                    {'V':[0,2],
                     'U':[1,3,5]},
                    {'V':[0,1],'U':[0,1,2]})
    X = rng.standard_normal((50,10))
    W = I.fit_transform(X)

    W2 = np.array([X[:,0]*X[:,1],
                   X[:,0]*X[:,3],
                   X[:,0]*X[:,5],
                   X[:,2]*X[:,1],
                   X[:,2]*X[:,3],
                   X[:,2]*X[:,5]]).T
    np.testing.assert_allclose(W, W2)
    assert list(W.columns) == ['0:0', '0:1', '0:2', '1:0', '1:1', '1:2']

def test_ndarray():
    
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50,5))

    M = ModelSpec(terms=[1, (3,2)],
                  categorical_encoders=default_encoders, intercept=True)
    M.fit(X)
    MX = M.transform(X)

    np.testing.assert_allclose(X[:,1], MX[:,1])
    np.testing.assert_allclose(X[:,2] * X[:,3], MX[:,2])
    
def test_dataframe1():
    
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelSpec(terms=['A','D',('D','E')],
                  categorical_encoders=default_encoders, intercept=True)
    clone(M)
    MX = np.asarray(M.fit_transform(D))

    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(X[:,3], MX[:,2])
    np.testing.assert_allclose(X[:,3]*X[:,4], MX[:,3])    

def test_dataframe2():
    
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['V','B','A','D','E'])
    
    M = ModelSpec(terms=['A', 'D', 'B', ('D','E'), 'V'],
                  categorical_encoders=default_encoders, intercept=True)
    clone(M)

    MX = M.fit_transform(D)

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)
    
def test_dataframe3():
    
    rng = np.random.default_rng(8)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    M = ModelSpec(terms=['A', 'E', ('D','E')],
                  categorical_encoders=default_encoders, intercept=True)
    MX = np.asarray(M.fit_transform(D))
    M2 = clone(M)

    DE = pd.get_dummies(D['E'])
    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(DE, MX[:,2:6])    
    np.testing.assert_allclose(np.asarray(DE) * np.multiply.outer(X[:,3], np.ones(4)), MX[:,-4:])

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)

def test_dataframe4():
    
    rng = np.random.default_rng(9)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    M = ModelSpec(terms=['A', 'E', ('D','E'), 'D'],
                  categorical_encoders=default_encoders, intercept=True)
    MX = np.asarray(M.fit_transform(D))

    DE = pd.get_dummies(D['E'])
    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(DE, MX[:,2:6])    

    # check they agree on copy of dataframe

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)

    print(MX2.columns)
    
def test_dataframe5():
    
    rng = np.random.default_rng(10)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    M = ModelSpec(terms=['A', 'E', ('D','E')],
                  categorical_encoders=default_encoders, intercept=True)
    MX = np.asarray(M.fit_transform(D))

    # check they agree on copy of dataframe

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)
    
def test_dataframe6():
    
    rng = np.random.default_rng(11)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    W = Feature(('A','E'), 'AE', None)
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    M = ModelSpec(terms=['A',W,(W,'D',)],
                  categorical_encoders=default_encoders, intercept=True)
    MX = M.fit_transform(D)

    # intercept, A, AE (A and 4 levels of E), AE:D (5 x 3 levels of D)
    assert MX.shape == (50, 1 + 1 + 5 + 15)
    MX = np.asarray(MX)
    DE = np.asarray(pd.get_dummies(D['E']), float)
    DD = np.asarray(pd.get_dummies(D['D']), float)
    np.testing.assert_allclose(MX[:,M.column_map_[W]],
                               np.column_stack([D['A'], DE]))
    AE = np.column_stack([D['A'], DE])
    np.testing.assert_allclose(MX[:,M.column_map_[(W,'D')]],
                               np.einsum('ij,ik->ijk', AE, DD).reshape((50, -1)))

def test_dataframe7():
    
    rng = np.random.default_rng(12)
    X = rng.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['AA','Bbbb','C','Ddd','Y','Eee'])
    D['Ddd'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['Eee'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
        
    M = ModelSpec(terms=D.columns.drop(['Y','C']),
                  categorical_encoders=default_encoders, intercept=True)
    MX = M.fit_transform(D)
    assert list(MX.columns) == ['intercept', 'AA', 'Bbbb',
                                'Ddd[a]', 'Ddd[b]', 'Ddd[c]',
                                'Eee[4]', 'Eee[5]', 'Eee[6]', 'Eee[7]']
    np.testing.assert_allclose(MX[['AA', 'Bbbb']], D[['AA', 'Bbbb']])

def test_dataframe8():
    
    rng = np.random.default_rng(13)
    X = rng.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['A','B','C','D','Y','E'])
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    poly =  Poly(degree=3)
    W = Feature(('A',), 'poly(A)', poly)
    M = ModelSpec(terms=list(D.columns.drop(['Y','C'])) + [(W,'E')],
                  categorical_encoders=default_encoders, intercept=True)
    MX = M.fit_transform(D)

    # main effects: intercept, A, B, 3 levels of D, 4 levels of E;
    # then 3 poly columns x 4 levels of E
    assert MX.shape == (50, 1 + 2 + 3 + 4 + 12)
    P = np.asarray(Poly(degree=3).fit_transform(D[['A']]))
    DE = np.asarray(pd.get_dummies(D['E']), float)
    np.testing.assert_allclose(np.asarray(MX)[:,-12:],
                               np.einsum('ij,ik->ijk', P, DE).reshape((50, -1)))

def test_dataframe9():
    
    rng = np.random.default_rng(14)
    X = rng.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['A','B','C','D','Y','E'])
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    poly =  Poly(degree=3)
    W = Feature(('A',), 'poly(A)', poly)
    U = Feature(('B',), 'poly(B)', clone(poly))
    M = ModelSpec(terms=list(D.columns.drop(['Y','C'])) + [W,U],
                  categorical_encoders=default_encoders, intercept=True)
    MX = M.fit_transform(D)

    # each Feature gets its own fitted copy of the transform
    MX = np.asarray(MX)
    np.testing.assert_allclose(MX[:,M.column_map_[W]],
                               Poly(degree=3).fit_transform(D[['A']]))
    np.testing.assert_allclose(MX[:,M.column_map_[U]],
                               Poly(degree=3).fit_transform(D[['B']]))

def test_dataframe10():
    
    rng = np.random.default_rng(15)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    W = Feature(('A','E'), 'AE', None)
    U = Feature((W, 'C'), 'WC', None)
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    M = ModelSpec(terms=['A', 'E', 'C', W, (W, 'D',), U],
                  categorical_encoders=default_encoders, intercept=True)
    MX = M.fit_transform(D)
    MX = np.asarray(MX)

    V = MX[:,-6:]
    V2 = np.column_stack([MX[:,M.column_map_['A']],
                          MX[:,M.column_map_['E']],
                          MX[:,M.column_map_['C']]])
    np.testing.assert_allclose(V, V2)

def test_poly_ns_bs():
    
    rng = np.random.default_rng(16)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelSpec(terms=[poly('A', intercept=True, degree=3),
                         ns('E', df=5),
                         bs('D', df=4)], intercept=True)

    MX = M.fit_transform(D)
    A =  M.column_info_['A']
    M2 = ModelSpec(terms=[poly(A, intercept=True, degree=3),
                          ns('E', df=5),
                          bs('D', df=4)], intercept=True)
    MX2 = M2.fit_transform(D)
    # 1 intercept + 4 poly + 5 ns + 4 bs
    assert MX.shape == (50, 14)
    assert list(MX.columns) == list(MX2.columns)
    np.testing.assert_allclose(MX, MX2)

def test_submodel():
    
    rng = np.random.default_rng(17)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelSpec(terms=[poly('A', intercept=True, degree=3),
                         ns('E', df=5),
                         bs('D', df=4)], intercept=True)

    M.fit(D)
    MX = M.transform(D)
    MXsub = build_model(M.column_info_,
                        D,
                        M.terms[:2],
                        intercept=M.intercept,
                        encoders=M.encoders_)
    # submodel is the intercept, poly and ns columns of the full model
    assert list(MXsub.columns) == list(MX.columns[:10])
    np.testing.assert_allclose(MXsub, MX.iloc[:,:10])

def test_contrast():
    
    rng = np.random.default_rng(18)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['C'] = pd.Categorical(rng.choice(range(4,9), 50, replace=True))
    # C has 5 levels: 'sum' and 'drop' give 4 columns, None and identity give 5
    for method, ncol in [('sum', 4),
                         ('drop', 4),
                         (None, 5),
                         (lambda p: np.identity(p), 5)]:
        M = ModelSpec(terms=[poly('A', intercept=True, degree=3),
                             contrast('C', method),
                             bs('D', df=4)], intercept=True)

        M.fit(D)
        MX = M.transform(D)
        MXsub = build_model(M.column_info_,
                            D,
                            M.terms[:2],
                            intercept=M.intercept,
                            encoders=M.encoders_)
        assert MX.shape == (50, 1 + 4 + ncol + 4)
        assert list(MXsub.columns) == list(MX.columns[:5 + ncol])
        np.testing.assert_allclose(MXsub, MX.iloc[:,:5 + ncol])

        C = np.asarray(MX.iloc[:,5:5 + ncol])
        if method == 'drop':
            # first level is dropped
            assert np.all(C[D['C'] == 4] == 0)
        elif method == 'sum':
            # last level is coded as -1 in every column
            assert np.all(C[D['C'] == 8] == -1)
        else:
            np.testing.assert_allclose(C, pd.get_dummies(D['C']))
    
def test_sequence():
    
    rng = np.random.default_rng(19)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelSpec(terms=[poly('A', intercept=True, degree=3),
                         ns('E', df=5),
                         bs('D', df=4)], intercept=True)
    M.fit(D)
    MX = M.transform(D)
    # each design adds one term to the previous one
    seq = list(M.build_sequence(D))
    assert [df.shape[1] for df in seq] == [1, 5, 10, 14]
    for df in seq:
        assert list(df.columns) == list(MX.columns[:df.shape[1]])
        np.testing.assert_allclose(df, MX.iloc[:,:df.shape[1]])
    
def test_poly_ns_bs2():
    
    rng = np.random.default_rng(20)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['C'] = pd.Categorical(rng.choice(range(4,9), 50, replace=True))
    M = ModelSpec(terms=[(poly('A', intercept=True, degree=3),'C')], intercept=True)
    MX = M.fit_transform(D)
    # intercept + 4 poly columns x 4 levels of C (first level dropped)
    assert MX.shape == (50, 1 + 4 * 4)
    assert MX.columns[1] == 'poly(A, degree=3, intercept=True)[0]:C[5]'
    P = np.asarray(Poly(degree=3, intercept=True).fit_transform(D[['A']]))
    DC = np.asarray(pd.get_dummies(D['C']), float)[:,1:]
    np.testing.assert_allclose(np.asarray(MX)[:,1:],
                               np.einsum('ij,ik->ijk', P, DC).reshape((50, -1)))

    
def test_pca():
    
    rng = np.random.default_rng(21)
    X = rng.standard_normal((50,8))
    D = pd.DataFrame(X, columns=['A','B','C','D','E', 'F', 'G', 'H'])
    
    pca_ = Feature(('A','B','C','D'), 'pca(ABCD)', PCA(n_components=2))
    M = ModelSpec(terms=[poly('F', intercept=True, degree=3),
                         pca_], intercept=True)

    MX = M.fit_transform(D)

    M2 = ModelSpec(terms=[poly('F', intercept=True, degree=3),
                          pca(['A','B','C','D'], 'ABCD', n_components=2)], intercept=True)
    MX2 = M2.fit_transform(D)

    np.testing.assert_allclose(MX, MX2)

    
    
