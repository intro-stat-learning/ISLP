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
    print(np.linalg.norm(W-W2))
    print(W.columns)

def test_ndarray():
    
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50,5))

    M = ModelSpec(terms=[1, (3,2)],
                  categorical_encoders=default_encoders)
    M.fit(X)
    MX = M.transform(X)

    np.testing.assert_allclose(X[:,1], MX[:,1])
    np.testing.assert_allclose(X[:,2] * X[:,3], MX[:,2])
    
def test_dataframe1():
    
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelSpec(terms=['A','D',('D','E')],
                  categorical_encoders=default_encoders)
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
                  categorical_encoders=default_encoders)
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
                  categorical_encoders=default_encoders)
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
                  categorical_encoders=default_encoders)
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
                  categorical_encoders=default_encoders)
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
                  categorical_encoders=default_encoders)
    MX = M.fit_transform(D)

    MX = np.asarray(MX)

def test_dataframe7():
    
    rng = np.random.default_rng(12)
    X = rng.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['AA','Bbbb','C','Ddd','Y','Eee'])
    D['Ddd'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['Eee'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
        
    M = ModelSpec(terms=D.columns.drop(['Y','C']),
                  categorical_encoders=default_encoders)
    MX = M.fit_transform(D)
    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe8():
    
    rng = np.random.default_rng(13)
    X = rng.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['A','B','C','D','Y','E'])
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    poly =  Poly(degree=3)
    # raises a ValueError because poly will have been already fit -- need new instance of Poly
    W = Feature(('A',), 'poly(A)', poly)
    M = ModelSpec(terms=list(D.columns.drop(['Y','C'])) + [(W,'E')],
                  categorical_encoders=default_encoders)
    MX = M.fit_transform(D)

    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe9():
    
    rng = np.random.default_rng(14)
    X = rng.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['A','B','C','D','Y','E'])
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    poly =  Poly(degree=3)
    # raises a ValueError because poly will have been already fit -- need new instance of Poly
    W = Feature(('A',), 'poly(A)', poly)
    U = Feature(('B',), 'poly(B)', clone(poly))
    M = ModelSpec(terms=list(D.columns.drop(['Y','C'])) + [W,U],
                  categorical_encoders=default_encoders)
    MX = M.fit_transform(D)

    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe10():
    
    rng = np.random.default_rng(15)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    W = Feature(('A','E'), 'AE', None)
    U = Feature((W, 'C'), 'WC', None)
    D['D'] = pd.Categorical(rng.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(rng.choice(range(4,8), 50, replace=True))
    
    M = ModelSpec(terms=['A', 'E', 'C', W, (W, 'D',), U],
                  categorical_encoders=default_encoders)
    MX = M.fit_transform(D)
    print(MX.columns)
    MX = np.asarray(MX)

    V = MX[:,-6:]
    V2 = np.column_stack([MX[:,M.column_map_['A']],
                          MX[:,M.column_map_['E']],
                          MX[:,M.column_map_['C']]])
    print(np.linalg.norm(V-V2))

def test_poly_ns_bs():
    
    rng = np.random.default_rng(16)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelSpec(terms=[poly('A', intercept=True, degree=3),
                         ns('E', df=5),
                         bs('D', df=4)])

    MX = M.fit_transform(D)
    A =  M.column_info_['A']
    M2 = ModelSpec(terms=[poly(A, intercept=True, degree=3),
                          ns('E', df=5),
                          bs('D', df=4)])
    MX2 = M2.fit_transform(D)
    print(MX.columns)
    print(MX2.columns)

def test_submodel():
    
    rng = np.random.default_rng(17)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelSpec(terms=[poly('A', intercept=True, degree=3),
                         ns('E', df=5),
                         bs('D', df=4)])

    M.fit(D)
    MX = M.transform(D)
    MXsub = build_model(M.column_info_,
                        D,
                        M.terms[:2],
                        intercept=M.intercept,
                        encoders=M.encoders_)
    print(MX.columns)
    print(MXsub.columns)

def test_contrast():
    
    rng = np.random.default_rng(18)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['C'] = pd.Categorical(rng.choice(range(4,9), 50, replace=True))
    for method in ['sum', 'drop', None, lambda p: np.identity(p)]:
        M = ModelSpec(terms=[poly('A', intercept=True, degree=3),
                             contrast('C', method),
                             bs('D', df=4)])

        M.fit(D)
        MX = M.transform(D)
        MXsub = build_model(M.column_info_,
                            D,
                            M.terms[:2],
                            intercept=M.intercept,
                            encoders=M.encoders_)
        print(method, MX.columns)
    print(MXsub.columns)
    
def test_sequence():
    
    rng = np.random.default_rng(19)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelSpec(terms=[poly('A', intercept=True, degree=3),
                         ns('E', df=5),
                         bs('D', df=4)])
    M.fit(D)
    for df in M.build_sequence(D):
        print(df.columns)
    
def test_poly_ns_bs2():
    
    rng = np.random.default_rng(20)
    X = rng.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['C'] = pd.Categorical(rng.choice(range(4,9), 50, replace=True))
    M = ModelSpec(terms=[(poly('A', intercept=True, degree=3),'C')])
    MX = M.fit_transform(D)
    print(MX.columns)

    
def test_pca():
    
    rng = np.random.default_rng(21)
    X = rng.standard_normal((50,8))
    D = pd.DataFrame(X, columns=['A','B','C','D','E', 'F', 'G', 'H'])
    
    pca_ = Feature(('A','B','C','D'), 'pca(ABCD)', PCA(n_components=2))
    M = ModelSpec(terms=[poly('F', intercept=True, degree=3),
                         pca_])

    MX = M.fit_transform(D)

    M2 = ModelSpec(terms=[poly('F', intercept=True, degree=3),
                          pca(['A','B','C','D'], 'ABCD', n_components=2)])
    MX2 = M2.fit_transform(D)

    np.testing.assert_allclose(MX, MX2)

    
    
