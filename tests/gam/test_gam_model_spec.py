"""
ISLP.gam: glmstar's GAMs specified with ModelSpec, smooth features (`s()`,
`lo()`, `re()`) next to ordinary columns, contrasts and bases.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.base import clone

pytest.importorskip('glmnet.gam')

from ISLP import load_data
from ISLP.models import ModelSpec as MS, ns, contrast
from ISLP.gam import GAM, GAMControl, s, lo, re, factor, linear

GROUPS = [f'g{i:02d}' for i in range(10)]

TIGHT = GAMControl(epsnr=1e-12, mxitnr=100, bf_tol=1e-12, bf_maxit=2000)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 300
    D = pd.DataFrame({'a': np.round(rng.uniform(0, 1, n), 2),
                      'b': np.round(rng.uniform(-2, 2, n), 1),
                      'c': pd.Categorical(rng.choice(list('xyz'), n)),
                      'g': rng.choice(GROUPS, n),
                      'v': rng.uniform(0, 2, n)})
    effects = dict(zip(GROUPS, rng.normal(0, 0.5, len(GROUPS))))
    eta = (np.sin(2 * np.pi * D['a']) + 0.4 * D['b'] +
           D['c'].map({'x': 0, 'y': .5, 'z': -.5}).astype(float) + D['g'].map(effects))
    D['y'] = eta + 0.5 * rng.standard_normal(n)
    D['yb'] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return D


@pytest.mark.parametrize('family, y', [(sm.families.Gaussian, 'y'), (sm.families.Binomial, 'yb')])
def test_model_spec_is_the_terms_fit(data, family, y):
    # MS([s(a), s(b), 'c']): the same fit as the terms s(a), s(b), factor(c)
    X = data[['a', 'b', 'c']]
    G = GAM(model_spec=MS([s('a', df=4), s('b', lam=1e-2), 'c']), family=family(),
            control=TIGHT).fit(X, data[y])
    T = GAM(terms=[s('a', df=4), s('b', lam=1e-2), factor('c')], family=family(),
            control=TIGHT).fit(X, data[y])
    np.testing.assert_allclose(G.state_.linear_predictor, T.state_.linear_predictor,
                               rtol=1e-8, atol=1e-9)
    assert list(G.edf_.index) == ['s(a, df=4)', 's(b, lam=0.01)', 'c']
    np.testing.assert_allclose(G.edf_.values, T.edf_.values, rtol=1e-8)
    # predictions at new data go through the spec
    new = X.iloc[:20]
    np.testing.assert_allclose(G.predict(new, 'link'), T.predict(new, 'link'), rtol=1e-8)
    np.testing.assert_allclose(G.partial_effect('s(a, df=4)', new),
                               T.partial_effect('s(a, df=4)', new['a']), rtol=1e-8, atol=1e-10)


def test_parametric_spec_is_least_squares(data):
    # no smooth terms: the least squares fit on the design of the spec
    spec = MS(['a', ns('b', df=3), 'c'], intercept=True)
    G = GAM(model_spec=spec, control=TIGHT).fit(data, data['y'])
    D = spec.fit(data).transform(data)
    ols = sm.OLS(data['y'], D).fit()
    np.testing.assert_allclose(G.predict(data), ols.fittedvalues, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(G.edf_['ns(b, df=3)'], 3)


@pytest.mark.parametrize('smooth, name', [(lambda: s('a', df=4), 's(a, df=4)'),
                                         (lambda: lo('a', span=0.7), 'lo(a, span=0.7)')])
def test_basis_and_smooth(data, smooth, name):
    # a basis (ns) next to a smooth; plain and modified backfitting agree
    spec = MS([ns('b', df=3), smooth(), 'c'])
    fits = [GAM(model_spec=spec, control=GAMControl(epsnr=1e-12, mxitnr=100, bf_tol=1e-13,
                                                    bf_maxit=5000, bf_modified=m)).fit(data, data['y'])
            for m in (False, True)]
    np.testing.assert_allclose(fits[0].state_.linear_predictor, fits[1].state_.linear_predictor,
                               rtol=1e-7, atol=1e-8)
    for G in fits:
        np.testing.assert_allclose(G.predict(data), G.state_.linear_predictor, rtol=1e-8, atol=1e-9)
    assert list(fits[1].edf_.index) == ['ns(b, df=3)', name, 'c']


def test_errors(data):
    with pytest.raises(ValueError, match='one of terms and model_spec'):
        GAM(terms=[s('a')], model_spec=MS([s('a')])).fit(data, data['y'])
    with pytest.raises(ValueError, match='not a GAM term'):
        GAM(terms=['a']).fit(data, data['y'])


def test_wage_against_R(Rinfo):
    # ISLP's Ch07 GAM: wage ~ s(year, 4) + s(age, 5) + education (R's gam).
    # Wage's education is an ordered categorical, which ModelSpec encodes by
    # its codes (one column); contrast() gives an effect per level, as R's factor
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("gam", quietly=TRUE))')[0]:
        pytest.skip('requires the R package gam')
    Wage = load_data('Wage')
    G = GAM(model_spec=MS([s('year', df=4), s('age', df=5),
                           contrast('education', method='drop')]),
            control=GAMControl(epsnr=1e-10, bf_tol=1e-12)).fit(Wage, Wage['wage'])
    W = Wage[['wage', 'year', 'age']].assign(education=Wage['education'].astype(str))
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('W', W)
    rpy.r('''suppressPackageStartupMessages(library(gam))
             W$education <- factor(W$education)
             g <- gam::gam(wage ~ s(year, 4) + s(age, 5) + education, data=W,
                           control=gam::gam.control(epsilon=1e-12, bf.epsilon=1e-12, bf.maxit=200))''')
    fitted_R = np.asarray(rpy.r('fitted(g)'))
    # R matches df to about 1e-4
    np.testing.assert_allclose(G.predict(Wage), fitted_R, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(G.edf_.iloc[:2] - 1, np.asarray(rpy.r('g$nl.df')), atol=1e-3)


def test_intercept_of_the_spec(data):
    # the GAM fits the intercept: a spec with intercept left out is fitted
    # without the column of ones (and without ModelSpec's FutureWarning on its
    # default); with intercept=True the column is not a term: the same fit
    import warnings
    terms = [s('a', df=4), 'b', 'c']
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        G = GAM(model_spec=MS(terms), control=TIGHT).fit(data, data['y'])
    assert G.model_spec_.intercept is False and G.model_spec.intercept is None
    H = GAM(model_spec=MS(terms, intercept=True), control=TIGHT).fit(data, data['y'])
    np.testing.assert_allclose(G.predict(data), H.predict(data), rtol=1e-9, atol=1e-10)
    assert list(G.edf_.index) == list(H.edf_.index) == ['s(a, df=4)', 'b', 'c']


def test_estimator_api(data):
    # model_spec is a parameter: get_params, clone, set_params
    spec = MS([s('a', df=4), 'c'])
    G = GAM(model_spec=spec)
    assert G.get_params()['model_spec'] is spec
    C = clone(G).fit(data, data['y'])
    assert C.model_spec_ is not spec and C.model_spec_.terms_ is not None
    np.testing.assert_allclose(C.predict(data),
                               GAM(model_spec=spec).fit(data, data['y']).predict(data))


def test_plot_and_partial_effect(data):
    # plot and partial_effect take the data frame through the spec
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    G = GAM(model_spec=MS([s('a', df=4), ns('b', df=3), 'c'])).fit(data, data['y'])
    axes = G.plot()
    assert [ax.get_ylabel() for ax in axes] == ['s(a, df=4)']      # blocks: no curve
    fit, std = G.partial_effect('s(a, df=4)', data.iloc[:7], return_std=True)
    np.testing.assert_allclose(fit, G.partial_effect(0, data['a'].iloc[:7].values))
    assert np.all(std > 0)
    import matplotlib.pyplot as plt
    plt.close('all')


def test_joint_model_spec(data):
    # parametric blocks of a ModelSpec (a basis, a contrast) and smooths: the
    # joint solve is backfitting's fit
    spec = MS([s('a', df=4), ns('b', df=3), contrast('c', method='sum'), re('g', lam=0.05)])
    tight = dict(epsnr=1e-12, mxitnr=100, bf_tol=1e-13, bf_maxit=5000)
    fits = [GAM(model_spec=spec, control=GAMControl(**tight, solver=solver)).fit(data, data['y'])
            for solver in ('backfit', 'joint')]
    np.testing.assert_allclose(fits[0].state_.linear_predictor, fits[1].state_.linear_predictor,
                               rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fits[1].predict(data), fits[1].state_.linear_predictor,
                               rtol=1e-10, atol=1e-10)
    # the random effects are named by the groups
    assert list(fits[1].terms_[3].effects_.index) == GROUPS


def test_model_spec_reml(data):
    # REML with a ModelSpec is REML with the terms; the effects named by the
    # groups; standard errors at the REML dispersion
    pytest.importorskip('jax')
    spec = MS([s('a'), 'c', re('g'), re('g', by='v')])
    control = GAMControl(epsnr=1e-12, mxitnr=100, bf_tol=1e-12, bf_maxit=2000)
    G = GAM(model_spec=spec, method='REML', control=control).fit(data, data['y'])
    T = GAM(terms=[s('a'), factor('c'), re('g'), re('g', by='v')], method='REML',
            control=control).fit(data, data['y'])
    np.testing.assert_allclose(G.predict(data), T.predict(data), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(G.variance_components_['variance'].values,
                               T.variance_components_['variance'].values, rtol=1e-5)
    assert list(G.terms_[2].effects_.index) == GROUPS
    _, se = G.predict(data.iloc[:5], return_std=True)
    rows = G.design_.rows(G._columns(G._design(data.iloc[:5]), G.terms_))
    np.testing.assert_allclose(se**2, np.einsum('ij,jk,ik->i', rows, G.posterior_covariance_, rows),
                               rtol=1e-10)


@pytest.mark.parametrize('method', ['REML', 'ML'])
def test_wage_mgcv(Rinfo, method):
    # Wage through a ModelSpec, smoothing parameters by REML/ML, against
    # mgcv: wage ~ s(year) + s(age) + education, our knots given to mgcv
    pytest.importorskip('jax')
    rpy = Rinfo['rpy']
    if not rpy.r('isTRUE(requireNamespace("mgcv", quietly=TRUE))')[0]:
        pytest.skip('requires the R package mgcv')
    Wage = load_data('Wage')
    G = GAM(model_spec=MS([s('year'), s('age'), contrast('education', method='drop')]),
            method=method,
            control=GAMControl(epsnr=1e-12, mxitnr=100, bf_tol=1e-12, bf_maxit=2000)).fit(
                Wage, Wage['wage'])
    knots = {t.name: t.knots_ for t in G.terms_[:2]}
    W = Wage[['wage', 'year', 'age']].assign(education=Wage['education'].astype(str))
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('D', W)
        rpy.r.assign('ky', knots['s(year)'])
        rpy.r.assign('ka', knots['s(age)'])
    rpy.r(f"""D$education <- factor(D$education)
              m <- mgcv::gam(wage ~ s(year, bs='cr', k=length(ky)) + s(age, bs='cr', k=length(ka))
                             + education, data=D, method='{method}', knots=list(year=ky, age=ka),
                             control=mgcv::gam.control(epsilon=1e-12, newton=list(conv.tol=1e-10)))""")
    with Rinfo['np_cv_rules'].context():
        fitted_R = np.asarray(rpy.r('fitted(m)'))
        edf_R = np.asarray(rpy.r('sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))'))
        scale_R = np.asarray(rpy.r('m$reml.scale'))[0]
    np.testing.assert_allclose(G.predict(Wage), fitted_R, rtol=5e-5)
    np.testing.assert_allclose(G.edf_.values[:2], edf_R, rtol=1e-3)
    np.testing.assert_allclose(G.dispersion_, scale_R, rtol=1e-6)
