"""
Adversarial tests for ModelSpec and the transforms it relies on.

Tests marked ``xfail(strict=True)`` document known bugs: once a bug is
fixed the test will XPASS and fail the suite, signalling that the
marker should be removed.
"""
import pickle
from copy import deepcopy

import numpy as np, pandas as pd
import pytest

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import statsmodels.api as sm

from ISLP.models import (ModelSpec as MS,
                         poly,
                         ns,
                         bs,
                         pca,
                         contrast,
                         sklearn_sm)
from ISLP.models.model_spec import Contrast
from ISLP.transforms import Poly, BSpline, NaturalSpline

bug = lambda reason: pytest.mark.xfail(strict=True, reason=reason)

# these tests use ModelSpec's default intercept (True, with a FutureWarning
# until it becomes False in ISLP 0.6, when those relying on it must change)
pytestmark = pytest.mark.filterwarnings("ignore:ModelSpec's `intercept` defaults:FutureWarning")

@pytest.fixture
def D():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({'x': rng.uniform(0, 10, n),
                         'z': rng.standard_normal(n),
                         'g': pd.Categorical(rng.choice(['a', 'b', 'c'], n)),
                         'o': pd.Categorical(rng.choice(['lo', 'mid', 'hi'], n),
                                             categories=['lo', 'mid', 'hi'],
                                             ordered=True),
                         'b': rng.choice([True, False], n)})

# reference values from R: library(splines)
# x <- c(0.5,1.3,2,2.2,3.7,4.1,5,6.6,7,9.4); xn <- c(-1,10,12)

R_x = np.array([0.5, 1.3, 2, 2.2, 3.7, 4.1, 5, 6.6, 7, 9.4])
R_xn = np.array([-1, 10, 12.])

# poly(x, 3); predict(poly(x, 3), xn)
R_poly3 = np.array([
    [-0.432265634900399, 0.480076800172643, -0.492074588241283],
    [-0.33829484470466, 0.202199643711212, 0.0288590978338659],
    [-0.256070403283389, 0.00601813290167871, 0.269254002877709],
    [-0.232577705734454, -0.0419832699991788, 0.306657205224392],
    [-0.0563824741174433, -0.287945646241078, 0.257001756602356],
    [-0.00939707901957392, -0.319544793399292, 0.174814373415067],
    [0.0963200599506324, -0.33831490185743, -0.0557761335361995],
    [0.28426164034211, -0.192784931773284, -0.407723859127603],
    [0.33124703543998, -0.120622628894747, -0.44128781443095],
    [0.613159406027197, 0.612901595379477, 0.360275959382646]])
R_poly3_new = np.array([
    [-0.60846086651741, 1.15539690070455, -2.36769085138778],
    [0.683637498674001, 0.876787224752409, 0.942282386603316],
    [0.918564474163348, 1.9889747566526, 4.46468148147144]])

# ns(x, df=4); predict(ns(x, df=4), xn)
R_ns4 = np.array([
    [0, 0, 0, 0],
    [0.0170445088052199, -0.126976258270368, 0.345216702172563, -0.218240443902195],
    [0.112353939878158, -0.189074505657504, 0.514046312256339, -0.324971806598835],
    [0.163210626930167, -0.192299563791128, 0.522977044802782, -0.330617672001759],
    [0.633757240749562, -0.00854698697500502, 0.239665368301396, -0.151512589156055],
    [0.677452961078211, 0.0905991512403247, 0.167628785520828, -0.105899529804294],
    [0.570168367862708, 0.341126610235441, 0.121936366947747, -0.0649922559664666],
    [0.16969696969697, 0.568006802401707, 0.229595142333996, 0.0327010855673276],
    [0.106864564007421, 0.535077582575308, 0.25926524352366, 0.0987926098936108],
    [0, -0.151489068686521, 0.411860905491479, 0.739628163195042]])
R_ns4_new = np.array([
    [0, 0.257561121712616, -0.700244299656176, 0.442683177943559],
    [0, -0.3714925116199, 0.447495265966603, 0.923997245653297],
    [0, -1.10483732139783, 0.566276467550352, 1.53856085384748]])

# bs(x, df=5)
R_bs5 = np.array([
    [0, 0, 0, 0, 0],
    [0.654880149163343, 0.189217994999522, 0.00752001175001836, 0, 0],
    [0.473573512449963, 0.475227764602206, 0.0495703899537343, 0, 0],
    [0.38716049382716, 0.540679705923152, 0.0721598002496879, 0, 0],
    [0.0387477954144621, 0.578448343340665, 0.359552373149634, 0.0232514880952381, 0],
    [0.0128571428571429, 0.502016006777243, 0.437872937226285, 0.0472539131393298, 0],
    [0, 0.302122347066167, 0.546643085032598, 0.151234567901235, 0],
    [0, 0.0778572239246397, 0.410842908714887, 0.463215720102772, 0.0480841472577009],
    [0, 0.0490296220633299, 0.336645309066147, 0.520410718757826, 0.0939143501126972],
    [0, 0, 0, 0, 1]])

#### numerics: agreement with R

def test_poly_matches_R():
    P = Poly(degree=3).fit(R_x)
    np.testing.assert_allclose(P.transform(R_x), R_poly3, atol=1e-12)
    np.testing.assert_allclose(P.transform(R_xn), R_poly3_new, atol=1e-12)

def test_ns_matches_R():
    N = NaturalSpline(df=4).fit(R_x)
    np.testing.assert_allclose(N.transform(R_x), R_ns4, atol=1e-12)
    np.testing.assert_allclose(N.transform(R_xn), R_ns4_new, atol=1e-12)

def test_bs_matches_R():
    np.testing.assert_allclose(BSpline(df=5).fit_transform(R_x), R_bs5, atol=1e-12)

def test_ns_through_modelspec_matches_R():
    df = pd.DataFrame({'x': R_x})
    M = MS([ns('x', df=4)], intercept=False).fit(df)
    np.testing.assert_allclose(M.transform(pd.DataFrame({'x': R_xn})), R_ns4_new, atol=1e-12)

def test_bspline_partition_of_unity():
    x = np.linspace(0, 1, 50)
    B = BSpline(df=7, intercept=True).fit_transform(x)
    np.testing.assert_allclose(B.sum(1), 1)

def test_ns_linear_beyond_boundary():
    x = np.linspace(0, 1, 50)
    N = NaturalSpline(df=5, intercept=True).fit(x)
    V = N.transform(np.linspace(1.5, 4, 10))
    np.testing.assert_allclose(np.diff(V, 2, axis=0), 0, atol=1e-10)

#### invariants that currently hold

def test_column_map_matches_names(D):
    M = MS(['x', 'g', poly('z', 3), ('x', 'g'), ('g', 'o')]).fit(D)
    X = M.transform(D)
    for term, slc in M.column_map_.items():
        if term != 'intercept':
            assert list(X.columns[slc]) == list(M.column_names_[term])
    assert X.shape[1] == max(s.stop for s in M.column_map_.values())

def test_index_preserved(D):
    D.index = np.arange(100, 100 + D.shape[0])
    X = MS(['x', poly('z', 2), 'g', ('x', 'g'), ns('x', df=3)]).fit_transform(D)
    assert (X.index == D.index).all()
    assert not X.isna().any().any()

def test_transform_ignores_column_order(D):
    M = MS(['x', 'z', 'g', ('x', 'g')]).fit(D)
    np.testing.assert_allclose(M.transform(D), M.transform(D[D.columns[::-1]]))

def test_unused_columns_not_needed_at_transform(D):
    M = MS(['x']).fit(D)
    assert M.transform(D[['x']]).shape == (D.shape[0], 2)

def test_pickle_roundtrip(D):
    M = MS(['x', ns('z', df=3), 'g', pca(['x', 'z'], 'pc', n_components=1)]).fit(D)
    M2 = pickle.loads(pickle.dumps(M))
    np.testing.assert_allclose(M.transform(D), M2.transform(D))

def test_predict_categories_as_strings(D):
    M = MS(['g']).fit(D)
    np.testing.assert_allclose(M.transform(D), M.transform(D.assign(g=D.g.astype(str))))

def test_bool_column_is_float(D):
    X = MS(['b']).fit_transform(D)
    assert (X.dtypes == float).all()

def test_sum_contrast(D):
    X = MS([contrast('g', 'sum')], intercept=False).fit_transform(D)
    np.testing.assert_allclose(X[D.g == 'c'], -1)

def test_cv_pipeline_unfitted_spec(D):
    y = D.x + D.z
    pipe = make_pipeline(MS([poly('x', 3), 'g', ns('z', df=3)]), LinearRegression())
    cross_val_score(pipe, D, y, cv=4, error_score='raise')

def test_gridsearch_over_spec_param(D):
    pipe = make_pipeline(MS([ns('x', df=4)]), LinearRegression())
    GridSearchCV(pipe, {'modelspec__intercept': [True, False]}, cv=3,
                 error_score='raise').fit(D, D.z)

#### refitting: encoders are fit on clones, `terms` is never mutated

# a spec touching every kind of term that carries an encoder
def _rich_terms():
    p = poly('x', 2)
    return ['x',
            p,
            ns('z', df=3),
            bs('x', df=4),
            contrast('g', 'sum'),
            (p, 'g'),
            (bs('z', df=3), 'x'),
            pca(['x', 'z'], 'pc', scale=True, n_components=1)]

def _shifted(D):
    D2 = D.assign(x=D.x * 3 + 100, z=D.z * 5 - 2)
    D2['g'] = pd.Categorical(D.g.astype(str).replace('a', 'c'))
    return D2

def _assert_same_design(A, B):
    assert list(A.columns) == list(B.columns)
    np.testing.assert_allclose(A, B)

def test_refit_matches_fresh_fit(D):
    D2 = _shifted(D)
    M = MS(_rich_terms()).fit(D)
    M.fit(D2)
    _assert_same_design(M.transform(D2), MS(_rich_terms()).fit_transform(D2))

def test_clone_fitted_then_fit(D):
    D2 = _shifted(D)
    M = MS(_rich_terms()).fit(D)
    A = M.transform(D)
    _assert_same_design(clone(M).fit_transform(D2), MS(_rich_terms()).fit_transform(D2))
    _assert_same_design(M.transform(D), A)   # original unaffected

def test_shared_terms_two_specs(D):
    D2 = _shifted(D)
    terms = _rich_terms()
    M1 = MS(terms).fit(D)
    A = M1.transform(D)
    M2 = MS(terms).fit(D2)
    _assert_same_design(M1.transform(D), A)   # fitting M2 leaves M1 alone
    _assert_same_design(M2.transform(D2), MS(_rich_terms()).fit_transform(D2))

def test_terms_not_mutated(D):
    terms = _rich_terms()
    MS(terms).fit(D)
    for t in terms:
        if hasattr(t, 'encoder') and t.encoder is not None:
            with pytest.raises(NotFittedError):
                check_is_fitted(t.encoder)

def test_categorical_encoder_not_mutated(D):
    enc = Contrast(method=None)
    MS(['g'], categorical_encoders={'g': enc}).fit(D)
    with pytest.raises(NotFittedError):
        check_is_fitted(enc)

def test_shared_feature_fit_once(D):
    # poly('x') used as a term and inside an interaction:
    # the interaction must use the same fitted copy
    p = poly('x', 2)
    X = MS([p, (p, 'z')], intercept=False).fit_transform(D)
    np.testing.assert_allclose(X.iloc[:, 2:], X.iloc[:, :2].values * D.z.values[:, None])

def test_cv_prefitted_spec(D):
    M = MS([poly('x', 3), 'g']).fit(D)
    cross_val_score(sklearn_sm(sm.OLS, M), D, D.z, cv=4, error_score='raise')

def test_cv_folds_refit_encoders(D):
    # each fold's design must be fit on that fold alone
    from sklearn.model_selection import KFold
    spec = MS([ns('x', df=4)])
    for train, test in KFold(4).split(D):
        M = clone(spec).fit(D.iloc[train])
        knots = M.terms_[0].encoder.boundary_knots_
        assert knots == [D.x.iloc[train].min(), D.x.iloc[train].max()]

def test_build_submodel_accepts_original_terms(D):
    terms = _rich_terms()
    M = MS(terms).fit(D)
    X = M.transform(D)
    sub = M.build_submodel(D, terms[1:3])
    _assert_same_design(sub, X[['intercept'] + M.column_names_[terms[1]] + M.column_names_[terms[2]]])
    _assert_same_design(M.build_submodel(D, M.terms_[1:3]), sub)

def test_build_submodel_copied_feature_is_unfitted(D):
    # a copy of a term is a different (unfitted) term
    f = poly('x', 2)
    M = MS([f]).fit(D)
    with pytest.raises(NotFittedError):
        M.build_submodel(D, [deepcopy(f)])

def test_pickle_refit(D):
    M = pickle.loads(pickle.dumps(MS(_rich_terms()).fit(D)))
    D2 = _shifted(D)
    _assert_same_design(M.fit_transform(D2), MS(_rich_terms()).fit_transform(D2))

def test_bspline_does_not_mutate_params():
    B = BSpline(df=4).fit(np.linspace(0, 1, 20))
    assert B.lower_bound is None and B.upper_bound is None

def test_bspline_refit_bounds():
    B = BSpline(df=4)
    B.fit(np.linspace(0, 1, 20))
    B.fit(np.linspace(0, 100, 20))
    assert B.boundary_knots_ == [0, 100]

def test_contrast_bad_drop_level_message(D):
    with pytest.raises(ValueError):
        MS([contrast('g', 'drop', 'zzz')]).fit_transform(D)

#### frozen encoders: fit elsewhere, used as is

@pytest.fixture
def frozen(D):
    # models fit on other data, frozen so that ModelSpec.fit does not refit them;
    # they are fit with the column names ModelSpec will pass them
    FrozenEstimator = pytest.importorskip('sklearn.frozen').FrozenEstimator
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsRegressor
    from ISLP.models import derived_feature
    other = D.assign(x=D.x * 2 + 1, z=D.z + 3)
    XZ = other[['x', 'z']]
    knn = KNeighborsRegressor(5).fit(XZ, other.x + other.z)
    pc = PCA(n_components=1).fit(XZ)
    terms = [derived_feature(['x', 'z'], encoder=FrozenEstimator(knn),
                             name='knn', use_transform=False),
             derived_feature(['x', 'z'], encoder=FrozenEstimator(pc), name='pc')]
    return knn, pc, terms

def test_frozen_used_as_is(D, frozen):
    knn, pc, terms = frozen
    X = MS(terms, intercept=False).fit_transform(D)
    XZ = D[['x', 'z']]
    np.testing.assert_allclose(X['knn'], knn.predict(XZ))
    np.testing.assert_allclose(X['pc'], pc.transform(XZ)[:, 0])

def test_frozen_survives_refit_clone_cv_pickle(D, frozen):
    knn, pc, terms = frozen
    components = pc.components_.copy()
    expected = MS(terms + ['g']).fit_transform(D)

    M = MS(terms + ['g']).fit(_shifted(D))
    M.fit(D)                                   # refit: the same frozen objects
    assert M.terms_[0].encoder.estimator is knn
    assert M.terms_[1].encoder.estimator is pc
    _assert_same_design(M.transform(D), expected)

    # clone() deep-copies `terms`, and pickling copies everything, so
    # these hold copies of the frozen models: compare values
    for est in [clone(M).fit(_shifted(D)).fit(D),
                pickle.loads(pickle.dumps(M)).fit(_shifted(D)).fit(D)]:
        _assert_same_design(est.transform(D), expected)

    cross_val_score(make_pipeline(MS(terms + ['g']), LinearRegression()),
                    D, D.z, cv=4, error_score='raise')
    np.testing.assert_allclose(pc.components_, components)   # never refit

def test_frozen_unfitted_raises(D):
    FrozenEstimator = pytest.importorskip('sklearn.frozen').FrozenEstimator
    from sklearn.decomposition import PCA
    from ISLP.models import derived_feature
    f = derived_feature(['x', 'z'], encoder=FrozenEstimator(PCA(1)), name='pc')
    # freezing an unfitted model is an error: ModelSpec tries to fit it
    with pytest.raises((NotFittedError, TypeError)):
        MS([f]).fit_transform(D)

def test_prefit_unfrozen_encoder_is_refit(D):
    # a fitted (but not frozen) encoder is cloned, so refit on the spec's data
    from sklearn.decomposition import PCA
    from ISLP.models import derived_feature
    pc = PCA(n_components=1).fit(D[['x', 'z']].values * 10)
    components = pc.components_.copy()
    M = MS([derived_feature(['x', 'z'], encoder=pc, name='pc')]).fit(D)
    np.testing.assert_allclose(M.terms_[0].encoder.components_,
                               PCA(n_components=1).fit(D[['x', 'z']].values).components_)
    np.testing.assert_allclose(pc.components_, components)   # original untouched

#### known bugs

# strategy.py: the ModelSpec <-> selection boundary

@bug('_postprocess_best reduces state to names; build_submodel cannot resolve derived names')
@pytest.mark.parametrize('term', [poly('z', 2), ('x', 'z')])
def test_stepwise_with_derived_term(D, term):
    from ISLP.models import Stepwise, sklearn_selected
    M = MS(['x', term, 'g']).fit(D)
    strategy = Stepwise.first_peak(M, direction='forward', max_terms=3)
    sklearn_selected(sm.OLS, strategy).fit(D, D.x + D.z)

def test_best_1sd_picks_smaller_model():
    from ISLP.models.strategy import _postprocess_best_1sd
    class T(str):
        name = property(str)
    results = [((T('a'),), 0, np.array([-1.0, -1.1, -0.9])),
               ((T('a'), T('b')), 1, np.array([-0.98, -0.99, -0.97]))]
    assert _postprocess_best_1sd(results)[0] == ('a',)

# sklearn API conformance

def test_unfitted_raises_notfitted(D):
    with pytest.raises(NotFittedError):
        check_is_fitted(MS(['x']))

def test_set_params_categorical_encoders(D):
    M = MS(['g']).set_params(categorical_encoders={'categorical': Contrast(method=None)})
    assert list(M.fit_transform(D).columns) == ['intercept', 'g[a]', 'g[b]', 'g[c]']

@bug('no get_feature_names_out / set_output support')
def test_get_feature_names_out(D):
    M = MS(['x', 'g']).fit(D)
    assert list(M.get_feature_names_out()) == ['intercept', 'x', 'g[b]', 'g[c]']

# input handling

@bug('single-row transform: np.squeeze yields a 0-d array')
@pytest.mark.parametrize('term', [poly('x', 3), ns('x', df=4), bs('x', df=4)])
def test_single_row_predict(D, term):
    M = MS([term]).fit(D)
    assert M.transform(D.iloc[[0]]).shape[0] == 1

def test_ndarray_categorical_features():
    rng = np.random.default_rng(1)
    A = np.column_stack([rng.standard_normal(30), rng.choice([0, 1, 2], 30)])
    assert MS([0, 1], categorical_features=[1]).fit_transform(A).shape == (30, 4)

@bug('pd.Series input fails: no .columns')
def test_series_input(D):
    MS(['x']).fit_transform(D.x)

@bug('fit on ndarray then transform DataFrame fails (int labels via .loc)')
def test_fit_ndarray_transform_df(D):
    A = D[['x', 'z']]
    MS([0, 1]).fit(A.values).transform(A)

# categoricals

@bug('unused level in dtype.categories (e.g. after subsetting) raises')
def test_unused_category_level(D):
    Du = D.assign(g=pd.Categorical(D.g.astype(str), categories=['a', 'b', 'c', 'd']))
    MS(['g']).fit_transform(Du)

@bug('training fold missing a level present in dtype.categories raises')
def test_train_missing_level(D):
    Dt = D[D.g != 'c']           # categories still a, b, c
    MS(['g']).fit(Dt)

@bug('NaN in a categorical is silently encoded as the baseline level')
def test_nan_categorical_not_baseline(D):
    g = D.g.astype(object)
    g.iloc[:3] = np.nan
    X = MS(['g'], intercept=False).fit_transform(D.assign(g=pd.Categorical(g)))
    assert X.iloc[:3].isna().all().all()

def test_contrast_from_column(D):
    col = MS(['g']).fit(D).column_info_['g']
    assert contrast(col, 'drop').name == 'g'

@bug('non-string drop_level on a non-categorical column fails (compared as str)')
def test_contrast_int_drop_level(D):
    k = np.tile([1, 2, 3], D.shape[0] // 3 + 1)[:D.shape[0]]
    X = MS([contrast('k', 'drop', 2)]).fit_transform(D.assign(k=k))
    assert list(X.columns) == ['intercept', 'k[1]', 'k[3]']

# error messages

@bug('missing column gives a generic "should be a Feature, Column..." message')
def test_missing_column_message(D):
    with pytest.raises((KeyError, ValueError), match='nope'):
        MS(['nope']).fit(D)

@bug('BSpline/NaturalSpline with neither df nor knots: TypeError instead of ValueError')
@pytest.mark.parametrize('klass', [BSpline, NaturalSpline])
def test_spline_requires_df_or_knots(klass):
    with pytest.raises(ValueError):
        klass().fit(np.linspace(0, 1, 10))
