"""
Fixtures for the ISLP.gam tests: R through rpy2 (for the comparisons with gam
and mgcv; the tests skip without it).
"""
import pytest


@pytest.fixture(scope='session')
def Rinfo():
    try:
        import rpy2.robjects as rpy
        from rpy2.robjects import default_converter, numpy2ri, pandas2ri
        rpy.r('1')
    except Exception as e:
        pytest.skip(f'R comparisons need rpy2 and R: {e}')
    return {'rpy': rpy,
            'np_cv_rules': default_converter + numpy2ri.converter + pandas2ri.converter}
