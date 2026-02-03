"""
Helper functions for survival analysis
======================================

This module contains functions used for the survival analysis lab of
ISLP. Currently it contains just a simple function to generate
survival times from a Cox proportional hazards model with a given
baseline cumulative hazard.

"""

import numpy as np
from scipy.optimize import root

def sim_time(linpred,
             cum_hazard,
             rng=None):
    """
    Simulate a survival time for a 
    cumulative hazard function $H$ with cumulative hazard

    .. math::

        H_l(t) = e^l \cdot H(t)
    
    with `l` the linear predictor `linpred` as in a
    Cox proportional hazards model.

    Parameters
    ----------

    lin_pred : float
        Linear predictor value.

    cum_hazard : callable
        Cumulative hazard function, takes a single non-negative argument.

    rng : numpy random number generator
        Used to generate survival times.
    """

    if rng is None:
        rng = np.random.default_rng()
        
    U = rng.uniform()
    B = - np.log(U) /  np.exp(linpred)
    lower, upper = 1, 2

    while True:
        if cum_hazard(lower) > B:
            lower /= 2
        if cum_hazard(upper) < B:
            upper *= 2
        if ((cum_hazard(lower) < B) and
            (cum_hazard(upper) > B)):
            break
    T = root(lambda t: cum_hazard(t) - B,
             lower).x.item()

    return T
