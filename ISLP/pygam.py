"""
Helper functions for GAMs
=========================

This module contains functions used for the GAM lab of
ISLP.

"""

import numpy as np
import pandas as pd

from scipy.stats import (f as f_dbn,
                         chi2 as chisquared_dbn)
from scipy.optimize import bisect
from pygam.pygam import check_X

import matplotlib.pyplot as plt

def approx_lam(X,
               term,
               df,
               W=None,
               CUTOFF=1e12):
    
    """

    For a given term, try to find multiplier of
    penalty to achieve a specified degrees of freedom.

    Parameters
    ----------

    X : array-like of shape `(n_samples, n_features)`
        Input dataset

    term : Term
        Term for which we which to scale penalty

    df : float
        Desired degrees of freedom

    W : array-like (optional)
        Diagonal weight matrix.

    CUTOFF : float
        Search for solution in interval [0,CUTOFF].

    Returns
    -------

    lam : array-like
        Rescaled lam values.

    Notes
    -----

    The term must be part of a GAM that has already been fit.

    """
    df = df + 1e-7 # to account for boundary case
    
    X = check_X(X)
    X_term = np.asarray(term.build_columns(X).todense())

    if df > X_term.shape[0]:
        raise ValueError('degrees of freedom cannot exceed %d' % X_term.shape[0])
    if df == X_term.shape[0]:
        return 0
    if df <= 0:
        raise ValueError('degrees of freedom must be positive')
    
    evals = _eigvals(X_term, term, W=W)

    def df_(lam):
        return np.sum(1 / (1 + evals * lam)) - df
    
    guess = bisect(df_, 0, CUTOFF)

    if np.sum(1 / (1 + guess * evals)) > df + 1e-3:
        raise ValueError('unable to achieve such a small degrees of freedom -- may be some unpenalized term')

    return guess * np.array(term.lam)

def degrees_of_freedom(X,
                       term,
                       lam=None,
                       W=None):

    """

    For a given term, try to find multiplier of
    penalty to achieve a specified degrees of freedom.

    Parameters
    ----------

    X : array-like of shape `(n_samples, n_features)`
        Input dataset

    term : Term
        Term for which we which to scale penalty.

    lam : array-like (optional)
        Values at which to approximate degrees of freedom.
        If None, defaults to term.lam

    W : array-like (optional)
        Diagonal weight matrix.

    Returns
    -------

    df : float
        Degrees of freedom as computed by the trace of
        the smoother matrix.

    Notes
    -----

    The term must be part of a GAM that has already been fit.

    """
    X = check_X(X)
    D_term = np.asarray(term.build_columns(X).todense())
    if W is not None:
        D_term = D_term * np.sqrt(W[:,None])
    D2 = D_term.T.dot(D_term)

    # find penalty matrix
    if lam is not None:
        old_lam = term.lam
        term.lam = lam
    P = term.build_penalties().todense()
    if lam is not None:
        term.lam = old_lam

    # compute trace of smoother matrix
    df_trace = np.diag(D2.dot(np.linalg.inv(D2 + P))).sum()

    return df_trace

def _eigvals(D_term, term, W=None):
    
    r"""

    D_term : array-like
        Columns of design corresponding to term.

    term : term in GAM

    W : array-like (optional)
        Diagonal weight matrix.

    Construct columns D and penalty matrix P
    for a given term and find eigenvalues in problem

    .. math::

        D^TWDv = \theta Pv

    These can be used to evaluate (for any
    $\lambda$ the trace of the smoother matrix, i.e.

    .. math::

       \text{Tr}(D^TWD (D^TWD+\lambda P)^{-1})

    """

    if W is None:
        W = np.ones(D_term.shape[0])
    u, d, v  = np.linalg.svd(D_term * np.sqrt(W[:,None]), full_matrices=False)

    D_ = d[:,None] * v
    Di_ = np.linalg.pinv(D_) # v.T * (1/d)[None,:] 
    P = term.build_penalties().todense()
    A = Di_.T.dot(P).dot(Di_)

    evals = np.linalg.svd(A)[1] 
    
    return evals

def anova(*models,
          scale=None,
          useF=True):

    """

    Compute an ANOVA table for a sequence of GAM models.

    Parameters
    ----------

    models : GAMs
        Sequence of fitted GAM models.

    scale : float
        Estimate of noise level, defaults to None. 

    useF : bool
        If True use an F distribution for p-value computation
        based on degrees of freedom of largest model. Otherwise,
        use chi-squared.

    Returns
    -------

    anova_table: pd.DataFrame

    Notes
    -----

    Implicitly assumes models are nested, fit on the same `(X,y)` with the 
    same sample weights.

    """

    if scale is None:
        scale = models[-1].statistics_['scale']

    # find the unscaled deviance
    deviances = [m.statistics_['deviance'] * m.statistics_['scale'] for m in models]
    dfs = [m.statistics_['n_samples'] - m.statistics_['edof'] for m in models]

    total_df = models[-1].statistics_['edof']
    deviance_diffs = -np.hstack([np.nan, np.diff(deviances)])
    df_diffs = -np.hstack([np.nan, np.diff(dfs)])

    if useF:
        stat_name = 'F'
        results = np.array([((dev/d)/scale,
                             f_dbn.sf((dev/d)/scale, d, total_df))
                            for (dev, d) in zip(deviance_diffs, df_diffs)])
    else:
        stat_name = 'chi-squared'
        results = np.array([((dev/d)/scale,
                             chisquared_dbn.sf(dev/scale, d))
                            for (dev, d) in zip(deviance_diffs, df_diffs)])

    table = pd.DataFrame({'deviance':deviances,
                          'df':dfs,
                          'deviance_diff':deviance_diffs,
                          'df_diff':df_diffs,
                          stat_name:results[:,0],
                          'pvalue':results[:,1]})
    return table

def plot(gam,
         term_idx,
         quantiles=[0.025,0.975],
         ax=None,
         levels=None,
         partial_kwargs={'c':'b', 'linewidth':4},
         err_kwargs={'c':'r', 'ls':'--', 'linewidth':4},
         bar_kwargs={'capsize':10}):

    """

    Plot the fitted function of a term in a GAM model.

    Parameters
    ----------

    gam : GAM
        A fitted GAM model.

    term_idx : int
        Which term in the GAM to plot?

    quantiles : [float, float], default=[0.025, 0.0975]
        Which quantiles for pointwise confidence bands?

    ax : matplotlib axes, optional
        
    levels : sequence
        For categorical features, which indices to include
        in plot. Defaults to all levels.

    partial_kwargs : dict
        Keyword arguments for partial dependence plot
        for continuous variables.
    
    err_kwargs : dict
        Keyword arguments for pointwise confidence bands
        for continuous variables.
    
    bar_kwargs : dict
        Keyword arguments for barplot for 
        for categorical variables.
    

    Returns
    -------

    ax : matplotlib axes
        Axes with partial dependence plot added.

    """

    if ax is None:
        ax = plt.gca()

    if gam.dtype[term_idx] == 'numerical':
        X_grid = gam.generate_X_grid(term_idx,
                                     meshgrid=False)[:,gam.terms[term_idx].feature]
        partial, bounds = gam.partial_dependence(term_idx,
                                                 quantiles=quantiles,
                                                 X=(X_grid,),
                                                 meshgrid=True)

        ax.plot(X_grid, partial, **partial_kwargs)
        ax.plot(X_grid, bounds[:,0], **err_kwargs)
        ax.plot(X_grid, bounds[:,1], **err_kwargs)
    else:
        if levels is None:
            term = gam.terms[term_idx]
            levels = np.arange(term.edge_knots_[0]+0.5, term.edge_knots_[-1]+.5, 1)
        partial, bounds = gam.partial_dependence(2, 
                                                 quantiles=quantiles,
                                                 X=(levels,),
                                                 meshgrid=True)
        yerr = 0.5 * (bounds[:,1] - bounds[:,0])
        levels_a = np.arange(len(levels))
        ax.bar(levels_a,
               partial,
               yerr=yerr,
               **bar_kwargs)
        ax.set_xticks(levels_a)
        ax.set_xticklabels(levels)
    return ax
