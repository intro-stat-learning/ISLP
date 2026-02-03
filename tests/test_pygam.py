import numpy as np
import pandas as pd
from pygam import LinearGAM, s, l

from ISLP.pygam import degrees_of_freedom, anova

def test_degrees_of_freedom():
    # Create a synthetic dataset
    X = np.random.rand(100, 1)
    y = np.sin(X).ravel()
    
    # Create a spline term
    term = s(0)
    
    # Fit a GAM to build the term
    gam = LinearGAM(term).fit(X, y)
    
    # Get the built term
    built_term = gam.terms[0]
    
    # Calculate degrees of freedom
    df = degrees_of_freedom(X, built_term)
    
    # Check that the result is a float
    assert isinstance(df, float)
    
    # Check that the value is plausible
    assert 0 < df < built_term.n_coefs

def test_anova():
    # Create a synthetic dataset
    X = np.random.rand(100, 3)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100)
    
    # Fit nested GAM models
    gam1 = LinearGAM(l(0)).fit(X, y)
    gam2 = LinearGAM(l(0) + l(1)).fit(X, y)
    gam3 = LinearGAM(l(0) + l(1) + l(2)).fit(X, y)
    
    # Compute ANOVA table
    anova_table = anova(gam1, gam2, gam3)
    
    # Check the shape of the output
    assert isinstance(anova_table, pd.DataFrame)
    assert anova_table.shape[0] == 3
    
    # Check column names
    expected_columns = ['deviance', 'df', 'deviance_diff', 'df_diff', 'F', 'pvalue']
    assert all(col in anova_table.columns for col in expected_columns)
    
    # Check that p-values are between 0 and 1
    assert np.all((anova_table['pvalue'].dropna() >= 0) & (anova_table['pvalue'].dropna() <= 1))
