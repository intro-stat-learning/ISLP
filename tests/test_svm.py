import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

from ISLP.svm import plot

def test_plot_basic_execution():
    # Generate a simple dataset
    X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    # Fit an SVC model
    svm = SVC(kernel='linear').fit(X, y)
    
    # Test basic execution without errors
    fig, ax = plt.subplots()
    try:
        plot(X, y, svm, ax=ax)
    except Exception as e:
        assert False, f"`plot` function raised an unexpected exception: {e}"
    finally:
        plt.close(fig) # Close the plot to prevent it from being displayed

def test_plot_value_error_for_insufficient_features():
    # Generate a dataset with only one feature
    X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    # Fit an SVC model (even though it's not ideal for 1 feature, it's for testing input validation)
    svm = SVC(kernel='linear').fit(X, y)
    
    # Expect a ValueError when X has less than 2 features
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match='expecting at least 2 columns to display decision boundary'):
            plot(X[:, :1], y, svm, ax=ax)
    finally:
        plt.close(fig)
