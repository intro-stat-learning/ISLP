"""
**ISLP**  is a Python library to accompany
*Introduction to Statistical Learning with applications in Python*.
See the `statistical learning homepage <http://statlearning.com/>`_
for more details.
"""

from os.path import join as pjoin
from importlib.resources import (as_file,
                                 files)
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics._classification import unique_labels

# data originally saved via: [sm.datasets.get_rdataset(n, 'ISLR').data.to_csv('../ISLP/data/%s.csv' % n, index=False) for n in ['Carseats', 'College', 'Credit', 'Default', 'Hitters', 'Auto', 'OJ', 'Portfolio', 'Smarket', 'Wage', 'Weekly', 'Caravan']]

def _make_categorical(dataset):
    unordered = _unordered.setdefault(dataset, [])
    ordered = _ordered.setdefault(dataset, [])
    with as_file(files('ISLP').joinpath('data', '%s.csv' % dataset)) as filename:
        df = pd.read_csv(filename)
        for col in unordered:
            df[col] = pd.Categorical(df[col])
        for col in ordered:
            df[col] = pd.Categorical(df[col], ordered=True)
        if dataset in _index:
            df = df.set_index(_index[dataset])
        return df

_unordered = {'Hitters':['League', 'Division', 'NewLeague'],
              'Caravan':['Purchase'],
              'Carseats':['ShelveLoc', 'Urban', 'US'],
              'College':['Private'],
              'Publication':['mech'],
              'BrainCancer':['sex', 'diagnosis', 'loc', 'stereo'],
              'Wage':['maritl', 'race', 'region', 'jobclass', 'health', 'health_ins'],
              'Default':['default', 'student'],
              'Credit':['Gender', 'Student', 'Married', 'Ethnicity'],
              'OJ':['Purchase', 'Store7'],
              'Smarket':['Direction'],
              'Weekly':['Direction']
              }
_ordered = {'Wage':['education'],
            }
_index = {'Auto':'name'}

_datasets = sorted(list(_unordered.keys()) +
                   list(_ordered.keys()) +
                   ['NCI60',
                    'Khan',
                    'Bikeshare',
                    'NYSE'])

def load_data(dataset):
    
    if dataset == 'NCI60':
        with as_file(files('ISLP').joinpath('data', 'NCI60data.npy')) as features:
            X = np.load(features)
        with as_file(files('ISLP').joinpath('data', 'NCI60labs.csv')) as labels:
            Y = pd.read_csv(labels)
        return {'data':X, 'labels':Y}
    elif dataset == 'Khan':
        with as_file(files('ISLP').joinpath('data', 'Khan_xtest.csv')) as xtest:
            xtest = pd.read_csv(xtest)
        xtest = xtest.rename(columns=dict([('V%d' % d, 'G%04d' % d) for d in range(1, len(xtest.columns)+0)]))
        with as_file(files('ISLP').joinpath('data', 'Khan_ytest.csv')) as ytest:
            ytest = pd.read_csv(ytest)
        ytest = ytest.rename(columns={'x':'Y'})
        ytest = ytest['Y']
        
        with as_file(files('ISLP').joinpath('data', 'Khan_xtrain.csv')) as xtrain:
            xtrain = pd.read_csv(xtrain)
            xtrain = xtrain.rename(columns=dict([('V%d' % d, 'G%04d' % d) for d in range(1, len(xtest.columns)+0)]))

        with as_file(files('ISLP').joinpath('data', 'Khan_ytrain.csv')) as ytrain:
            ytrain = pd.read_csv(ytrain)
        ytrain = ytrain.rename(columns={'x':'Y'})
        ytrain = ytrain['Y']

        return {'xtest':xtest,
                'xtrain':xtrain,
                'ytest':ytest,
                'ytrain':ytrain}

    elif dataset == 'Bikeshare':
        with as_file(files('ISLP').joinpath('data', '%s.csv' % dataset)) as filename:
            df = pd.read_csv(filename)
        df['weathersit'] = pd.Categorical(df['weathersit'], ordered=False)
        # setting order to avoid alphabetical
        df['mnth'] = pd.Categorical(df['mnth'],
                                    ordered=False,
                                    categories=['Jan', 'Feb',
                                                'March', 'April',
                                                'May', 'June',
                                                'July', 'Aug',
                                                'Sept', 'Oct',
                                                'Nov', 'Dec'])
        df['hr'] = pd.Categorical(df['hr'],
                                  ordered=False,
                                  categories=range(24))
        return df
    elif dataset == 'NYSE':
        with as_file(files('ISLP').joinpath('data', '%s.csv' % dataset)) as filename:
            df = pd.read_csv(filename)
        # setting order to avoid alphabetical
        df['day_of_week'] = pd.Categorical(df['day_of_week'],
                                           ordered=False,
                                           categories=['mon',
                                                       'tues',
                                                       'wed',
                                                       'thur',
                                                       'fri'])
        return df.set_index('date')
    else:
        return _make_categorical(dataset)
load_data.__doc__ = f"""
Load dataset from ISLP package.

Choices are: {_datasets}

Parameters
----------

dataset: str

Returns
-------

data: array-like or dict
    Either a `pd.DataFrame` representing the dataset or a dictionary
    containing different parts of the dataset.
    
"""

def confusion_table(predicted_labels,
                    true_labels,
                    labels=None):
    """
    Return a data frame version of confusion 
    matrix with rows given by predicted label
    and columns the truth.

    Parameters
    ----------

    predicted_labels: array-like
        These will form rows of confusion matrix.

    true_labels: array-like
        These will form columns of confusion matrix.
    """

    if labels is None:
        labels = unique_labels(true_labels,
                               predicted_labels)
    C = _confusion_matrix(true_labels,
                          predicted_labels,
                          labels=labels)
    df = pd.DataFrame(C.T, columns=labels) # swap rows and columns
    df.index = pd.Index(labels, name='Predicted')
    df.columns.name = 'Truth'
    return df
        

from importlib.metadata import version
__version__ = version("ISLP")

