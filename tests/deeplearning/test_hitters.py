import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from sklearn.linear_model import \
     (LinearRegression,
      Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.model_selection import \
     (train_test_split,
      GridSearchCV)

# torch

import torch
from torch import nn
from torch.utils.data import TensorDataset

# torch helpers

from torchmetrics import MeanAbsoluteError
from torchinfo import summary

# pytorch lightning

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

# setting seed

from pytorch_lightning import seed_everything
seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)

# ISLP.torch

from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker,
                        rec_num_workers)


def test_hitters(tmp_path,
                 max_epochs=2,
                 num_lam=5):

    Hitters = load_data('Hitters').dropna()
    n = Hitters.shape[0]

    #  We will fit two linear models (least squares  and lasso) and  compare their performance
    # to that of a neural network. For this comparison we will use mean absolute error on a validation dataset.
    # \begin{equation*}
    # \begin{split}
    # \mbox{MAE}(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^n |y_i-\hat{y}_i|.
    # \end{split}
    # \end{equation*}
    # We set up the model matrix and the response.

    # In[11]:


    model = MS(Hitters.columns.drop('Salary'), intercept=False)
    X = model.fit_transform(Hitters).to_numpy()
    Y = Hitters['Salary'].to_numpy()


    # The `to_numpy()`  method above converts `pandas`
    # data frames or series to `numpy` arrays.
    # We do this because we will need to  use `sklearn` to fit the lasso model,
    # and it requires this conversion. 
    # We also use  a linear regression method from `sklearn`, rather than the method
    # in Chapter~3 from `statsmodels`, to facilitate the comparisons.

    # We now split the data into test and training, fixing the random
    # state used by `sklearn` to do the split.

    # In[12]:


    (X_train, 
     X_test,
     Y_train,
     Y_test) = train_test_split(X,
                                Y,
                                test_size=1/3,
                                random_state=1)


    # ### Linear Models
    # We fit the linear model and evaluate the test error directly.

    # In[13]:


    hit_lm = LinearRegression().fit(X_train, Y_train)
    Yhat_test = hit_lm.predict(X_test)
    np.abs(Yhat_test - Y_test).mean()


    # Next we fit the lasso using `sklearn`. We are using
    # mean absolute error to select and evaluate a model, rather than mean squared error.
    # The specialized solver we used in Section 6.5.2 uses only mean squared error. So here, with a bit more work,  we create a cross-validation grid and perform the cross-validation directly.  
    # 
    # We encode a pipeline with two steps: we first normalize the features using a `StandardScaler()` transform,
    # and then fit the lasso without further normalization.

    # In[14]:


    scaler = StandardScaler(with_mean=True, with_std=True)
    lasso = Lasso(warm_start=True, max_iter=30000)
    standard_lasso = Pipeline(steps=[('scaler', scaler),
                                     ('lasso', lasso)])


    # We need to create a grid of values for $\lambda$. As is common practice, 
    # we choose a grid of 100 values of $\lambda$, uniform on the log scale from `lam_max` down to  `0.01*lam_max`. Here  `lam_max` is the smallest value of
    # $\lambda$ with an  all-zero solution. This value equals the largest absolute inner-product between any predictor and the (centered) response. {The derivation of this result is beyond the scope of this book.}

    # In[15]:


    X_s = scaler.fit_transform(X_train)
    n = X_s.shape[0]
    lam_max = np.fabs(X_s.T.dot(Y_train - Y_train.mean())).max() / n
    param_grid = {'alpha': np.exp(np.linspace(0, np.log(0.01), num_lam))
                 * lam_max}


    # Note that we had to transform the data first, since the scale of the variables impacts the choice of $\lambda$.
    # We now perform cross-validation using this sequence of $\lambda$ values.

    # In[16]:


    cv = KFold(10,
               shuffle=True,
               random_state=1)
    grid = GridSearchCV(lasso,
                        param_grid,
                        cv=cv,
                        scoring='neg_mean_absolute_error')
    grid.fit(X_train, Y_train);


    # We extract the lasso model with best cross-validated mean absolute error, and evaluate its
    # performance on `X_test` and `Y_test`, which were not used in
    # cross-validation.

    # In[17]:


    trained_lasso = grid.best_estimator_
    Yhat_test = trained_lasso.predict(X_test)
    np.fabs(Yhat_test - Y_test).mean()


    # This is similar to the results we got for the linear model fit by least squares. However, these results can vary a lot for different train/test splits; we encourage the reader to try a different seed in code block 12 and rerun the subsequent code up to this point.
    # 
    # ### Specifying a Network: Classes and Inheritance
    # To fit the neural network, we first set up a model structure
    # that describes the network.
    # Doing so requires us to define new classes specific to the model we wish to fit.
    # Typically this is done in  `pytorch` by sub-classing a generic
    # representation of a network, which is the approach we take here.
    # Although this example is simple, we will go through the steps in some detail, since it will serve us well
    # for the more complex examples to follow.

    # In[18]:


    class HittersModel(nn.Module):

        def __init__(self, input_size):
            super(HittersModel, self).__init__()
            self.flatten = nn.Flatten()
            self.sequential = nn.Sequential(
                nn.Linear(input_size, 50),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(50, 1))

        def forward(self, x):
            x = self.flatten(x)
            return torch.flatten(self.sequential(x))


    # The `class` statement identifies the code chunk as a
    # declaration for a class `HittersModel`
    # that inherits from the  base class `nn.Module`. This base
    # class is ubiquitous in `torch` and represents the
    # mappings in the neural networks.
    # 
    # Indented beneath the `class` statement are the methods of this class:
    # in this case `__init__` and `forward`.  The `__init__` method is
    # called when an instance of the class is created as in the cell
    # below. In the methods, `self` always refers to an instance of the
    # class. In the `__init__` method, we have attached two objects to
    # `self` as attributes: `flatten` and `sequential`. These are used in
    # the `forward` method to describe the map that this module implements.
    # 
    # There is one additional line in the `__init__` method, which
    # is a call to
    # `super()`. This function allows subclasses (i.e. `HittersModel`)
    # to access methods of the class they inherit from. For example,
    # the class `nn.Module` has its own `__init__` method, which is different from
    # the `HittersModel.__init__()` method we’ve written above.
    # Using `super()` allows us to call the method of the base class. For
    # `torch` models, we will always be making this `super()` call as it is necessary
    # for the model to be properly interpreted by `torch`.
    # 
    # The object `nn.Module` has more methods than simply `__init__` and `forward`. These
    # methods are directly accessible to `HittersModel` instances because of this inheritance.
    # One such method we will see shortly is the `eval()` method, used
    # to disable dropout for when we want to evaluate the model on test data.

    # In[19]:


    hit_model = HittersModel(X.shape[1])


    # The object `self.sequential` is a composition of four maps. The
    # first maps the 19 features of `Hitters` to 50 dimensions, introducing $50\times 19+50$ parameters
    # for the weights and *intercept*  of the map (often called the *bias*). This layer
    # is then mapped to a ReLU layer followed by a 40% dropout layer, and finally a
    # linear map down to 1 dimension, again with a bias. The total number of
    # trainable parameters is therefore $50\times 19+50+50+1=1051$.

    # The package `torchinfo` provides a `summary()` function that neatly summarizes
    # this information. We specify the size of the input and see the size
    # of each tensor as it passes through layers of the network.

    # In[20]:


    summary(hit_model, 
            input_size=X_train.shape,
            col_names=['input_size',
                       'output_size',
                       'num_params'])


    # We have truncated the end of the output slightly, here and in subsequent uses.
    # 
    # We now need to transform our training data into a form accessible to `torch`.
    # The basic
    # datatype in `torch` is a `tensor`, which is very similar
    # to an `ndarray` from early chapters.
    # We also note here that `torch` typically
    # works with 32-bit (*single precision*)
    # rather than 64-bit (*double precision*) floating point numbers.
    # We therefore convert our data to `np.float32` before
    # forming the tensor.
    # The $X$ and $Y$ tensors are then arranged into a `Dataset`
    # recognized by `torch`
    # using `TensorDataset()`.

    # In[21]:


    X_train_t = torch.tensor(X_train.astype(np.float32))
    Y_train_t = torch.tensor(Y_train.astype(np.float32))
    hit_train = TensorDataset(X_train_t, Y_train_t)


    # We do the same for the test data.

    # In[22]:


    X_test_t = torch.tensor(X_test.astype(np.float32))
    Y_test_t = torch.tensor(Y_test.astype(np.float32))
    hit_test = TensorDataset(X_test_t, Y_test_t)


    # Finally, this dataset is passed to a `DataLoader()` which ultimately
    # passes data into our network. While this may seem
    # like a lot of overhead, this structure is helpful for more
    # complex tasks where data may live on different machines,
    # or where data must be passed to a GPU.
    # We provide a helper function `SimpleDataModule()` in `ISLP` to make this task easier for
    # standard usage.
    # One of its arguments is `num_workers`, which indicates
    # how many processes we will use
    # for loading the data. For small
    # data like `Hitters` this will have little effect, but
    # it does provide an advantage for the `MNIST`  and `CIFAR100` examples below.
    # The `torch` package will inspect the process running and determine a
    # maximum number of workers. {This depends on the computing hardware and the number of cores available.} We’ve included a function
    # `rec_num_workers()` to compute this so we know how many
    # workers might be reasonable (here the max was 16).

    # In[23]:


    max_num_workers = rec_num_workers()


    # The general training setup in `pytorch_lightning` involves
    # training, validation and test data. These are each
    # represented by different data loaders. During each epoch,
    # we run a training step to learn the model and a validation
    # step to track the error. The test data is typically
    # used at the end of training to evaluate the model.
    # 
    # In this case, as we had split only into test and training,
    # we’ll use the test data as validation data with the
    # argument `validation=hit_test`. The
    # `validation` argument can be a float between 0 and 1, an
    # integer, or a
    # `Dataset`. If a float (respectively, integer), it is interpreted
    # as a percentage (respectively number) of the *training* observations to be used for validation.
    # If it is a `Dataset`, it is passed directly to a data loader.

    # In[24]:


    hit_dm = SimpleDataModule(hit_train,
                              hit_test,
                              batch_size=32,
                              num_workers=min(4, max_num_workers),
                              validation=hit_test)


    # Next we must provide a `pytorch_lightning` module that controls
    # the steps performed during the training process. We provide methods for our
    # `SimpleModule()` that simply record the value
    # of the loss function and any additional
    # metrics at the end of each epoch. These operations
    # are controlled by the methods `SimpleModule.[training/test/validation]_step()`, though
    # we will not be modifying these in our examples.

    # In[25]:


    hit_module = SimpleModule.regression(hit_model,
                               metrics={'mae':MeanAbsoluteError()})


    #  By using the `SimpleModule.regression()` method,  we indicate that we will use squared-error loss as in
    # (10.23).
    # We have also asked for mean absolute error to be tracked as well
    # in the metrics that are logged.
    # 
    # We log our results via `CSVLogger()`, which in this case stores the results in a CSV file within a directory `logs/hitters`. After the fitting is complete, this allows us to load the
    # results as a `pd.DataFrame()` and visualize them below. There are
    # several ways to log the results within `pytorch_lightning`, though
    # we will not cover those here in detail.

    # In[26]:


    hit_logger = CSVLogger(tmp_path / 'logs', name='hitters')


    # Finally we are ready to train our model and log the results. We
    # use the `Trainer()` object from `pytorch_lightning`
    # to do this work. The argument `datamodule=hit_dm` tells the trainer
    # how training/validation/test logs are produced,
    # while the first argument `hit_module`
    # specifies the network architecture
    # as well as the training/validation/test steps.
    # The `callbacks` argument allows for
    # several tasks to be carried out at various
    # points while training a model. Here
    # our `ErrorTracker()` callback will enable
    # us to compute validation error while training
    # and, finally, the test error.
    # We now fit the model for 50 epochs.

    # In[27]:


    hit_trainer = Trainer(deterministic=True,
                          max_epochs=max_epochs,
                          default_root_dir=tmp_path,
                          log_every_n_steps=5,
                          logger=hit_logger,
                          callbacks=[ErrorTracker()])
    hit_trainer.fit(hit_module, datamodule=hit_dm)


    # At each step of SGD, the algorithm randomly selects 32 training observations for
    # the computation of the gradient. Recall from Section 10.7
    # that an epoch amounts to the number of SGD steps required to process $n$
    # observations. Since the training set has
    # $n=175$, and we specified a `batch_size` of 32 in the construction of  `hit_dm`, an epoch is $175/32=5.5$ SGD steps.
    # 
    # After having fit the model, we can evaluate performance on our test
    # data using the `test()` method of our trainer.

    # In[28]:


    hit_trainer.test(hit_module, datamodule=hit_dm)


    # The results of the fit have been logged into a CSV file. We can find the
    # results specific to this run in the `experiment.metrics_file_path`
    # attribute of our logger. Note that each time the model is fit, the logger will output
    # results into a new subdirectory of our directory `logs/hitters`.
    # 
    # We now create a plot of the MAE (mean absolute error) as a function of
    # the number of epochs.
    # First we retrieve the logged summaries.

    # In[29]:


    hit_results = pd.read_csv(hit_logger.experiment.metrics_file_path)


    # Since we will produce similar plots in later examples, we write a
    # simple generic function to produce this plot.

    # In[30]:


    def summary_plot(results,
                     ax,
                     col='loss',
                     valid_legend='Validation',
                     training_legend='Training',
                     ylabel='Loss',
                     fontsize=20):
        for (column,
             color,
             label) in zip([f'train_{col}_epoch',
                            f'valid_{col}'],
                           ['black',
                            'red'],
                           [training_legend,
                            valid_legend]):
            results.plot(x='epoch',
                         y=column,
                         label=label,
                         marker='o',
                         color=color,
                         ax=ax)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        return ax


    # We now set up our axes, and use our function to produce the MAE plot.

    # In[31]:


    fig, ax = subplots(1, 1, figsize=(6, 6))
    ax = summary_plot(hit_results,
                      ax,
                      col='mae',
                      ylabel='MAE',
                      valid_legend='Validation (=Test)')
    ax.set_ylim([0, 400])
    ax.set_xticks(np.linspace(0, 50, 11).astype(int));


    # We can predict directly from the final model, and
    # evaluate its performance on the test data.
    # Before fitting, we call the `eval()` method
    # of `hit_model`.
    # This tells
    # `torch` to effectively consider this model to be fitted, so that
    # we can use it to predict on new data. For our model here,
    # the biggest change is that the dropout layers will
    # be turned off, i.e. no weights will be randomly
    # dropped in predicting on new data.

    # In[32]:


    hit_model.eval() 
    preds = hit_module(X_test_t)
    torch.abs(Y_test_t - preds).mean()



