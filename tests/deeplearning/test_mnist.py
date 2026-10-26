
# torch

import pytest
import torch
from torch import nn

# torch helpers

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
                        ErrorTracker)

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

@pytest.mark.network
def test_mnist(tmp_path, max_epochs=2):


    # ## Multilayer Network on the MNIST Digit Data
    # The `torchvision` package comes with a number of example datasets,
    # including the `MNIST`  digit data. Our first step is to retrieve
    # the training and test data sets; the `MNIST()` function within
    # `torchvision.datasets` is provided for this purpose. The
    # data will be downloaded the first time this function is executed, and stored in the directory `data/MNIST`.

    # In[34]:


    (mnist_train, 
     mnist_test) = [MNIST(root=tmp_path / 'data',
                          train=train,
                          download=True,
                          transform=ToTensor())
                    for train in [True, False]]
    mnist_train


    # There are 60,000 images in the training data and 10,000 in the test
    # data. The images are $28\times 28$, and stored as a matrix of pixels. We
    # need to transform each one into a vector.
    # 
    # Neural networks are somewhat sensitive to the scale of the inputs, much as ridge and
    # lasso regularization are affected by scaling.  Here the inputs are eight-bit
    # grayscale values between 0 and 255, so we rescale to the unit
    # interval. {Note: eight bits means $2^8$, which equals 256. Since the convention
    # is to start at $0$, the possible values  range from $0$ to $255$.}
    # This transformation, along with some reordering
    # of the axes, is performed by the `ToTensor()` transform
    # from the `torchvision.transforms` package.
    # 
    # As in our `Hitters` example, we form a data module
    # from the training and test datasets, setting aside 20%
    # of the training images for validation.

    # In[35]:


    mnist_dm = SimpleDataModule(mnist_train,
                                mnist_test,
                                validation=0.2,
                                num_workers=2,
                                batch_size=256)


    # Let’s take a look at the data that will get fed into our network. We loop through the first few
    # chunks of the test dataset, breaking after 2 batches:

    # In[36]:


    for idx, (X_ ,Y_) in enumerate(mnist_dm.train_dataloader()):
        print('X: ', X_.shape)
        print('Y: ', Y_.shape)
        if idx >= 1:
            break


    # We see that the $X$ for each batch consists of 256 images of size `1x28x28`.
    # Here the `1` indicates a single channel (greyscale). For RGB images such as `CIFAR100` below,
    # we will see that the `1` in the size will be replaced by `3` for the three RGB channels.
    # 
    # Now we are ready to specify our neural network.

    # In[37]:


    class MNISTModel(nn.Module):
        def __init__(self):
            super(MNISTModel, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 256),
                nn.ReLU(),
                nn.Dropout(0.4))
            self.layer2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3))
            self._forward = nn.Sequential(
                self.layer1,
                self.layer2,
                nn.Linear(128, 10))
        def forward(self, x):
            return self._forward(x)


    # We see that in the first layer, each `1x28x28` image is flattened, then mapped to
    # 256 dimensions where we apply a ReLU activation with 40% dropout.
    # A second layer maps the first layer’s output down to
    # 128 dimensions, applying a ReLU activation with 30% dropout. Finally,
    # the 128 dimensions are mapped down to 10, the number of classes in the
    # `MNIST`  data.

    # In[38]:


    mnist_model = MNISTModel()


    # We can check that the model produces output of expected size based
    # on our existing batch `X_` above.

    # In[39]:


    mnist_model(X_).size()


    # Let’s take a look at the summary of the model. Instead of an `input_size` we can pass
    # a tensor of correct shape. In this case, we pass through the final
    # batched `X_` from above.

    # In[40]:


    summary(mnist_model,
            input_data=X_,
            col_names=['input_size',
                       'output_size',
                       'num_params'])


    # Having set up both  the model and the data module, fitting this model is
    # now almost identical to the `Hitters` example. In contrast to our regression model, here we will use the
    # `SimpleModule.classification()` method which
    # uses the  cross-entropy loss function instead of mean squared error. It must be supplied with the number of classes in the problem.

    # In[41]:


    mnist_module = SimpleModule.classification(mnist_model,
                                               num_classes=10)
    mnist_logger = CSVLogger(tmp_path / 'logs', name='MNIST')


    # Now we are ready to go. The final step is to supply training data, and fit the model.

    # In[42]:


    mnist_trainer = Trainer(deterministic=True,
                            max_epochs=max_epochs,
                            default_root_dir=tmp_path,
                            logger=mnist_logger,
                            callbacks=[ErrorTracker()])
    mnist_trainer.fit(mnist_module,
                      datamodule=mnist_dm)


    # We have suppressed the output here, which is a progress report on the
    # fitting of the model, grouped by epoch. This is very useful, since on
    # large datasets fitting can take time. Fitting this model took 245
    # seconds on a MacBook Pro with an Apple M1 Pro chip with 10 cores and 16 GB of RAM.
    # Here we specified a
    # validation split of 20%, so training is actually performed on
    # 80% of the 60,000 observations in the training set. This is an
    # alternative to actually supplying validation data, like we did for the `Hitters` data.
    # SGD  uses batches
    # of 256 observations in computing the gradient, and doing the
    # arithmetic, we see that an epoch corresponds to 188 gradient steps.

    # `SimpleModule.classification()` includes
    # an accuracy metric by default. Other
    # classification metrics can be added from `torchmetrics`.
    # We will use  our `summary_plot()` function to display 
    # accuracy across epochs.


    mnist_trainer.test(mnist_module,
                       datamodule=mnist_dm)


    # Table 10.1 also reports the error rates resulting from LDA (Chapter 4) and multiclass logistic
    # regression. For LDA we refer the reader to Section 4.7.3.
    # Although we could use the `sklearn` function `LogisticRegression()` to fit  
    # multiclass logistic regression, we are set up here to fit such a model
    # with `torch`.
    # We just have an input layer and an output layer, and omit the hidden layers!

    # In[45]:


    class MNIST_MLR(nn.Module):
        def __init__(self):
            super(MNIST_MLR, self).__init__()
            self.linear = nn.Sequential(nn.Flatten(),
                                        nn.Linear(784, 10))
        def forward(self, x):
            return self.linear(x)

    mlr_model = MNIST_MLR()
    mlr_module = SimpleModule.classification(mlr_model,
                                             num_classes=10)
    mlr_logger = CSVLogger(tmp_path / 'logs', name='MNIST_MLR')


    # In[46]:


    mlr_trainer = Trainer(deterministic=True,
                          max_epochs=max_epochs,
                          default_root_dir=tmp_path,
                          callbacks=[ErrorTracker()])
    mlr_trainer.fit(mlr_module, datamodule=mnist_dm)


    # We fit the model just as before and compute the test results.

    # In[47]:


    mlr_trainer.test(mlr_module,
                     datamodule=mnist_dm)


    # The accuracy is above 90% even for this pretty simple model.
    # 
    # As in the `Hitters` example, we delete some of
    # the objects we created above.

    # In[48]:




