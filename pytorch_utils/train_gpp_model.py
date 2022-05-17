#!/usr/bin/env python
# coding: utf-8

# In[165]:


## import modules
import sys
from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import LRP, IntegratedGradients as IG
from torch import Tensor
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm, trange
from scipy import stats

sys.path.append('../')

from pytorch_utils.data_pipeline import TSData
from hybrid_model import HybridModel

from captum.attr._utils.lrp_rules import IdentityRule
from captum.attr._core.lrp import SUPPORTED_NON_LINEAR_LAYERS

# ## Data loader for a sequential model
# 
# * For the training loader, we use a window size of 100 time steps from the range 1980-01-01 to 2000-12-31
# * For the validation loader, we use the entire sequence 2001-01-01 to 2020-12-31


def load_data():
    ## some settings
    LCT = 'MF'  # land cover type
    norm_kind = 'mean_std'  # min_max
    deseasonalize = False
    target_var = 'GPP'


    # In[31]:


    ## load data from csv to Xarray dataset
    df = pd.read_csv('../simple_gpp_model/data/predictor-variables+GPP_Jena_' + LCT + '.csv', index_col=0, parse_dates=True)

    if deseasonalize:
        df = df.groupby(by=df.index.dayofyear).transform(lambda x: x - x.mean())

    ds = df.to_xarray().rename({'index': 'time'})
    ds = ds.sel(time=slice('1982-01-15', None))  # FAPAR is NaN the first two weeks


    # In[32]:


    variables = ['t2mmin', 'vpd', 'ssrd', 'FPAR', 'GPP', 'GPP_constant-Tmin', 'GPP_constant-SWrad', 'GPP_constant-VPD']
    df.loc['2016'][variables].plot(subplots=True, layout=(4, 2), figsize=(14, 10), ax=None)
    plt.show()


    # In[33]:


    train_loader = torch.utils.data.DataLoader(
        TSData(ds=ds, features=['ssrd', 't2mmin', 'vpd', 'FPAR'], targets=target_var, time_slice=slice('1982', '2000'),
               normalize=False,
               #norm_kind=norm_kind,
               ),
        batch_size=50,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TSData(ds=ds, features=['ssrd', 't2mmin', 'vpd', 'FPAR'], targets=target_var, time_slice=slice('2001', '2016'),
               normalize=False,
               #norm_kind=norm_kind,
               return_seq=True,
               ts_window_size=-1,
               norm_stats=train_loader.dataset.norm_stats
               ),
        batch_size=1,
        shuffle=False)

    return train_loader, valid_loader

# In[34]:

def prepare_lrp():
    for nl in [nn.Sigmoid, nn.Identity]:
        if nl in SUPPORTED_NON_LINEAR_LAYERS:
            continue

        SUPPORTED_NON_LINEAR_LAYERS.append(nl)

    ACTIVATIONS = {'relu': nn.ReLU,
                   'sigmoid': nn.Sigmoid,
                   'tanh': nn.Tanh,
                   'identity': nn.Identity}

    return SUPPORTED_NON_LINEAR_LAYERS, ACTIVATIONS


@dataclass(unsafe_hash=True)
class FCNLayer(nn.Module):
    in_features: int
    out_features: int
    activation: str = 'sigmoid'
    dropout_rate: float = 0.0

    def __post_init__(self):
        super(FCNLayer, self).__init__()
        self.fc = nn.Linear(self.in_features, self.out_features)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else None
        SUPPORTED_NON_LINEAR_LAYERS, ACTIVATIONS = prepare_lrp()
        self.activation_fcn = ACTIVATIONS[self.activation]()

    def forward(self, x):
        x = self.fc(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation_fcn(x)
        return x


class FCN(nn.Module):
    def __init__(
            self,
            num_feature: int,
            num_targets: int = 1,
            num_hidden: int = 8,
            num_layers: int = 2,
            dropout: float = 0.0,
            activation: str = 'sigmoid',
            output_activation: str = 'identity',
            learning_rate: float = 0.001,
            weight_decay: float = 0.0) -> None:

        """A fully connected feed-forward model.
        
        Parameters
        -----------
        num_features: int
            The number of input features.
        num_targets: int (default is 1)
            The number of targets.
        num_hidden: int (default is 8)
            The number of hidden nodes per layer.
        num_layers: int (default is 2)
            The number of hidden nodes.
        dropout: float (default is 0.0)
            The dropout used in all hidden layers.
        activation: str (default is sigmoid)
            The activation function.
        output_activation: str (default is identity)
            The output activation function, default is nn.Identity and does not transform the output.
        learning_rate: float (default is 0.001):
            The learning rate.
        weight_decay: float (default is 0.0)
            The weight decay (L2 regularization).
        """
        super(FCN, self).__init__()
        self.valid_losses = None
        self.train_losses = None

        layers = OrderedDict()
        in_sizes = [num_feature] + [num_hidden] * num_layers
        out_sizes = [num_hidden] * num_layers + [num_targets]

        for i, (nin, nout) in enumerate(zip(in_sizes, out_sizes)):
            is_input_layer = i == 0
            is_output_layer = i == num_layers

            if not is_input_layer and not is_output_layer:
                dropout_rate = dropout
            else:
                dropout_rate = 0.0

            nl = output_activation if is_output_layer else activation
            layer = FCNLayer(in_features=nin, out_features=nout, dropout_rate=dropout_rate, activation=nl)

            layers.update({f'layer{i}': layer})

        self.model = nn.Sequential(layers)

        self.optimizer = self.get_optimizer(self.model.parameters(), learning_rate=learning_rate,
                                            weight_decay=weight_decay)

        #@todo: add scaling by target magnitude to increase preformance for high values (Albrecht's idea)
        self.loss_fn = F.mse_loss

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Tensor:
        self.train()

        loss_sum = torch.zeros(1)
        loss_counter = 0

        for x, y in tqdm(train_loader, leave=False, desc='training step'):
            self.optimizer.zero_grad()

            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            loss.backward()

            self.optimizer.step()

            loss_sum += loss.item()
            loss_counter += 1

        return (loss_sum / loss_counter).item()

    @torch.no_grad()
    def eval_epoch(self, valid_loader: torch.utils.data.DataLoader) -> Tensor:
        self.eval()

        loss_sum = torch.zeros(1)
        loss_counter = 0

        for x, y in tqdm(valid_loader, leave=False, desc='validation step'):
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)

        loss_sum += loss.item()
        loss_counter += 1

        return (loss_sum / loss_counter).item()

    def tune(self, num_epochs: int, train_loader: torch.utils.data.DataLoader,
             valid_loader: torch.utils.data.DataLoader) -> None:
        self.train_losses = np.zeros(num_epochs)
        self.valid_losses = np.zeros(num_epochs)

        train_loss = -1
        valid_loss = -1
        pbar = trange(num_epochs)
        for epoch in pbar:
            pbar.set_description(f'train loss: {train_loss}, valid loss: {valid_loss}')

            train_loss = self.train_epoch(train_loader=train_loader)
            valid_loss = self.eval_epoch(valid_loader=valid_loader)

            self.train_losses[epoch] = train_loss
            self.valid_losses[epoch] = valid_loss

    def get_optimizer(self, params: torch.ParameterDict, learning_rate: float = 0.001,
                      weight_decay: float = 0.0) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


# ## We can iterate the data in batches of size 50

# In[85]:

def plot_wss(model, data_loader):
    data, label = next(iter(data_loader))

    data = data[0]
    m, n = data.shape

    mean_data = torch.mean(data, axis = 0)
    data_new = torch.stack([
                    torch.ones(m) * mean_data[0],
                    torch.ones(m) * mean_data[1],
                    data[:,2],
                    torch.ones(m) * mean_data[3]], axis = -1)

    print(data_new.shape)

    model.eval()
    y = model.forward(data_new).detach().numpy()

    print(y.shape)
    print(data[:,2].shape)

    plt.figure()
    plt.scatter(data[:,2], y)
    plt.xlabel('vpd')
    plt.ylabel('GPP')
    plt.show()
    plt.close()

if __name__ == '__main__':
    train_loader, valid_loader = load_data()

    model = FCN(num_feature=4, learning_rate=0.1, activation='relu', output_activation='identity')

    num_epochs = 100
    hybrid_model = HybridModel(True)
    hybrid_model.train()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(hybrid_model.parameters(), lr = 0.003)
    sheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, verbose = True)

    train_losses = np.zeros(num_epochs)
    valid_losses = np.zeros(num_epochs)

    train_loss = -1
    valid_loss = -1

    for epoch in range(num_epochs):
        loss_sum = torch.zeros(1)
        loss_counter = 0
        for x, y in tqdm(train_loader, leave=False, desc='training step'):
            optimizer.zero_grad()

            y_hat = hybrid_model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()

            optimizer.step()

            loss_sum += loss.item()
            loss_counter += 1

        sheduler.step()

        train_loss = (loss_sum / loss_counter).item()
        hybrid_model.eval()
        loss_sum = torch.zeros(1)
        loss_counter = 0

        for x, y in tqdm(valid_loader, desc='validation step'):
            y_hat = hybrid_model(x[0])
            loss = loss_fn(y_hat, y[0])

        loss_sum += loss.item()
        loss_counter += 1

        valid_loss = (loss_sum / loss_counter).item()
        print(valid_loss)
                
        train_losses[epoch] = train_loss
        valid_losses[epoch] = valid_loss

    print('Train Losses: ', train_losses)
    print('Test Losses: ', valid_losses)


    plt.figure()
    plt.plot(train_losses, label='training', color='tab:orange')
    plt.plot(valid_losses, label='validation', color='tab:red')
    plt.xlabel('epoch')
    plt.ylabel('mse loss')
    plt.legend()
    plt.show()

    plot_wss(hybrid_model, valid_loader)

    model.tune(num_epochs=num_epochs, train_loader=train_loader, valid_loader=valid_loader)

    # In[143]:

    plt.figure()
    plt.plot(model.train_losses, label='training', color='tab:orange')
    plt.plot(model.valid_losses, label='validation', color='tab:red')
    plt.xlabel('epoch')
    plt.ylabel('mse loss')
    plt.legend()
    plt.show()

    plot_wss(hybrid_model, valid_loader)
