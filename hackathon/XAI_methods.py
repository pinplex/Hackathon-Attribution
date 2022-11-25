import numpy as np
from torch import Tensor
import xarray as xr
import torch
from torch import Tensor
from tqdm import tqdm



def compute_saliency(batch_x, model):
    n = 2 ## can be changed

    for i in tqdm(range(1, n + 1)):
        x = batch_x
        x.requires_grad = True
        y = model(x)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=[torch.ones_like(y)])
        gradients = grad.numpy()
        heatmap = np.abs(gradients)
        argmax = np.argmax(heatmap,axis=2)
        unique, counts = np.unique(argmax, return_counts=True)
        counts = counts/3652 # validation set
        grad = counts
        g_average = np.average(heatmap,axis=1)

    return heatmap,argmax, grad, g_average



def GradxInp(batch_x, model):
    n = 2 ## can be changed

    for i in tqdm(range(1, n + 1)):
        x = batch_x
        x.requires_grad = True
        y = model(x)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=[torch.ones_like(y)])
        gradients = grad
        grad_input = gradients * x
        grad_input= grad_input.detach().numpy()
        heatmap = np.abs(grad_input)
        argmax = np.argmax(heatmap,axis=2)
        unique, counts = np.unique(argmax, return_counts=True)
        counts = counts/3652 # validation set
        grad = counts
        g_average = np.average(heatmap,axis=1)

    return heatmap,argmax, grad, g_average



