import numpy as np
from torch import Tensor
import xarray as xr


val_loader=next(iter(datamodule.val_dataloader()))

batch_x = {key: val_loader[key] for key in val_loader.keys()  ## extarcting specific keys
       & {'x'}} 
batch_y = {key: val_loader[key] for key in val_loader.keys()  ## extarcting specific keys
       & {'y'}} 

for i in batch_x.keys():
  print(i, batch_x[i]) 

for iy in batch_y.keys():
  print(iy, batch_y[iy]) 

batch_x_bl = torch.zeros_like(batch_x[i])

def compute_saliency(batch_x, model):
    n = 2

    for i in tqdm(range(1, n + 1)):
        x = batch_x
        x.requires_grad = True
        y = model(x)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=[torch.ones_like(y)])
        gradients = grad.numpy()
        heatmap = np.abs(gradients)
        argmax = np.argmax(heatmap,axis=2)
        unique, counts = np.unique(argmax, return_counts=True)
        counts = counts/3652
        grad = counts
        g_average = np.average(heatmap,axis=1)

    return heatmap,argmax, grad, g_average

dg=compute_saliency(batch_x[i], model)
heatmap,argmax, grad, g_average =dg


def GradxInp(batch_x, model):
    n = 2

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
        counts = counts/3652
        grad = counts
        g_average = np.average(heatmap,axis=1)

    return heatmap,argmax, grad, g_average

dg=GradxInp(batch_x[i], model)
heatmap,argmax, grad, g_average =dg


