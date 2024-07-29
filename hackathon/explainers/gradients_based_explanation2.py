from typing import Callable, Any, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hackathon import Ensemble
from hackathon.base_explanation import BaseExplainer
from hackathon.data_pipeline import TSData


def zeros_baseline(batch: dict[str, Any]) -> torch.Tensor:
    return torch.zeros_like(batch['x'])


def randn_baseline(std: float = 1) -> Callable[[dict[str, Any]], torch.Tensor]:
    def fn(batch: dict[str, Any]) -> torch.Tensor:
        return torch.randn_like(batch['x']) * torch.tensor(std)

    return fn


class IntegratedGradientsExplainer(BaseExplainer):

    def __init__(self, n_step: int = 20,
                 baseline_fn: Callable[[dict[str, Any]], Optional[torch.Tensor]] = zeros_baseline,
                 pbar_loops: bool = False,
                 n_sensitivity_days: int = 30,
                 co2_idx: int = 7):
        super(IntegratedGradientsExplainer, self).__init__()
        assert n_step >= 0, 'n_step must be positive or zero.'

        self.baseline_fn = baseline_fn
        self.n_step = n_step
        self.bar_loops = pbar_loops
        self.n_sensitivity_days = n_sensitivity_days
        self.co2_idx = co2_idx

        if pbar_loops:
            from tqdm import tqdm
            self._loop_wrapper = tqdm
        else:
            self._loop_wrapper = lambda it, **kwargs: it

    def _custom_explanations(self, model: Ensemble,
                             val_dataloader: DataLoader) -> Tuple[Tuple[float], float, xr.Dataset]:

        assert isinstance(val_dataloader.dataset, TSData), 'Dataset in DataLoader is of wrong type.'
        dataset: TSData = val_dataloader.dataset

        with torch.enable_grad():
            self._compute_grad_for_ds(model, val_dataloader, dataset)

        # sensitivities shape: (cxt_len, time, batch, feats)
        sensitivities = dataset.sensitivities['GPP_sens'].values

        var_prob = tuple(1 - np.nanmean(np.abs(sensitivities), axis=(0, 1, 2)))
        gpp_sens = float(np.nanmean(sensitivities[..., self.co2_idx]))

        return var_prob, gpp_sens, dataset.sensitivities

    def _compute_grad_for_ds(self, model: Ensemble,
                             dataloader: DataLoader,
                             dataset: TSData) -> Tuple[np.ndarray, np.ndarray]:
        grads_dataset = list()
        inp_x_grads_dataset = list()
        ig_dataset= list()

        for batch in self._loop_wrapper(dataloader, desc='dataloader'):
            batch = self.batch_to_device(batch)
            grads_ig_batch, inp_x_grads_batch,grads_batch, y_hat_batch = self._compute_ig_for_batch(model, batch)
            print(grads_ig_batch)

            if isinstance(self, InputXGradExplainer):
                sensitivities = inp_x_grads_batch
            elif isinstance(self, GradExplainer):
                sensitivities = grads_batch
            else:
                sensitivities = grads_ig_batch  # default case if we're using IG explainer

            print("sensitivities.shape")
            print(sensitivities.shape)
            time_pad = batch['x'].shape[1] - sensitivities.shape[2]
            pad_width = ((0, 0),  # batch dim
                         (0, 0),  # contex dim. 4yrs
                         (time_pad, 0),  # time dim
                         (0, 0))  # features dim
            sensitivities = np.pad(sensitivities, pad_width, constant_values=np.nan)

            # Assign sensitivities to xr.Dataset.
            dataset.assign_sensitivities(
                sensitivities=torch.from_numpy(sensitivities),
                data_sel=batch['data_sel']
            )

            grads_dataset.append(grads_batch)
            inp_x_grads_dataset.append(inp_x_grads_batch)
            ig_dataset.append(grads_ig_batch)


        grads_dataset = np.array(grads_dataset)
        inp_x_grads_dataset = np.array(inp_x_grads_dataset)
        ig_dataset = np.array(ig_dataset)

        return grads_dataset, inp_x_grads_dataset,ig_dataset

    def _compute_ig_for_batch(self, model: Ensemble, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        ig_res_batch = list()
        ig_x_inp_res_batch = list()
        grads_res_batch = list()
        y_hat_list_batch = list()

        batch_iter = zip(batch['x'].unsqueeze(1),
                     batch['y'].unsqueeze(1),
                     batch['data_sel']['pred_len'])

        for xi, yi, pred_len in batch_iter:
            grads, g_x_inp, y_hat = self._compute_grads_foreach_batch_entry(model, pred_len, xi, yi)

        # Accumulate gradients across steps using np.cumsum
            ig = np.cumsum(grads, axis=0)

            ig_res_batch.append(ig)  # Take the final integrated gradient
            ig_x_inp_res_batch.append(g_x_inp)
            grads_res_batch.append(grads)  # Append the gradients
            y_hat_list_batch.append(y_hat)

        batch_ig = np.concatenate(ig_res_batch)
        ig_x_inp_res_batch = np.concatenate(ig_x_inp_res_batch)
        grads_res_batch = np.concatenate(grads_res_batch)  # Concatenate the gradients
        y_hat_list_batch = torch.stack(y_hat_list_batch)
        return batch_ig, ig_x_inp_res_batch, grads_res_batch, y_hat_list_batch

    def _compute_grads_foreach_batch_entry(self, model: Ensemble,
                                           pred_len: torch.Tensor,
                                           x: torch.Tensor,
                                           y: torch.Tensor):
        all_grads = list()
        y_hat = None

        batch_i = dict(x=x, y=y, data_sel=dict(pred_len=pred_len.unsqueeze(0)))

        baseline = self.baseline_fn(batch_i)
        assert (baseline is None and self.n_step == 0) or (baseline is not None and self.n_step > 0)

        if baseline is None:
            grads, y_hat, x_norm = self._grad_step_foreach_ig_step(x, batch_i, model)

            inp = x_norm
        else:
            for ig_step in self._loop_wrapper(range(0, self.n_step + 1), leave=False, desc='IG loop'):
                grads, y_hat, x_norm = self._grad_step_foreach_ig_step(x, batch_i, model,
                                                                       ig_step=ig_step,
                                                                       baseline=baseline)
                all_grads.append(grads)
            grads = np.array(all_grads).mean(0)

            inp = (x - baseline).detach().cpu()

        grads = np.transpose(grads, (1, 2, 0, 3))

        # remove on since it's nowcasting, i.e. the pred time is also considered
        min_x_time = pred_len + self.n_sensitivity_days - 1

        inp = F.pad(inp, pad=(0, 0, min_x_time - inp.shape[1], 0), value=torch.nan)[:, -min_x_time:, :]

        inp = inp.unfold(1, self.n_sensitivity_days, 1).numpy()
        inp = np.transpose(inp, (0, 3, 1, 2))

        # note that `grads` is either ig or just grads
        g_x_inp = grads * inp

        assert y_hat is not None, 'y_hat_wo_baseline must be assigned.. Bug?'

        y_hat = torch.stack(y_hat).mean(0).detach().cpu()[:, -pred_len:, :]
        return grads, g_x_inp, y_hat[:, -pred_len:, :]

    def _grad_step_foreach_ig_step(self, x: torch.Tensor,
                                   batch_curr: dict,
                                   model: Ensemble,
                                   ig_step: Optional[int] = None,
                                   baseline: Optional[torch.Tensor] = None):
        pred_len = batch_curr['data_sel']['pred_len'][0]

        if baseline is None:
            batch_curr['x'] = x
        else:
            beta = float(ig_step) / self.n_step
            x_scaled = baseline + beta * (x - baseline)
            batch_curr['x'] = x_scaled

        batch_curr['x'].requires_grad = True

        # normalization is performed in the model
        y_hat, x_norm = model(batch_curr, return_members_io=True)

        grad_list = list()

        for pred_off in self._loop_wrapper(range(pred_len), leave=False, desc='time_loop'):
            pred_idx = (pred_off - pred_len) % batch_curr['y'].shape[-2]
            grads = self._grad_step_foreach_output_timestep(model, pred_idx, x_norm, y_hat)
            grad_list.append(grads)
        grads = np.array(grad_list)
        return grads, y_hat, torch.stack(x_norm).mean(0).detach().cpu()

    def _grad_step_foreach_output_timestep(self, model, pred_idx, x_norm, y_hat):
        num_ensembles = len(x_norm)

        grads_per_ensemble = list()
        for ensemble_idx in range(num_ensembles):
            model.zero_grad()
            grads = torch.autograd.grad(outputs=y_hat[ensemble_idx][0, pred_idx, 0],
                                        inputs=x_norm[ensemble_idx],
                                        retain_graph=True,
                                        allow_unused=False)[0]
            start_idx = max(0, pred_idx - self.n_sensitivity_days + 1)
            end_idx = pred_idx + 1

            grads = grads[:, start_idx:end_idx].detach().cpu()
            grads_per_ensemble.append(grads.numpy())

        grads = np.stack(grads_per_ensemble).mean(0)

        time_pad = self.n_sensitivity_days - grads.shape[1]
        pad = ((0, 0),  # batch (no padding)
               (time_pad, 0),  # time (left padding)
               (0, 0))  # features (no padding)
        grads = np.pad(grads, pad_width=pad, constant_values=torch.nan)
        return grads


class _BaseGradExplainer(IntegratedGradientsExplainer):
    def __init__(self, **kwargs):
        assert len({'baseline_fn', 'n_step'} - set(kwargs.keys())) == 2, ('baseline_fn and n_step can''t be set for InpXGradExplainer.')
        super(_BaseGradExplainer, self).__init__(n_step=0, baseline_fn=lambda batch: None, **kwargs)


class InputXGradExplainer(_BaseGradExplainer):
    pass


class GradExplainer(_BaseGradExplainer):
    pass
