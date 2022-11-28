
import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader

from hackathon.model_runner import Ensemble
from hackathon.base_explanation import BaseExplainer
from hackathon.data_pipeline import TSData

class TestExplainer(BaseExplainer):

    def _custom_explanations(
            self,
            model: Ensemble,
            val_dataloader: DataLoader) -> tuple[tuple[float], float, xr.Dataset]:

        dataset: TSData = val_dataloader.dataset
        for batch in val_dataloader:
            # Leave this!
            batch = self.batch_to_device(batch)

            # Use `return_quantiles=False`:
            # batch_size (1) x seq_len x num_targets (1)
            y_hat = model(batch, return_quantiles=False)

            # Calculate sensitivities. Result must be of shape.
            # Create dummy data of shape:
            # batch_size (1) x context_size (1461) x seq_len x num_features (8)
            sensitivities = torch.randn(batch['y'].shape[0], 1461, batch['y'].shape[1], 8)

            # Assign sensitivities to xr.Dataset.
            dataset.assign_sensitivities(
                sensitivities=sensitivities,
                data_sel=batch['data_sel']
            )

        # Dummy data.
        var_prob = torch.softmax(torch.randn(8), 0).numpy()
        co2_sens = np.random.uniform()

        return (
            var_prob,
            co2_sens,
            dataset.sensitivities
        )
