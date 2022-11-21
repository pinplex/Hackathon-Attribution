
from abc import abstractmethod
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np

from hackathon.base_model import BaseModel
from hackathon.data_pipeline import TSData

class Explainer(object):
    """Base explanation class, meant to be subclassed."""

    @abstractmethod
    def get_explanations(
            self,
            model: BaseModel,
            test_dataloader: DataLoader) -> tuple[tuple[float], float, xr.Dataset]:
        """Returns explanations in a predefined format. Must be overridden.

        Notes
        -----
        The sensitivities represent the last four years of the target data (length of 1461) towards
        the entire time range. For each location, the dimensionality of the sensitivity tensor is
        1461 x seq_len x num_features, where seq_len is the number of days in the dataset. E.g., for a dataset
        of 8 years, the sensitivities have shape 365 * 4 + 1 = 1461 times 365 * 8 + 2 = 2922 times the number
        of features -> 1461 x 2922 x 8. The tensor can be assigned to an xarray.Dataset using the DataLoader.dataset.assign_sensitivities method. The method expects a tensor of shape:
            - (batch_size x 1461 x seq_len x num_features)
        ...along with batch['data_sel'] to assign the sensitivities to the right location. 

        Parameters
        ----------
        model: a tuned model, subclass of BaseModel.
        test_dataloader: a dataloader.

        Returns
        -------
        Tuple of three elements:
        - Dummy variable probability: a tuple containing the probability of each of the 8 input
        features to be the dummy variable (which has no impact on the predictions).
        - The GPP sensitivity towards CO2, a single float.
        - The sensitivities of GPP towards all variables, an xarray.Dataset. See 'Notes' for more details.
        """

        model.eval()
        ds: TSData = test_dataloader.dataset
        for batch_idx, batch in enumerate(test_dataloader):
            # Do explanations.
            s = np.random.normal(size=(batch['x'].shape[0], 1461, len(ds.ds.time), 8))
            # Assign explanations.
            ds.assign_sensitivities(sensitivities=s, data_sel=batch['data_sel'])

        var_prob = np.random.rand(8)
        var_prob /= var_prob.sum()

        co2_sens = np.random.rand()

        return (
            var_prob,
            co2_sens,
            ds.sensitivities,
        )
