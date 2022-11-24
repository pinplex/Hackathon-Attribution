
from abc import abstractmethod
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np

from hackathon.base_model import BaseModel
from hackathon.data_pipeline import TSData

class BaseExplainer(object):
    """Base explanation class, meant to be subclassed.

    Usage
    -----
    > Subclass this and at least override the `Explainer._custom_explanations` method. You must exactly
      follow the signature defined here (see 'Parameters' and 'Returns'.)
    > You may first calculate the full sensitivities and derive the other return items from that.
        For sensitivities, we speak about sensitivity of a target *towards* a feature. We store the
        sensitivity of the last four years of the test data towards all time steps of the features.
        Make sure that you follow this patter; For each of the last four years days (= 1461 values),
        calculate the sensitivity towards each time step. This yields a sensitivity tensor of
        - 1461 x num_time x num_features for each batch element, or
        - num_batch x 1461 x num_time x num_features in total.
    > The sensitivitiy tensor can be assigned to an xr.Dataset by using the
        `test_dataloader.dataset.assign_sensitivities(sensitivities=..., data_sel=batch['data_sel'])` method.
    > After having assigned all sensitivities, the xarray.Dataset with the stored valeus is accessible via
        `test_dataloader.dataset.sensitivities`.
    > Don't call `_custom_explanations`, use `get_explanations` instead.

    """

    @abstractmethod
    def _custom_explanations(
            self,
            model: BaseModel,
            test_dataloader: DataLoader) -> tuple[tuple[float], float, xr.Dataset]:
        """Returns explanations in a predefined format. Must be overridden.

        Parameters
        ----------
        model: a tuned model, subclass of BaseModel.
        test_dataloader: a dataloader.

        Returns
        -------
        Tuple of three elements:
        - Dummy variable probability: a tuple containing the probability of each of the input
          features to be the dummy variable (which has no impact on the predictions).
        - The GPP sensitivity towards CO2, a single float.
        - The sensitivities of GPP towards all variables, an xarray.Dataset. See 'Notes' for more details.
        """

        ds: TSData = test_dataloader.dataset
        for batch_idx, batch in enumerate(test_dataloader):
            # Do explanations.
            s = np.random.normal(size=(batch['x'].shape[0], 1461, len(ds.ds.time), ds.num_features))
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

    def get_explanations(
            self,
            model: BaseModel,
            test_dataloader: DataLoader) -> tuple[tuple[float], float, xr.Dataset]:
        """Returns explanations.

        Parameters
        ----------
        model: a tuned model, subclass of BaseModel.
        test_dataloader: a dataloader.

        Returns
        -------
        Tuple of three elements:
        - Dummy variable probability: an array or tuple containing the probability of each of the input
          features to be the dummy variable (which has no impact on the predictions).
        - The GPP sensitivity towards CO2, a single float.
        - The sensitivities of GPP towards all variables, an xarray.Dataset. See 'Usage' for more details.

        """

        model.eval()
        return self._custom_explanations(model=model, test_dataloader=test_dataloader)
