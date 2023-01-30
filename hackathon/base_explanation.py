from abc import abstractmethod
from typing import Iterable, Optional

import torch
import xarray as xr
from torch import Tensor
from torch.utils.data import DataLoader

from hackathon.model_runner import Ensemble


class BaseExplainer(object):
    """Base explanation class, meant to be subclassed.

    Usage
    -----
    > Subclass this and at least override the `BaseExplainer._custom_explanations` method.
      You must exactly follow the signature defined here (see 'Parameters' and 'Returns'.)
    > You may first calculate the full sensitivities and derive the other return items from that.
        For sensitivities, we speak about sensitivity of a target *towards* a feature. We store the
        sensitivity of the last four years of the test data towards all time steps of the features.
        Make sure that you follow this patter; For the past 30 days calculate the sensitivity towards
        each time step. This yields a sensitivity tensor of
        - 30 x num_time x num_features for each batch element, or
        - num_batch x 30 x num_time x num_features in total.
    > The sensitivity tensor can be assigned to a xr.Dataset by using the
        `test_dataloader.dataset.assign_sensitivities(sensitivities=..., data_sel=batch['data_sel'])` method.
    > After having assigned all sensitivities, the xarray.Dataset with the stored values is accessible via
        `test_dataloader.dataset.sensitivities`.
    > Call the method `batch = batch_to_device(batch)` in your inference loop.
    > Don't call `BaseExplainer._custom_explanations`, use `BaseExplainer.get_explanations` instead.

    """

    device = None

    @abstractmethod
    def _custom_explanations(
            self,
            model: Ensemble,
            val_dataloader: DataLoader) -> tuple[tuple[float], float, xr.Dataset]:
        """Returns explanations in a predefined format. Must be overridden.

        Parameters
        ----------
        model: a tuned model, subclass of Ensemble.
        val_dataloader: a dataloader.

        Returns
        -------
        Tuple of three elements:
        - Dummy variable probability: a tuple containing the probability of each of the input
          features to be the dummy variable (which has no impact on the predictions).
        - The GPP sensitivity towards CO2, a single float.
        - The sensitivities of GPP towards all variables, a xarray.Dataset. See 'Notes' for more details.
        """

    def get_explanations(
            self,
            model: Ensemble,
            val_dataloader: DataLoader,
            gpu: Optional[int] = None) -> xr.Dataset:
        """Returns explanations.

        Parameters
        ----------
        model: a tuned model, an Ensemble.
        val_dataloader: a dataloader.
        gpu: the GPU device to train on. If `None`, inference is done on CPU.

        Returns
        -------
        An xr.Dataset, the sensitivities of GPP towards all variables with attributes
          `var_prob` and `co2_sens`

        """

        if not isinstance(model, Ensemble):
            raise TypeError(f'model is a {type(model).__name__}')

        self.device = None if gpu is None else torch.device(f'cuda:{gpu}')
        model.eval()

        if gpu is not None:
            model.to(self.device)

        explanations = self._custom_explanations(
            model=model,
            val_dataloader=val_dataloader)

        explanations = self._merge_explanation(explanations)

        self.device = None

        return explanations

    def _merge_explanation(self, ex: tuple):
        if not isinstance(ex, tuple):
            raise TypeError(
                '`explanations` returned by `_custom_explanations` must '
                f'of type `tuple, got type `{type(ex).__name__}`.'
            )

        if len(ex) != 3:
            raise ValueError(
                '`explanations` returned by `_custom_explanations` must be '
                f'a tuple of length `3`, has length `{len(ex)}`.'
            )

        if not isinstance(ex[0], Iterable):
            raise ValueError(
                '`explanations` returned by `_custom_explanations` must contain an '
                '`Iterable` at position 0 (variable probabilities), got '
                f'type `{type(ex[0]).__name__}`.'
            )

        if not len(ex[0]) == 8:
            raise ValueError(
                '`explanations` returned by `_custom_explanations` must contain an '
                '`Iterable` of size 8 at position 0 (variable probabilities), '
                f'length is `{len(ex[0])}`.'
            )

        # Try to cast to list of floats.
        ex0 = []
        for el in ex[0]:
            if isinstance(el, str):
                raise TypeError(
                    '`explanations` returned by `_custom_explanations` at position 0 '
                    '(variable probabilities) must contain an `Iterable` with '
                    'element type `float`, got elements of type `str`.'
                )
            try:
                el_f = float(el.cpu()) if isinstance(el, Tensor) else float(el)
            except Exception as e:
                raise e from TypeError(
                    'while trying to cast `explanations` position 0 (variable '
                    'probabilities) to a list of floats, an error occurred '
                    '(see Traceback).'
                )

            ex0.append(el_f)

        # Try to cast to float.
        try:
            if isinstance(ex[1], str):
                raise TypeError(
                    '`explanations` returned by `_custom_explanations` at position 1 '
                    '(co2 sensitivity) must contain a `float`, got a `str`.'
                )
            ex1 = float(ex[1])
        except Exception as e:
            raise e from TypeError(
                'while trying to cast `explanations` position 1 (co2 sensitivity) '
                'to a float, an error occurred (see Traceback).'
            )

        if not isinstance(ex[2], xr.Dataset):
            raise ValueError(
                '`explanations` returned by `_custom_explanations` must contain '
                'an xr.Dataset at position 2 (sensitivities), got '
                f'type `{type(ex[2]).__name__}`.'
            )

        ex_merged = ex[2]
        ex_merged.attrs['var_prob'] = ex0
        ex_merged.attrs['co2_sens'] = ex1

        return ex_merged

    def batch_to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:

        if self.device is not None:
            batch['x'] = batch['x'].to(self.device)
            batch['y'] = batch['y'].to(self.device)

        return batch
