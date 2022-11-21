"""Pytorch data pipeline for spatio-temporal sampling.

Author: bkraft@bgc-jena.mpg.de
"""

from abc import abstractmethod
import itertools as it
from typing import Any, Optional

import torch
from torch import Tensor
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import xarray as xr
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from typing import Union


class TSData(Dataset):
    """Defines a dataset."""

    def __init__(
            self,
            ds: xr.Dataset,
            features: list[str],
            targets: list[str],
            ts_window_size: int = -1,
            ts_context_size: int = 1,
            add_sensitifivy_ds: bool = False,
            dtype: str = 'float32'):

        """Time series dataset.

        Notes
        -----
        Defines a dataset and sampling strategy. The time will be sampled in full year chunks of length `ts_window_size`
        years with additional temporal context of `ts_context_size` years. While the time dimension is sampled with the
        window scheme, each location is considered a separate sample. This gives num_windows x num_locations samples.
        Use `ts_window_size=-1` to not subset the time dimension (but additional context of `ts_context_size` years will
         still) be used and dropped and may be dropped from predictions.

        This is the chunking for 6 years of data with `ts_context_size=1` and `ts_window_size=2`, 'w' for warmup and 'p'
        for prediction::

            |------|
            |wpp|
             |wpp|
              |wpp|
               |wpp|

        This is the chunking for 6 years of data with `ts_context_size=1` and `ts_window_size=-1`, 'w' for warmup and
        'p' for prediction::

            |------|
            |wppppp|

        Return shape
        ------------
        Returns a tuple of features, targets of shape (ts_window_size, num_features), (ts_window_size, num_targets)

        Parameters
        ----------
        ds: xr.Dataset
            The dataset. Must have dimensions `time` and `location`, for which sampling is done.
        features: list[str]
            A list of features.
        targets: list[str]
            A list of targets
        ts_window_size: int (default is -1)
            The sequence window size in YEARS that defines the sequence lengths of a sample.
            With '-1', the full sequence is returned. The value must be >0 or -1.
        ts_context_size: int (default is 1)
            The additional context length in YEARS to use at the start of the time series ("warm-up"). May vary depending on the number of days in the sequence (leap years).
        add_sensitifivy_ds: bool (default is `False`)
            If `True`, an empty dataset is added to later store the sensivivities.
        dtype: str (default is 'float32')
            The numeric data type to return.
        """
        super(TSData, self).__init__()

        self.ds = ds
        self.features = [features] if isinstance(features, str) else features
        self.targets = [targets] if isinstance(targets, str) else targets
        self.ts_window_size = ts_window_size
        self.ts_context_size = ts_context_size
        self.add_sensitifivy_ds = add_sensitifivy_ds
        self.dtype = dtype

        # Create empty sensitivity dataset
        if self.add_sensitifivy_ds:
            test_years = np.unique(self.ds.time.dt.year)
            if len(test_years) < 4:
                raise ValueError(
                    'test dataset too short to calculate sensitivities (minimum 4 years).'
                )
            time_slice = slice(str(test_years[-4]), str(test_years[-1]))
            dummy_arr = xr.DataArray(
                dims=('time', 'time_ref'),
                coords=(self.ds.sel(time=time_slice).time.values, self.ds.time.values)
            ).expand_dims(
                location=self.ds.location.values, axis=-1
            ).expand_dims(
                var=self.features, axis=-1
            ).astype('float32')
            self.sensitivities = xr.Dataset()
            for target in self.targets:
                self.sensitivities[target + '_sens'] = dummy_arr.copy()
        else:
            self.sensitivities = None

        self._check_args()

        time_coords = TSData._get_time_slices(
            ds=self.ds,
            ts_window_size=self.ts_window_size,
            ts_context_size=self.ts_context_size)
        location_coords = self.ds.location.values

        self.sample_coords = list(it.product(time_coords, location_coords))

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.sample_coords)

    def __getitem__(self, idx: int) -> dict[str, ArrayLike]:
        """Returns a sample.

        Return shapes
        -------------
        Returns a dict of features (x), targets (y), data_sel:
            x: (seq_len, num_features,)
            y: (seq_len, num_targets,)
            data_sel: {'loc': int, 'warmup_start': str, 'pred_start': str, 'pred_end': str, 'pred_end': int}

        Parameters
        ----------
        idx: int
            The sample index.

        Returns
        -------
        A tuple of numpy arrays for features and targets (see 'Return shapes' for details, and the coordinates for result assignment.
        """

        time_coord, location_coord = self.sample_coords[idx]

        ds = self.ds.sel(location=location_coord)

        ds_f = ds[self.features].sel(time=slice(time_coord['warmup_start'], time_coord['pred_end']))
        ds_t = ds[self.targets].sel(time=slice(time_coord['warmup_start'], time_coord['pred_end']))

        data_sel = {
            'loc': location_coord,
            **time_coord
        }

        return {
            'x': self._to_numpy(ds_f),
            'y': self._to_numpy(ds_t),
            'data_sel': data_sel
        }

    def assign_predictions(self, pred: Tensor, data_sel: dict[str, Any]) -> None:
        """Assign predictions to the dataset (`self.ds`), handles denormalization.

        Parameters
        ----------
        pred: Tensor
            The predictions with shape
                - (batch_size, seq_len, num_targets) or
                - (quantiles=3, batch_size, seq_len, num_targets)
        data_sel: dict[str, Any]
            The dictionary which is returned by the dataloader, containing location and time
            slices to assign the predictions to the right coordinates.

        """

        pred = pred.detach().cpu()
        if pred.shape[-1] != self.num_targets:
            raise AssertionError(
                'the last dimension of `pred` must have size equal to the number of targets.'
            )

        if pred.ndim == 3:

            time_sel_keys = ['loc', 'warmup_start', 'pred_start', 'pred_end', 'pred_len']
            if any((len(data_sel[key]) != pred.shape[0] for key in time_sel_keys)):
                raise AssertionError(
                    'first dimension size of argument `pred` must be equal to the length of each values in `data_sel`.'
                )

            for b in range(pred.shape[0]):

                sel_assign = {
                    'location': data_sel['loc'][b].cpu(),
                    'time': slice(data_sel['pred_start'][b], data_sel['pred_end'][b])
                }

                for target_i, target in enumerate(self.targets):
                    p = pred[b, -data_sel['pred_len'][b]:, target_i]

                    self.ds[target + '_hat'].loc[{**sel_assign, 'quantile': 0.5}] = p

        elif pred.ndim == 4:
            time_sel_keys = ['loc', 'warmup_start', 'pred_start', 'pred_end', 'pred_len']
            if any((len(data_sel[key]) != pred.shape[1] for key in time_sel_keys)):
                raise AssertionError(
                    'second dimension size of argument `pred` must be equal to the length of each values in `data_sel`.'
                )

            for b in range(pred.shape[1]):

                sel_assign = {
                    'location': data_sel['loc'][b].cpu(),
                    'time': slice(data_sel['pred_start'][b], data_sel['pred_end'][b])
                }

                for target_i, target in enumerate(self.targets):
                    p = pred[:, b, -data_sel['pred_len'][b]:, target_i]

                    for i, q in enumerate([0.1, 0.5, 0.9]):
                        self.ds[target + '_hat'].loc[{**sel_assign, 'quantile': q}] = p[i, ...]
        else:
            raise ValueError(
                f'predictions must have 3 or 4 dimensions, is {pred.ndim}.'
            )

    def assign_sensitivities(
            self,
            sensitivities: Union[Tensor, list[Tensor]],
            data_sel: dict[str, Any]) -> None:
        """Assign predictions to the dataset (`self.ds`), handles denormalization.

        Parameters
        ----------
        sensitivities: Tensor or list[Tensor]
            If a Tensor is passed, num_targets must be 1. Else, list elements correspond to targets.
            Each list element has shape
                - (batch_size, *1461, seq_len, num_features)
        data_sel: dict[str, Any]
            The dictionary which is returned by the dataloader, containing location and time
            slices to assign the predictions to the right coordinates.

        """

        sensitivities = self._check_sensitivities(sensitivities, data_sel)

        for target_i, target in enumerate(self.targets):
            sens = sensitivities[target_i].detach().cpu()

            for b in range(sens.shape[0]):

                sel_assign = {
                    'location': data_sel['loc'][b].cpu(),
                }

                for target_i, target in enumerate(self.targets):
                    
                    p = sens[b, :, :]

                    self.sensitivities[target + '_sens'].loc[{**sel_assign}] = p


    def _check_sensitivities(self, sensitivities: Tensor, data_sel: dict[str, Any]) -> list[Tensor]:
        if isinstance(sensitivities, Tensor):
            sensitivities = [sensitivities]
        elif isinstance(sensitivities, list):
            pass
        else:
            raise TypeError(
                f'`sensitivities` must be a Tensor or a list of Tensors, is `{type(sensitivities).__name__}`.'
            )

        for sens_i, sens in enumerate(sensitivities):
            if not isinstance(sens, Tensor):
                raise TypeError(
                    f'`sensitivities` elements must be Tensors, is `{type(sens).__name__}`.'
                )
            _, num_time, num_ref_time, num_vars =  sens.shape
            if not num_time == 1461:
                raise ValueError(
                    f'The sensitivities second dimension must have a length of 1461, is {num_time}.'
                )
            if num_time > num_ref_time:
                raise ValueError(
                    'the second dimension (time) is larger than the third dimension (ref_time). Did you mix up '
                    'the dimensions? Hint: the target\'s second variable time t`s sensitivity towards feature at '
                    'time in time_ref.'
                )
            if num_vars != self.num_features:
                raise ValueError(
                    'the last dimension of `sensitivities` must be equal to the number of features.'
                )

            time_sel_keys = ['loc', 'warmup_start', 'pred_start', 'pred_end', 'pred_len']
            if any((len(data_sel[key]) != sens.shape[0] for key in time_sel_keys)):
                raise ValueError(
                    f'first dimension size of argument `sensitivities[{sens_i}]` must be equal to the length '
                    'of each values in `data_sel`.'
                )

        return sensitivities

    @staticmethod
    def get_norm_stats(
            ds: xr.Dataset,
            features: list[str],
            dtype: str = 'float32') -> dict[str, Tensor]:

        norm_stats = {
            'mean': torch.as_tensor(ds[features].mean().to_array().values.astype(dtype)),
            'std': torch.as_tensor(ds[features].std().to_array().values.astype(dtype)),
        }

        return norm_stats

    def _to_numpy(self, ds: xr.Dataset) -> ArrayLike:
        return ds.to_array().transpose(..., 'variable').values.astype(self.dtype)

    @classmethod
    def _get_time_slices(cls, ds: xr.Dataset, ts_window_size: int, ts_context_size: int) -> list[dict[str, Any]]:
        """Get window timestamps of samples with len `ts_window_size` and additional context `ts_context_size`.

        Note that the windows cover full years only, i.e., always start at first day of the year and end at last. The
        returned list contains items of which each defines a temporal range of a sample.

        Usage
        -----
        - Select `warmup_start` to `pred_end` from features with `ds[features].sel(time=slice(warmup_start, pred_end))`.
        - Select `warmup_start` to `pred_end` from targets with `ds[targets].sel(time=slice(warmup_start, pred_end))`
        - Take `pred_len` elements from the prediction with `y_hat[:, -pred_len:, ...]` (assuming 2nd dimension of the
            predictions to be the sequence dimension) and assign them to a target xr.Dataset subset with
            `time=slice(pred_start, pred_end)`.

        Parameters
        ----------
        See `__init__`.

        Returns
        -------
        A dictionary with keys:
            - warmup_start: the moving window start including the additional temporal context (warmup).
            - pred_start: the start of the prediction, i.e., not including the warmup.
            - pred_end: the end of the prediction.
            - pred_len: the number of days from pred_start to pred_end.
        """
        start_year = min(ds.time.dt.year).item()
        end_year = max(ds.time.dt.year).item()

        if ts_window_size == -1:
            ts_window_size = end_year - start_year - ts_context_size + 1

        if (ts_context_size + ts_window_size) > (end_year - start_year + 1):
            err_msg = (f'args `ts_window_size` ({ts_window_size}) and '
                       f'`ts_context_size` ({ts_context_size}) with '
                       f'{end_year - start_year} years lead to zeros samples.')

            raise RuntimeError(err_msg)

        for year in range(start_year, end_year + 1):
            if len(ds.time.sel(time=str(year))) not in [365, 366]:
                raise ValueError(f'can only handle full years, but some days seem to be missing for the year {year}.')

        if ts_context_size < 1:
            raise ValueError(
                f'`ts_context_size` must be > 0, is {ts_context_size}.'
            )

        # Due to leap years, sequences may have different size. Here we cut the context size
        # to match the smallest possible sequence, to assert consistent sequence sizes in batches.
        full_window_size = ts_window_size + ts_context_size
        max_leap_years = full_window_size // 4
        max_seq_len = max_leap_years * 366 + (full_window_size - max_leap_years) * 365

        time_slices = list()

        for year in range(start_year, end_year - ts_context_size - ts_window_size + 2):
            # The start of the prediction period.
            pred_start = f'{year + ts_context_size}-01-01'
            # The end of the prediction period.
            pred_end = f'{year + ts_context_size + ts_window_size - 1}-12-31'
            # The start of the warmup period (is before pred_start).
            warmup_start = pd.to_datetime(pred_end) - pd.Timedelta(max_seq_len - 1, 'D')

            ts = dict(
                # Use this for selection of the sequence start.
                warmup_start=warmup_start.strftime('%Y-%m-%d'),
                # Use this for assignment of the predictions (to drop warmup).
                pred_start=pred_start,
                # Use this for selection of the sequence end and assignment of the predictions.
                pred_end=pred_end,
                # Use this to select n last predictions, matches pred_start to pred_end.
                pred_len=(pd.to_datetime(pred_end) - pd.to_datetime(pred_start)).days + 1
            )

            time_slices.append(ts)

        return time_slices

    def _check_args(self) -> None:
        missing = []

        for var in self.features + self.targets:

            if var not in self.ds.data_vars:
                missing.append(var)

        if len(missing) > 0:
            raise KeyError(f'one or more features or targets not found in the dataset `ds`: `{"`, `".join(missing)}`.')

        if len(self.ds.time) < self.ts_window_size:
            raise ValueError(
                f'`ts_window_size` ({self.ts_window_size}) must be smaller or equal to the number '
                f'of time steps ({len(self.ds.time)})'
            )

    @property
    def num_features(self) -> int:
        """Returns number of features"""
        return len(self.features)

    @property
    def num_targets(self) -> int:
        """Returns number of targets"""
        return len(self.targets)


class DataModule(pl.LightningDataModule):
    """Defines a lightning data module."""

    def __init__(
            self,
            data_path: str,
            features: list[str],
            targets: list[str],
            train_subset: dict[str, Any],
            valid_subset: dict[str, Any],
            test_subset: Optional[dict[str, Any]] = None,
            window_size: int = 10,
            context_size: int = 1,
            load_data: bool = True,
            dtype: str = 'float32',
            **dataloader_kwargs) -> None:
        """Initialize lightning data module.

        Return shapes
        -------------
        The dataloaders return a dict of features (x), targets (y), and data_sel:
            x: (batch, seq_len, num_features,)
            y: (batch, seq_len, num_targets,)
            data_sel: {
                'loc': (batch,),
                'warmup_start': (batch,),
                'pred_start': (batch,),
                'pred_end': (batch,),
                'pred_end': (batch,)
            }

        Parameters
        ----------
        data_path: str
            The data path to the NetCDF file.
        features: list[str]
            A list of features.
        targets: list[str]
            A list of targets.
        train_subset: dict[str, Any]
            The data subset that defines the training set. Keys correspond to xarray.Dataset
            dimensions, values to the selection. E.g.: `{'time':slice('2002', '2004'), 'location':[0, 1, 2]}`.
        valid_subset: dict[str, Any]
            Same as 'training_subset' for validations set.
        test_subset: dict[str, Any]
            Same as 'training_subset' for test set.
        window_size: int
            The window size in years used for training. For validation and test, the full sequence is
            used (also see `context size`).
        context_size: int (default is 1)
            The context size in years used at the start of each window to have some additional temporal
            context. Is the same for training, validation, and test set.
        load_data: bool (default is `True`)
            If 'True', data is loaded into memory.
        dtype: str (default is 'float32')
            The numeric data type to return.
        dataloader_kwargs:
            Keyword arguments passed to 'DataLoader' (e.g., batch_size) for all the sets. Note that
            the argument `shuffle` is already handled (`True` for training, `False` else).
        """
        super(DataModule, self).__init__()
        self.features = features
        self.targets = targets

        self.ds = xr.open_dataset(data_path)[self.features + self.targets]

        self.load_data = load_data

        self.train_subset = train_subset
        self.valid_subset = valid_subset
        self.test_subset = test_subset

        self.window_size = window_size
        self.context_size = context_size

        self.dtype = dtype

        self.dataloader_kwargs = dataloader_kwargs

        # Create empty target variables with naming `<target>_hat`.
        for target in self.targets:
            dummy = xr.full_like(self.ds[target], np.nan)
            dummy = dummy.expand_dims(quantile=[0.1, 0.5, 0.9])
            self.ds[target + '_hat'] = dummy

        self.ds['code'] = xr.full_like(self.ds[self.targets[0]], -1)

        training_set = self.ds.sel(**self.train_subset)
        self.norm_stats = TSData.get_norm_stats(
            ds=training_set,
            features=self.features,
            dtype=self.dtype)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return self._create_dataloader(
            self.train_subset,
            shuffle=True,
            return_full_seq=False,
            add_sensitifivy_ds=False)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return self._create_dataloader(
            self.valid_subset,
            shuffle=False,
            return_full_seq=True,
            add_sensitifivy_ds=True)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return self._create_dataloader(
            self.test_subset,
            shuffle=False,
            return_full_seq=True,
            add_sensitifivy_ds=True)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _create_dataloader(
            self,
            ds_selector: dict[str, Any],
            shuffle: bool,
            return_full_seq: bool,
            add_sensitifivy_ds: bool) -> DataLoader:
        self._assert_norm_stats()
        ds = self.ds.sel(**ds_selector)

        if self.load_data:
            ds = ds.load()

        dataset = TSData(
            ds=ds,
            features=self.features,
            targets=self.targets,
            ts_window_size=-1 if return_full_seq else self.window_size,
            ts_context_size=self.context_size,
            add_sensitifivy_ds=add_sensitifivy_ds,
            dtype=self.dtype,
        )
        return DataLoader(dataset, shuffle=shuffle, **self.dataloader_kwargs)

    def _assert_norm_stats(self):
        assert self.norm_stats, ('`norm_stats` have not been registered, '
                                 'should have been done in in the background. Bug?')

    @property
    def num_features(self) -> int:
        """Returns number of features"""
        return len(self.features)

    @property
    def num_targets(self) -> int:
        """Returns number of targets"""
        return len(self.targets)
