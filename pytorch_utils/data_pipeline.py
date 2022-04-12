
"""Pytorch data pipeline.

Author: bkraft@bgc-jena.mpg.de
        edited by awinkler@bgc-jena.mpg.de
"""

import torch
import xarray as xr

from numpy.typing import ArrayLike


class TSData(torch.utils.data.Dataset):
    def __init__(
            self,
            ds: xr.Dataset,
            features: list[str],
            targets: list[str],
            time_slice: slice,
            return_seq: bool = False,
            ts_window_size: int = -1,
            normalize: bool = True,
            norm_kind: str = 'mean_std',
            norm_stats: dict[str, dict[str, float]] = {},
            dtype: str = 'float32') -> None:

        """Time series dataset.

        Notes
        -----
        Defines a dataset and sampling strategy.

        For non-sequential models (`return_seq=False`):
            Returns a tuple of featurees, targets of shape (num_features,), (num_targets,)

        For sequential models (`return_seq=True`):
            Returns a tuple of featurees, targets of shape (ts_window_size, num_features,), (ts_window_size, num_targets,)

        Parameters
        ----------
        ds: xr.Dataset
            The dataset.
        features: list[str]
            A list of features.
        targets: list[str]
            A list of targets
        time_slice: slice
            A time slice defining the time range of the datasets, e.g., `slice('2001', '2018')`,
            or `slice('2001-01-01', '2018-06-20')`.
        return_seq: bool (default is `False`)
            Whether to return a time series (for sequential models) or individual time steps (for
            non-sequential models).
        ts_window_size: int (default is -1)
            The sequence window size, ignored if `return_seq=False`, else, it defines the sequence
            lengths of a samle.
        normalize: bool (default is `True`)
            Whether to normalize the data or not.
        norm_kind: str (default is `mean_std`)
            Kind of normalization technique: 'mean_std' versus 'min_max'.
        norm_stats: dict[str, xr.aAtaset],
            Normalization stats of format {'mean': xr.Dataset, 'st': xr.Dataset}. If not passed, the stats will be inferred from
            the input datasets 'ds'.
        dtype: str (default is 'float32')
            The return type of the nupy arrays.
        """

        self.features = [features] if isinstance(features, str) else features
        self.targets = [targets] if isinstance(targets, str) else targets
        self.return_seq = return_seq
        self.ts_window_size = ts_window_size
        self.dtype = dtype
        self.do_normalize = normalize
        self.norm_kind = norm_kind

        self.ds = ds.sel(time=time_slice)

        self._check_args()

        if len(norm_stats) == 0:
            self.norm_stats = self._get_norm_stats()
        else:
            self.norm_stats = norm_stats

    def __len__(self) -> int:
        """Returns the number of samples."""
        if self.return_seq:
            if self.ts_window_size == -1:
                return 1
            else:
                return len(self.ds.time) - self.ts_window_size + 1
        else:
            return len(self.ds.time)

    def __getitem__(self, idx: int) -> tuple[ArrayLike]:
        """Returns a sample.

        Return shapes
        -------------
        For non-sequential models (`return_seq=False`):
            Returns a tuple of featurees, targets of shape (num_features,), (num_targets,)

        For sequential models (`return_seq=True`):
            Returns a tuple of featurees, targets of shape:
            (ts_window_size, num_features,), (ts_window_size, num_targets,)

        Parameters
        ----------
        idx: int
            The sample index.

        Returns
        -------
        A tuple of numpy arrays for features and targets, see 'Return shapes' for details.
        """
        if self.return_seq:
            if self.ts_window_size == -1:
                if idx != 0:
                    self._raise_index_error(idx)

                sample = self.ds

            else:
                if (idx < 0) or (idx > len(self) - 1):
                    self._raise_index_error(idx)

                sample = self.ds.isel(time=slice(idx, idx + self.ts_window_size))

        else:
            if (idx < 0) or (idx > len(self) - 1):
                self._raise_index_error(idx)

            sample = self.ds.isel(time=idx)

        if self.do_normalize:
            sample = self.normalize(sample)

        features = sample[self.features]
        targets = sample[self.targets]

        return self._to_numpy(features), self._to_numpy(targets)

    def normalize(self, ds: xr.Dataset, keys: list[str] = []) -> xr.Dataset:
        if len(keys) == 0:
            keys = self.features + self.targets
                
        if self.norm_kind == 'min_max':
            return (ds[keys] - self.norm_stats['min'][keys]) / (self.norm_stats['max'][keys] - self.norm_stats['min'][keys])
        
        else: # mean_std
            return (ds[keys] - self.norm_stats['mean'][keys]) / self.norm_stats['std'][keys]

    def denormalize(self, ds: xr.Dataset, keys: list[str] = []) -> xr.Dataset:
        if len(keys) == 0:
            keys = self.features + self.targets

        return ds[keys] * self.norm_stats['std'][keys] / self.norm_stats['mean'][keys]

    def norm_np(self, x: ArrayLike, key: str):
        """Normalize a numpy array.

        Parameters
        ----------
        x: ArrayLike
            The numpy array, should only contain one varaible (`key`).
        key: str
            The name of the variable (must be present in `norm_stats`).

        Returns
        -------
        The denormalized numpy array.
        """
        return (x - self.norm_stats['mean'][key].item()) / self.norm_stats['std'][key].item()

    def denorm_np(self, x: ArrayLike, key: str):
        """Denormalize a numpy array.

        Parameters
        ----------
        x: ArrayLike
            The numpy array, should only contain one varaible (`key`).
        key: str
            The name of the variable (must be present in `norm_stats`).

        Returns
        -------
        The denormalized numpy array.
        """
        
        if self.norm_kind == 'min_max':
            return x * (self.norm_stats['max'][keys].item() - self.norm_stats['min'][keys].item()) + self.norm_stats['min'][keys].item() 
        
        else: # mean_std
            return x * self.norm_stats['std'][key].item() + self.norm_stats['mean'][key].item()

    def _get_norm_stats(self) -> xr.Dataset:
        
        if self.norm_kind == 'min_max':
            norm_stats = {
                'min': self.ds[self.features + self.targets].min().compute(),
                'max': self.ds[self.features + self.targets].max().compute(),
            }
            
        else: # mean_std
            norm_stats = {
                'mean': self.ds[self.features + self.targets].mean().compute(),
                'std': self.ds[self.features + self.targets].mean().compute(),
            }

        return norm_stats

    def _to_numpy(self, ds: xr.Dataset) -> ArrayLike:
        return ds.to_array().transpose(..., 'variable').values.astype(self.dtype)

    def _raise_index_error(self, idx: int) -> None:
        raise IndexError(
            f'index {idx} is out of bounds, must be in range [0, {len(self) - 1}]'
        )

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
                f'of time stepes ({len(self.ds.time)})'
            )
