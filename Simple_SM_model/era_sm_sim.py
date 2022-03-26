"""
Simulate soil moisture from ERA5 0.25 degree time series at 40N 5.8W.

@author: bkraft
"""


import xarray as xr
import numpy as np
from glob import glob
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from datetime import datetime


def extract_data(target_file: str = './data/era5_40N6W.zarr'):
    paths = []
    for year in range(1980, 2021):
        for var in ['tp', 't2m', 'ssr']:
            paths.extend(
                glob(
                    f'/Net/Groups/data_BGC/era5/e1/0d25_daily/{var}/{var}.daily.*.era5.1440.720.{year}.nc'
                )
            )

    era = xr.open_mfdataset(
        paths,
        chunks={'time': 100, 'latitude': 10, 'longitude': 10}
    ).sel(latitude=40, longitude=-5.8, method='nearest').drop(['latitude', 'longitude']).load()

    era['et'] = 0.7 + 0.289 * era['ssr'].clip(0, None) + 0.023 * (era['t2m'].clip(0, None) - 273.15)
    era['sm'] = xr.zeros_like(era['tp']) * np.nan
    era['sm'][0] = 50.0

    for t in tqdm(range(1, len(era.time)), desc='simulating SM'):
        era['sm'][t] = (era['sm'][t - 1] + era['tp'][t] - era['et'][t]).clip(0, None)

    era = era[['tp', 'sm', 'et']]

    encoding = {}
    for var in ['tp', 'sm', 'et']:
        encoding.update({var: {'chunks': (1000,)}})

    with ProgressBar():
        era.to_zarr(target_file, encoding=encoding, mode='w')


if __name__ == '__main__':
    printtime = lambda x: x.strftime("%m/%d/%Y-%H:%M:%S")
    start = datetime.now()
    print(f'>> {printtime(start)} Start processing data.')
    extract_data()
    end = datetime.now()
    print(f'>> {printtime(end)} Done [in {end-start}].')
