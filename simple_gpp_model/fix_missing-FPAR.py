#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:23:55 2022
Title: Fix missing FPAR values
@author: awinkler
"""

import xarray as xr

# for CMIP6
simulation = 'historical' #'ssp585'
infile = 'data/CMIP6/predictor-variables_'+simulation

ds = xr.open_dataset(infile+'.nc')

## get FPAR from other surround pixels and use mean
ds['FPAR'].loc[{'location':11}] = ds['FPAR'].sel(location=[7,12,13]).mean(dim='location').values
ds['FPAR'].loc[{'location':14}] = ds['FPAR'].sel(location=[9,12,13]).mean(dim='location').values
ds['FPAR'].loc[{'location':15}] = ds['FPAR'].sel(location=[10,12,13]).mean(dim='location').values

## save FPAR
ds.to_netcdf(infile+'_new.nc')
