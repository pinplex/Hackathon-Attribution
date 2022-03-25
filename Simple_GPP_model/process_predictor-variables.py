#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:07:53 2022
Title: Simple script to process EAR5 and AHVRR Fpar to times series
@author: awinkler
"""

## import modules
import xarray as xr
import pandas as pd

## Jena coordinates
lon = 11.5892
lat = 50.9271

## ERA5 Met
def obtain_ts(path, var): 
    ds = xr.open_mfdataset(path+'/'+var+'/*.nc') 
    ds = ds.sel(time=slice("1982","2016"))
    ts = ds.sel(latitude=lat, longitude=lon, method='nearest')[var].to_dataframe() 
    return ts

ERA5 = "/Net/Groups/data_BGC/era5/e1/0d25_daily/"

ssrd = obtain_ts(ERA5, 'ssrd')
t2mmin = obtain_ts(ERA5, 't2mmin')
vpd = obtain_ts(ERA5, 'vpd')

predictors = pd.concat([ssrd, t2mmin['t2mmin'], vpd['vpd']], axis=1)
predictors.index = pd.date_range(start='1982-01-01', periods=len(predictors), freq='1d')

# harmonize units
predictors['vpd'] = predictors['vpd'] * 100 # hPa to Pa
predictors['t2mmin'] = predictors['t2mmin'] - 273.15 # K to Celsius

## FPAR
fpar = xr.open_mfdataset('/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d083_15daily/gimms_fpar_3g/v04/Data/*.nc')
fpar = fpar.sel(lat=lat, lon=lon, method='nearest')['FPAR'].to_dataframe()
fpar = fpar.asfreq("1D").interpolate('linear')
fpar.index = pd.date_range(start='1982-01-15', periods=len(fpar), freq='1d')

## Combine and save to disk
predictors = pd.concat([predictors, fpar['FPAR']], axis=1)
predictors.to_csv('data/predictor-variables_Jena.csv')
