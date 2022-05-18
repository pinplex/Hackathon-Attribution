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

## define location
location = 'Jena-USA' # 'next-to-Jena' 'Jena-USA'

if location == 'Jena-USA':
    ## Jena (USA) coordinates
    lat = 31.6953
    lon = -92.1258

elif location == 'next-to-Jena':
    ## next to Jena (Germany) coordinates
    lon = 11.5892 + 1.5
    lat = 50.9271 - 1.5

elif location == 'Jena':
    ## Jena (Germany) coordinates
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
swvl1 = obtain_ts(ERA5, 'swvl1')
swvl1 = swvl1.rename({'swvl1': 'sSWC'}, axis=1) # surface soil water content
vpd = obtain_ts(ERA5, 'vpd_daytime_mean')
vpd = vpd.rename({'vpd_daytime_mean': 'vpd'}, axis=1)

predictors = pd.concat([ssrd, t2mmin['t2mmin'], vpd['vpd'], swvl1['sSWC']], axis=1)
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
predictors.to_csv('data/OBS/predictor-variables_'+location+'.csv')
