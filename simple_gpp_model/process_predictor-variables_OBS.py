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

## define locations with Mixed Forests
locations = [
             (117.602190, 32.198857), # Zhejiang, China
             (11.5892, 50.9271), # Thüringen, Germany
             (-72.987368, 42.138968), # New England, USA
             ]

## add cluster: select gird-cells ~ 1°N 1°E away
for i in range(len(locations)):
    locations.append((locations[i][0]+1, locations[i][1]+1))
    locations.append((locations[i][0]-1, locations[i][1]+1))
    locations.append((locations[i][0]+1, locations[i][1]-1))
    locations.append((locations[i][0]-1, locations[i][1]-1))

## ERA5 Met
def obtain_ts(path, var, lon, lat):
    ds = xr.open_mfdataset(path+'/'+var+'/*.nc')
    ds = ds.sel(time=slice("1982","2016"))
    ts = ds.sel(latitude=lat, longitude=lon, method='nearest')[var].to_dataframe()
    return ts

## data path
ERA5 = "/Net/Groups/data_BGC/era5/e1/0d25_daily/" # in MPI-BGC network

d = {}
## loop over locations
for i in range(len(locations)):

    lon, lat = locations[i]

    ## retrieve MET predictors
    ssrd = obtain_ts(ERA5, 'ssrd', lon, lat)
    t2mmin = obtain_ts(ERA5, 't2mmin', lon, lat)
    e = obtain_ts(ERA5, 'e', lon, lat)
    tp = obtain_ts(ERA5, 'tp', lon, lat)
    swvl1 = obtain_ts(ERA5, 'swvl1', lon, lat)
    swvl1 = swvl1.rename({'swvl1': 'sSWC'}, axis=1) # surface soil water content
    vpd = obtain_ts(ERA5, 'vpd_daytime_mean', lon, lat)
    vpd = vpd.rename({'vpd_daytime_mean': 'vpd'}, axis=1)

    ## read CO2
    co2 = pd.read_csv('data/OBS/co2_annmean_gl.txt', comment='#', names=['year', 'co2', 'uncertainty'], delim_whitespace=True)
    co2.index = pd.date_range(start=str(co2['year'].iloc[0]), freq='as', periods=len(co2))
    co2 = co2.asfreq('d').fillna(method='ffill')

    ## read FPAR
    fpar = xr.open_mfdataset('/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d083_15daily/gimms_fpar_3g/v04/Data/*.nc')
    fpar = fpar.sel(lat=lat, lon=lon, method='nearest')['FPAR'].to_dataframe()
    fpar = fpar.asfreq("1D").interpolate('linear')
    fpar.index = pd.date_range(start='1982-01-15', periods=len(fpar), freq='1d')

    ## merge predictors
    predictors = pd.concat([ssrd, t2mmin['t2mmin'], e['e'], tp['tp'], vpd['vpd'], swvl1['sSWC']], axis=1)
    predictors.index = pd.date_range(start='1982-01-01', periods=len(predictors), freq='1d')
    predictors = pd.concat([predictors, co2['co2'], fpar['FPAR']], axis=1)
    predictors = predictors.dropna()

    ## convert to xarray
    predictors = predictors.set_index([predictors.index, predictors['longitude'], predictors['latitude']]).drop(['latitude','longitude'], axis=1)
    predictors = predictors.to_xarray()
    predictors = predictors.rename_dims({'level_0': 'time'}).rename_vars({'level_0': 'time'})
    d[i] = predictors

## Combine and save to disk
#predictors = predictors.dropna()
#predictors.to_csv('data/OBS/predictor-variables_'+location+'.csv')
ds = xr.merge(d.values(), compat='no_conflicts')
ds.to_zarr('data/OBS/predictor-variables.csv')
