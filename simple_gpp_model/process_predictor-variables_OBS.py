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

## define locations
locations = [
             (-92.1258, 31.6953), # Jena-USA
             (11.5892, 50.9271), # Jena-GER
             ]

## ERA5 Met
def obtain_ts(path, var, lon, lat):
    ds = xr.open_mfdataset(path+'/'+var+'/*.nc')
    ds = ds.sel(time=slice("1982","2016"))
    ts = ds.sel(latitude=lat, longitude=lon, method='nearest')[var].to_dataframe()
    return ts

## data path
ERA5 = "/Net/Groups/data_BGC/era5/e1/0d25_daily/" # in MPI-BGC network

## loop over locations
for i in locations:

    lon, lat = i

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
    predictors = pd.concat([ssrd, t2mmin['t2mmin'], co2['co2'], e['e'], tp['tp'], vpd['vpd'], swvl1['sSWC'], fpar['FPAR']], axis=1)
    predictors.index = pd.date_range(start='1982-01-01', periods=len(predictors), freq='1d')

    ## harmonize units
    predictors['vpd'] = predictors['vpd'] * 100 # hPa to Pa
    predictors['t2mmin'] = predictors['t2mmin'] - 273.15 # K to Celsius

## Combine and save to disk
#predictors = predictors.dropna()
#predictors.to_csv('data/OBS/predictor-variables_'+location+'.csv')
