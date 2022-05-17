#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 
Title: Simple script to process EAR5 and AHVRR Fpar to times series
@author: awinkler
"""

## import modules
import glob
import xarray as xr
import pandas as pd
import numpy as np

## some util functions
#%% Calculate saturation vapor pressure
def vapor_saturated(t2m):
    # L is the latent heat of vaporization, 2.5×10^6 J kg−1
    # Rv is the gas constant for water vapor 461 J K−1 kg−1
    # t2m is 2m air temp in Celsius

    L = 2.5e6
    Rv = 461
    es = 6.11 * np.exp((L / Rv) * (1 / 273 - 1 / (273 + t2m)))

    return es # in hPa

#%% Calculate Vapor pressure deficit
def vapor_pressure_deficit(t2m, rh):
    # t2m is 2m air temp in Celsius
    # rh is relative humidity in %

    ## calculate saturation vapor pressure
    es = vapor_saturated(t2m)

    ## calculate vapor pressure deficit
    vpd = ((100 - rh) / 100) * es

    return vpd  # in hPa 

#%% Calculate fAPAR from LAI
def fapar_from_lai(lai, k=0.54):
    # lai is leaf area index
    # k is the light extinction coefficient; 
    # 0.5 for conifers, 0.58 for broadleaf (Jung et al., BG, 2007)
    return 1 - np.exp(-k * lai)

#%% obtain time-series for specific variables 
def obtain_ts(path, var, domain): 
    ds = xr.open_mfdataset(glob.glob(path+'/'+domain+'/'+var+'/**/*.nc', recursive=True)) 
    ts = ds.sel(lat=lat, lon=lon, method='nearest')[var].to_dataframe() 
    return ts

## Jena (Germany) coordinates
lon = 11.5892
lat = 50.9271

## on DKRZ
MIP = "CMIP" # ScenarioMIP
simulation = "historical" # ssp585 
CMIP6 = "/pool/data/CMIP6/data/"+MIP+"/MPI-M/MPI-ESM1-2-LR/"+simulation+"/r1i1p1f1/"

rsds = obtain_ts(CMIP6, 'rsds', 'day') # Surface Downwelling Shortwave Radiation
tasmin = obtain_ts(CMIP6, 'tasmin', 'day')
mrso = obtain_ts(CMIP6, 'mrso', 'day') # Total Soil Moisture Content
mrsos = obtain_ts(CMIP6, 'mrsos', 'day') # Upper Soil Moisture Content
tas = obtain_ts(CMIP6, 'tas', 'day')
hurs = obtain_ts(CMIP6, 'hurs', 'day') # Near-Surface Relative Humidity
lai = obtain_ts(CMIP6, 'lai', 'Eday') # Leaf area index

# harmonize units
tas['tas'] = tas['tas'] - 273.15 # K to Celsius
tasmin['tasmin'] = tasmin['tasmin'] - 273.15 # K to Celsius
rsds['rsds'] = rsds['rsds'] * 60 * 60 * 24 * 1e-6 # daily average W m-2 to cumulative MJ m-2
mrsos['mrsos'] = mrsos['mrsos'] * 1e-2 # kg m-2 in top 10cm layer to m3 m-3

# calculate VPD
vpd = vapor_pressure_deficit(tas['tas'], hurs['hurs']) * 100 # hPa to Pa

# calculate FAPAR
fapar = fapar_from_lai(lai['lai'])

# put all together
predictors = pd.concat([rsds, mrso['mrso'], mrsos['mrsos'], tasmin['tasmin'], lai['lai']], axis=1)
predictors['vpd'] = vpd
predictors['fapar'] = fapar

# rename variables to ERA5 terminology
predictors = predictors.rename({'rsds': 'ssrd', 'mrso': 'SWC', 'mrsos': 'sSWC', 
                                'tasmin': 't2mmin', 'fapar': 'FPAR'}, axis=1)

# store to disk
predictors.to_csv('data/predictor-variables_Jena_MPI-ESM1-2-LR_'+simulation+'.csv')

