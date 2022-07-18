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

#%% some util functions
# Calculate saturation vapor pressure
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
def obtain_ts(path, var, domain, lon, lat): 
    ds = xr.open_mfdataset(glob.glob(path+'/'+domain+'/'+var+'/**/*.nc', recursive=True)) 
    ts = ds.sel(lat=lat, lon=lon, method='nearest')[var].to_dataframe() 
    ts = ts.set_index([ts.index, ts['lat'], ts['lon']]).drop(['lat', 'lon'], axis=1)
    return ts

#%% define locations with Mixed Forests
locations = [
             (129.375, 47.563), # Meixi District, Yichun, Heilongjiang, China
             (9.375, 49.429), # Central Germany
             (-75, 49.429), # "New England", USA
             ]

## add cluster: select gird-cells ~ 1.875 °N 1.875 °E
inc = 1.875
for i in range(len(locations)):
    locations.append((locations[i][0], locations[i][1]+inc))
    locations.append((locations[i][0], locations[i][1]-inc))
    locations.append((locations[i][0]+inc, locations[i][1]))
    locations.append((locations[i][0]-inc, locations[i][1]))
    
# reorder
neworder = [0,3,4,5,6,1,7,8,9,10,2,11,12,13,14]
locations = [locations[i] for i in neworder]

#%%
## on DKRZ
MIP = "ScenarioMIP" # "CMIP"
simulation = "ssp585" #"historical" 
CMIP6 = "/pool/data/CMIP6/data/"+MIP+"/MPI-M/MPI-ESM1-2-LR/"+simulation+"/r1i1p1f1/"

d = {}
## loop over locations (the loop could be removed by xarray indexing)
for i in range(len(locations)):

    lon, lat = locations[i]

    rsds = obtain_ts(CMIP6, 'rsds', 'day', lon, lat) # Surface Downwelling Shortwave Radiation
    tasmin = obtain_ts(CMIP6, 'tasmin', 'day', lon, lat)
    hfls = obtain_ts(CMIP6, 'hfls', 'day', lon, lat) # ET
    pr = obtain_ts(CMIP6, 'pr', 'day', lon, lat)
    mrso = obtain_ts(CMIP6, 'mrso', 'day', lon, lat) # Total Soil Moisture Content
    mrsos = obtain_ts(CMIP6, 'mrsos', 'day', lon, lat) # Upper Soil Moisture Content
    tas = obtain_ts(CMIP6, 'tas', 'day', lon, lat)
    hurs = obtain_ts(CMIP6, 'hurs', 'day', lon, lat) # Near-Surface Relative Humidity
    lai = obtain_ts(CMIP6, 'lai', 'Eday', lon, lat) # Leaf area index
    sfcWind = obtain_ts(CMIP6, 'sfcWind', 'day', lon, lat) # Add surface windspeed as variable which correlates with other predictors but is not used in the model 

    # harmonize units
    tas['tas'] = tas['tas'] - 273.15 # K to Celsius
    tasmin['tasmin'] = tasmin['tasmin'] - 273.15 # K to Celsius
    rsds['rsds'] = rsds['rsds'] * 60 * 60 * 24 * 1e-6 # daily average W m-2 to cumulative MJ m-2
    mrsos['mrsos'] = mrsos['mrsos'] * 1e-2 # kg m-2 in top 10cm layer to m3 m-3
    hfls['hfls'] = hfls['hfls'] / (2.45 * 1e6) * 60 * 60 * 24 # W m-2 -> kg m-2 s-1 -> mm day-1
    pr['pr'] = pr['pr'] * 60 * 60 * 24 # kg m-2 s-1 -> mm day-1
    mrsos['mrsos'] = mrsos['mrsos'] * 1e-2 # kg m-2 in top 10cm layer to m3 m-3

    # calculate VPD
    vpd = vapor_pressure_deficit(tas['tas'], hurs['hurs']) * 100 # hPa to Pa

    # calculate FAPAR
    fapar = fapar_from_lai(lai['lai'])
    
    # put all together
    predictors = pd.concat([rsds, mrso['mrso'], mrsos['mrsos'], tasmin['tasmin'], lai['lai'], pr['pr'], hfls['hfls'], sfcWind['sfcWind']], axis=1)
    predictors['vpd'] = vpd
    predictors['fapar'] = fapar
    
    ## add location / cluster identifier
    predictors['location'] = i + 1
    
    if i <= 4:
        predictors['cluster'] = 0
        
    elif (i > 4) & (i <= 9):
        predictors['cluster'] = 1
        
    elif i > 9:
        predictors['cluster'] = 2

    ## convert to xarray
    predictors = predictors.set_index([predictors.index, predictors['location'], predictors['cluster']]).drop(['location','cluster'], axis=1)
    predictors = predictors.to_xarray()
    d[i] = predictors

## Combine and save to disk
ds = xr.merge(d.values(), compat='no_conflicts')

# rename variables to ERA5 terminology
ds = ds.rename({'rsds': 'ssrd', 'mrso': 'SWC', 'mrsos': 'sSWC','tasmin': 't2mmin', 'fapar': 'FPAR', 'pr': 'tp', 'hfls': 'e'})

## add CO2
## read CO2
co2 = xr.open_dataset('data/CMIP6/greenhouse_'+simulation+'.nc')
co2 = co2['CO2'].squeeze().to_dataframe()

if simulation == 'historical':
    co2 = co2.loc[1849:2014]
elif simulation == 'ssp585':
    co2 = co2.loc[2014:2100]
    
co2.index = pd.date_range(start=str(int(co2.index[0]+1)), freq='as', periods=len(co2))
co2 = co2.asfreq('d').fillna(method='ffill')
co2 = co2.iloc[:-1]
ds['co2'] = (("time"), co2['CO2'])

## save to file
ds.to_netcdf('data/CMIP6/predictor-variables_'+simulation+'.nc')
#to_zarr('data/OBS/predictor-variables.zarr')

