#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:42:33 2022
Title: Simple GPP model using the MODIS algo
Source: https://www.ntsg.umt.edu/project/modis/user-guides/mod17c61usersguidev11mar112021.pdf
@author: awinkler
"""
#%%
## import modules
import os
import pickle
from random import shuffle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

#%% define MOD17 parameters

# Parameters for MF (Mixed Forest)
LCT = 'MF' # land cover type
epsilon_max = 0.001051 # KgC/m2/d/MJ max light-use efficiency
Tmin_min = -7 # K
Tmin_max = 9.50 # K
VPD_min = 650 # Pa
VPD_max = 2400 # Pa; orig: 2400
bSWC_size = 200 # bucket size in mm for the upper 1m soil depth
SWC_min = 25 # mm; old: 0.1 #, m3 m-3; 25 mm ala Markus ?
SWC_max = 100 # mm; old: 0.25 # m3 m-3; 100 mm ala Markus ?
b = 0.383 # Power-Law 

#%% define input variables
kind = 'CMIP6' # 'CMIP6' or 'OBS'

# for CMIP6
simulation = 'ssp585' # 'historical'
ESM = 'MPI-ESM1-2-LR'

if kind == 'CMIP6':
    infile = 'data/'+kind+'/predictor-variables_'+simulation+''
    outfile = 'data/'+kind+'/predictor-variables_'+simulation+'+'+'GPP'
    
else:
    infile = 'data/'+kind+'/predictor-variables'
    outfile = 'data/'+kind+'/predictor-variables+GPP'
    
#%% MOD17 functions
#%% VPD scalar
def calc_f_VPD(VPD, VPD_min=VPD_min, VPD_max=VPD_max):
    
    m = -1 / (VPD_max - VPD_min)
    t = 1 - m * VPD_min

    VPD_scalar = m * VPD + t

    VPD_scalar = VPD_scalar.clip(0,1)

    return VPD_scalar

#%% Tmin scalar
def calc_f_Tmin(Tmin, Tmin_min=Tmin_min, Tmin_max=Tmin_max):

    m = 1 / (Tmin_max - Tmin_min)
    t = 1 - m * Tmin_max

    Tmin_scalar = m * Tmin + t

    Tmin_scalar = Tmin_scalar.clip(0,1)

    return Tmin_scalar

#%% SWC (soil water content) scalar
def calc_f_SWC(SWC, SWC_min=SWC_min, SWC_max=SWC_max, b=b):

    REW = (SWC - SWC_min) / (SWC_max - SWC_min) # what is REW?

    SWC_scalar = np.power(REW, b)

    SWC_scalar = SWC_scalar.clip(0,1)

    return SWC_scalar

#%% Retrieve APAR
def calc_APAR(SWRad, FPAR):
    IPAR = (SWRad * 0.45)
    APAR = FPAR * IPAR
    return APAR

#%% Bucket Model for Surface Water Content
def calc_SWC_bucket(p, et, S_max=200):
    
    P_minus_E = p + et # compute P minus E (et is defined with negative sign)

    ## copy data structure and set to 0
    S = P_minus_E.copy(deep=True)
    S = S * 0
    
    S_old = S.isel(time=0) + S_max # completely fill up bucket at the beginning
    
    for i in range(len(P_minus_E['time'])):
        S_new = S_old + P_minus_E.isel(time=i) # add or remove water from the bucket
        S[i] = xr.where(S_new > S_max, S_max, S_new)
        S_old = S[i]

    return S.clip(min=0)

#%% Scale max LUE efficiency with atmospheric CO2
def calc_f_CO2(CO2):

    CO2_init = CO2[0]
    CO2_scalar = (CO2 - CO2_init) / CO2 + 1

    return CO2_scalar

#%% Run GPP model
def calc_GPP(Tmin, VPD, SWRad, FPAR, SWC, CO2):
    
    if isinstance(Tmin, xr.DataArray) or isinstance(Tmin, xr.Dataset):
        f_Tmin = xr.apply_ufunc(calc_f_Tmin, Tmin)
    else:
        f_Tmin = calc_f_Tmin(np.array(Tmin))
    
    if isinstance(VPD, xr.DataArray) or isinstance(VPD, xr.Dataset):
        f_VPD = xr.apply_ufunc(calc_f_VPD, VPD)
    else:
        f_VPD = calc_f_VPD(np.array(VPD))
    
    if isinstance(SWC, xr.DataArray) or isinstance(SWC, xr.Dataset):
        f_SWC = xr.apply_ufunc(calc_f_SWC, SWC)
        f_SWC = f_SWC.fillna(0)
    else:
        f_SWC = calc_f_SWC(np.array(SWC))
    
    if isinstance(CO2, xr.DataArray) or isinstance(CO2, xr.Dataset):
        f_CO2 = xr.apply_ufunc(calc_f_CO2, CO2)
    else:
        f_CO2 = calc_f_CO2(np.array([CO2]))
        
    APAR = calc_APAR(SWRad,FPAR)

    return epsilon_max * APAR * f_CO2 * f_Tmin * f_VPD * f_SWC * 1000 # GPP in gC m-2 day-1#%%
#%%
if __name__ == "__main__":

    #% read data
    ds = xr.open_dataset(infile+'.nc')
    
    ## make sure time is in first place in the dimension order
    ds = ds.transpose("time", ...)
    
    ## check if ET is defined negative
    if ds['e'].median() > 0:
        ds['e'] = ds['e'] * -1

    ## get predictor variables
    Tmin = ds['t2mmin']
    VPD = ds['vpd']
    SWRad = ds['ssrd']
    FPAR = ds['FPAR']
    CO2 = ds['co2']

    #% calc Soil Moisture based on Precipitation and Evapotranspiration with a surface bucket
    ds['bSWC'] = calc_SWC_bucket(p=ds['tp'], et=ds['e'], S_max=bSWC_size)
    SWC = ds['bSWC']

    #% calc GPP
    ds['GPP'] = calc_GPP(Tmin, VPD, SWRad, FPAR, SWC, CO2)
    #ds['GPP_constant-Tmin'] = calc_GPP(10, VPD, SWRad, FPAR, SWC, CO2)
    #ds['GPP_constant-SWrad'] = calc_GPP(Tmin, VPD, 15, FPAR, SWC, CO2)
    #ds['GPP_constant-VPD'] = calc_GPP(Tmin, 650, SWRad, FPAR, SWC, CO2)
    #ds['GPP_constant-FPAR'] = calc_GPP(Tmin, VPD, SWRad, 0.5, SWC, CO2)
    #ds['GPP_constant-SWC'] = calc_GPP(Tmin, VPD, SWRad, FPAR, 100, CO2)
    #ds['GPP_constant-CO2'] = calc_GPP(Tmin, VPD, SWRad, FPAR, SWC, 340) ## CO2 at 1982
    
    #%% add random noise; better: noise should scale with the signal
    ds['GPP'] = ds['GPP'] + np.abs(np.random.normal(loc=0, scale=0.5, size=ds['GPP'].shape))

    #%% make plot
    # variables = ['t2mmin', 'bSWC', 'vpd', 'ssrd', 'FPAR', 'tp', 'e', 
    #              'GPP']#, 'bSWC', 'GPP_constant-Tmin', 'GPP_constant-SWrad', 'GPP_constant-VPD',
    #              #'GPP_constant-FPAR', 'GPP_constant-SWC', 'GPP_constant-CO2']
    # df = ds.isel(time=0, cluster=0, location=0).to_dataframe().reset_index().set_index('time')

    # df[variables].plot.line(subplots=True, layout=(6,4), figsize=(14,10))
    # plt.show()

    #%% diff: last 5 year minus first 5 years
    # diff = (ds.sel(time=slice(str(2016-5),str(2016))).groupby('time.dayofyear').mean(dim='time')\
    #         - ds.sel(time=slice(str(1982),str(1982+5))).groupby('time.dayofyear').mean(dim='time'))\
    #        .sel(location=2).to_dataframe().reset_index()
    # diff[variables].plot.line(subplots=True, layout=(6,4), figsize=(14,10))
    # plt.show()
    
    #%% disguise variables
    vrs = ['t2mmin', 'vpd', 'ssrd', 'FPAR', 'tp', 'e', 'sfcWind']
    ## store mapping
    if os.path.exists('variable_mapping.pickle'):
        print('Variable mapping found.')
        with open('variable_mapping.pickle', 'rb') as handle:
            mapping = pickle.load(handle)
            
        for var in mapping.keys():
            ds = ds.rename({var:mapping[var]})
    
    else:
        
        shuffle(vrs)
    
        mapping = {}
        for i in range(len(vrs)):
            ds = ds.rename({vrs[i]:'var'+str(i+1)})
            mapping[vrs[i]] = 'var'+str(i+1)
        
        with open('variable_mapping.pickle', 'wb') as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #%%save data to disk
    #ds = ds.astype('float32')
    #vrs = ['var'+str(i+1) for i in range(len(vrs))] + ['co2', 'GPP']
    #ds[vrs].sel(cluster=slice(0,1)).sel(location=slice(1,10)).to_netcdf(outfile+'.nc') # cluster 2 is for testing

    #%% save testing data set
    ds = ds.astype('float32')
    vrs = ['var'+str(i+1) for i in range(len(vrs))] + ['co2', 'GPP']
    
    if simulation == 'historical':
        ds[vrs].sel(location=slice(11,None)).to_netcdf(outfile+'_test.nc')
    else:
        ds[vrs].to_netcdf(outfile+'_test.nc') # cluster 2 is for testing