#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:42:33 2022
Title: Simple GPP model using the MODIS algo
Source: https://www.ntsg.umt.edu/project/modis/user-guides/mod17c61usersguidev11mar112021.pdf
@author: awinkler
"""

## import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## define model

## Parameters for DBF (Deciduous Broadleaf Forest)
#LCT = 'DBF' # land cover type
#epsilon_max = 0.001165 # KgC/m2/d/MJ max light-use efficiency
#Tmin_min = -6 # K
#Tmin_max = 9.94 # K
#VPD_min = 650 # Pa
#VPD_max = 1650 # Pa

# Parameters for MF (Mixed Forest)
LCT = 'MF' # land cover type
epsilon_max = 0.001051 # KgC/m2/d/MJ max light-use efficiency
Tmin_min = -7 # K
Tmin_max = 9.50 # K
VPD_min = 650 # Pa
VPD_max = 2400 # Pa
SWC_min = 25 # mm
SWC_max = 100 # mm
b = 0.383 # Power-Law 

#%% VPD scalar
def f_VPD(VPD, VPD_min=VPD_min, VPD_max=VPD_max):
#@todo introduce clamp instead of if statements   
    m = -1 / (VPD_max - VPD_min)
    t = 1 - m * VPD_min
    
    VPD_scalar = m * VPD + t
    
    if VPD < VPD_min:
        VPD_scalar = 1
        
    if VPD > VPD_max:
        VPD_scalar = 0
        
    return abs(round(VPD_scalar, 4))

#%% Tmin scalar
def f_Tmin(Tmin, Tmin_min=Tmin_min, Tmin_max=Tmin_max):
    
    m = 1 / (Tmin_max - Tmin_min)
    t = 1 - m * Tmin_max
    
    Tmin_scalar = m * Tmin + t
    
    if Tmin < Tmin_min:
        Tmin_scalar = 0
        
    if Tmin > Tmin_max:
        Tmin_scalar = 1
        
    return abs(round(Tmin_scalar, 4))

#%% SWC (soil water content) scalar
def f_SWC(SWC, SWC_min=SWC_min, SWC_max=SWC_max, b=b):
    
    REW = (SWC - SWC_min) / (SWC_max - SWC_min) # what is REW?
  
    SWC_scalar = np.power(REW, b)

    return abs(round(SWC_scalar, 4))

#%%
def APAR(SWRad, FPAR):
    IPAR = (SWRad * 0.45)
    APAR = FPAR * IPAR
    return APAR

#%%
def calc_GPP(Tmin, VPD, SWRad, FPAR, SWC):
    
    if isinstance(Tmin, pd.Series) or isinstance(Tmin, pd.DataFrame):
        Tmin = Tmin.apply(f_Tmin)
    else:
        Tmin = f_Tmin(Tmin)
    
    if isinstance(VPD, pd.Series) or isinstance(VPD, pd.DataFrame):
        VPD = VPD.apply(f_VPD)
    else:
        VPD = f_VPD(VPD)
        
    if isinstance(SWC, pd.Series) or isinstance(SWC, pd.DataFrame):
        SWC = SWC.apply(f_SWC)
    else:
        SWC = f_SWC(SWC)

    return epsilon_max * Tmin * VPD * APAR(SWRad,FPAR) * SWC * 1000 # GPP in gC m-2 day-1

#%% read data
df = pd.read_csv('data/predictor-variables_Jena.csv', index_col=0, parse_dates=True)

## get predictor variables
Tmin = df['t2mmin']
VPD = df['vpd']
SWRad = df['ssrd']
FPAR = df['FPAR']

# create fake SWC for now based on fpar signal and noise
#@todo add real SWC
df['SWC'] = (df['FPAR'] + np.random.normal(loc=0, scale=0.1, size=len(df['FPAR']))) * 150
SWC = df['SWC']

#%% calc GPP
df['GPP'] = calc_GPP(Tmin, VPD, SWRad, FPAR, SWC)
df['GPP_constant-Tmin'] = calc_GPP(10, VPD, SWRad, FPAR, SWC)
df['GPP_constant-SWrad'] = calc_GPP(Tmin, VPD, 15, FPAR, SWC)
df['GPP_constant-VPD'] = calc_GPP(Tmin, 650, SWRad, FPAR, SWC)
df['GPP_constant-FPAR'] = calc_GPP(Tmin, VPD, SWRad, 0.5, SWC)
df['GPP_constant-SWC'] = calc_GPP(Tmin, VPD, SWRad, FPAR, 30)

#%%make plot
variables = ['t2mmin', 'vpd', 'ssrd', 'FPAR', 'SWC', 'GPP', 
             'GPP_constant-Tmin', 'GPP_constant-SWrad', 'GPP_constant-VPD', 'GPP_constant-SWC', 'GPP_constant-FPAR']
df['2016'][variables].plot(subplots=True, layout=(4,3), figsize=(14,10))
plt.show()

#%%save data to disk
df.to_csv('data/predictor-variables+GPP_Jena_'+LCT+'.csv')
