#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:42:33 2022
Title: Simple GPP model using the MODIS algo
Source: https://www.ntsg.umt.edu/project/modis/user-guides/mod17c61usersguidev11mar112021.pdf
@author: awinkler
"""

## import modules
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

#%%
def VPD_scalar(VPD, VPD_min=VPD_min, VPD_max=VPD_max):
    
    m = -1 / (VPD_max - VPD_min)
    t = 1 - m * VPD_min
    
    VPD_scalar = m * VPD + t
    
    if VPD < VPD_min:
        VPD_scalar = 1
        
    if VPD > VPD_max:
        VPD_scalar = 0
        
    return abs(round(VPD_scalar, 4))

#%%
def Tmin_scalar(Tmin, Tmin_min=Tmin_min, Tmin_max=Tmin_max):
    
    m = 1 / (Tmin_max - Tmin_min)
    t = 1 - m * Tmin_max
    
    Tmin_scalar = m * Tmin + t
    
    if Tmin < Tmin_min:
        Tmin_scalar = 0
        
    if Tmin > Tmin_max:
        Tmin_scalar = 1
        
    return abs(round(Tmin_scalar, 4))

#%%
def APAR(SWRad, FPAR):
    IPAR = (SWRad * 0.45)
    APAR = FPAR * IPAR
    return APAR

#%%
def calc_GPP(Tmin, VPD, SWRad, FPAR):
    
    if isinstance(Tmin, pd.Series) or isinstance(Tmin, pd.DataFrame):
        Tmin = Tmin.apply(Tmin_scalar)
    else:
        Tmin = Tmin_scalar(Tmin)
    
    if isinstance(VPD, pd.Series) or isinstance(VPD, pd.DataFrame):
        VPD = VPD.apply(VPD_scalar)
    else:
        VPD = VPD_scalar(VPD)

    return epsilon_max * Tmin * VPD * APAR(SWRad,FPAR) * 1000 # GPP in gC m-2 day-1

if __name__ == "__main__":

    #%% read data
    df = pd.read_csv('data/predictor-variables_Jena.csv', index_col=0, parse_dates=True)

    ## get predictor variables
    Tmin = df['t2mmin']
    VPD = df['vpd']
    SWRad = df['ssrd']
    FPAR = df['FPAR']

    #%% calc GPP
    df['GPP'] = calc_GPP(Tmin, VPD, SWRad, FPAR)
    df['GPP_constant-Tmin'] = calc_GPP(10, VPD, SWRad, FPAR)
    df['GPP_constant-SWrad'] = calc_GPP(Tmin, VPD, 15, FPAR)
    df['GPP_constant-VPD'] = calc_GPP(Tmin, 650, SWRad, FPAR)
    df['GPP_constant-FPAR'] = calc_GPP(Tmin, VPD, SWRad, 0.5)


    #%%make plot
    variables = ['t2mmin', 'vpd', 'ssrd', 'FPAR', 'GPP', 
                 'GPP_constant-Tmin', 'GPP_constant-SWrad', 'GPP_constant-VPD', 'GPP_constant-FPAR']
    df['2016'][variables].plot(subplots=True, layout=(4,3), figsize=(14,10))
    plt.show()

    #%%save data to disk
    df.to_csv('data/predictor-variables+GPP_Jena_'+LCT+'.csv')
