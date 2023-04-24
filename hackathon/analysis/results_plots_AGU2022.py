#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:55:17 2022
Title: Analysis of USMILE Hackathon: Make plots for AGU
@author: awinkler
"""

# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from glob import glob
import os

# %%
pd.options.display.float_format = '{:,.2f}'.format
#sb.set_theme(style='whitegrid', palette='pastel')
sb.set_context("notebook", rc={"lines.linewidth": 0.6})
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 300

plt.rcParams['font.size']= 18
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''\usepackage{libertine}
                                          \usepackage[libertine]{newtxmath}'''

# %%
def get_msc(x, smooth=True):
    msc = x.groupby('time.dayofyear').mean()
    msc_0 = msc.copy().assign_coords(dayofyear=np.arange(1 - 366, 1))
    msc_1 = msc.copy().assign_coords(dayofyear=np.arange(367, 367 + 366))
    msc_stack = xr.concat((msc_0, msc, msc_1), dim='dayofyear')
    msc_stack = msc_stack.rolling(dayofyear=31, min_periods=4, center=True).median()
    msc_smooth = msc_stack.rolling(dayofyear=41, min_periods=21, center=True).mean().sel(dayofyear=slice(1, 366))

    if smooth:
        return msc_smooth
    else:
        return msc

def add_timescales(x):
    for var in x.data_vars:
        x[var + '_msc'] = get_msc(x[var])
        x[var + '_iav'] = x[var].groupby('time.dayofyear') - x[var + '_msc']
        x[var + '_ym'] = x[var].groupby('time.year').mean()
    return x

def drop_first_year(x: xr.Dataset):
    x = x.sel(
        time=slice(
            str(x.time.dt.year[0].item() + 1),
            None
        )
    )
    return x

def compute_metrics(x, dim=None):
    res = xr.Dataset()
    for metric in ['nse', 'rmse']:
        for ext in ['', '_msc', '_iav']:
            obs = x['GPP' + ext]
            mod = x['GPP_hat' + ext]

            if dim == 'time':
                if ext == '_msc':
                    obs_dim = mod_dim = 'dayofyear'
                else:
                    obs_dim = mod_dim = 'time'
            elif dim is None:
                obs_dim = set(obs.dims) - {'model', 'quantile'}
                mod_dim = set(mod.dims) - {'model', 'quantile'}

            if metric == 'nse':
                met = 1 - (
                    ((obs - mod) ** 2).sum(mod_dim) / ((obs - obs.mean(obs_dim)) ** 2).sum(obs_dim)
                )
                met = 1 / (2 - met)
            elif metric == 'rmse':
                met = ((obs - mod) ** 2).mean(mod_dim) ** 0.5

            res[metric + ext] = met

    return res

# %%
models = [os.path.basename(d) for d in glob('../logs/*')]
#models = models[-1]
models

# %%
ds_CO2 = xr.open_dataset('../logs/linear/sensitivity/CO2/predictions.nc').drop('GPP_hat')
ds_causal = xr.open_dataset('../logs/linear/sensitivity/causal/predictions.nc').drop('GPP_hat')
ds_time = xr.open_dataset('../logs/linear/extrap/time/predictions.nc').drop('GPP_hat')
ds_space = xr.open_dataset('../logs/linear/extrap/space/predictions.nc').drop('GPP_hat')
ds_val = xr.open_dataset('../logs/linear/xval/final/predictions.nc').drop('GPP_hat')            

for data, dirname in zip([ds_CO2, ds_causal, ds_time, ds_space, ds_val], ['CO2', 'causal', 'time', 'space', 'xval']):
    dt = []
    for model in models:
        print(model)
        if dirname in ['time', 'space']:
            dt.append(xr.open_dataset(f'../logs/{model}/extrap/{dirname}/predictions.nc')['GPP_hat'])
        elif dirname == 'xval':
            dt.append(xr.open_dataset(f'../logs/{model}/xval/final/predictions.nc')['GPP_hat'])
        else:
            if model == 'multimodel':
                continue
            dt.append(xr.open_dataset(f'../logs/{model}/sensitivity/{dirname}/predictions.nc')['GPP_hat'])

    data['GPP_hat'] = xr.concat(dt, dim='model')
    
#%% replace GPP for CO2; rerun sensitvitiy script to get rid off this step
GPP_obs = xr.open_dataset('../../simple_gpp_model/data/CMIP6/predictor-variables_historical+ssp585+GPP_no-CO2-change.nc')
ds_CO2['GPP'] = GPP_obs['GPP']
    
#%% calc annual means
ds_CO2_ym = ds_CO2.groupby('time.year').mean()
ds_causal_ym = ds_causal.groupby('time.year').mean()
ds_time_ym = ds_time.groupby('time.year').mean()
ds_space_ym = ds_space.groupby('time.year').mean()

#%% make space line plot
location = 2
ax = plt.subplot(111)
sb.despine()
ds_space_ym['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2)
l1 = plt.legend(models, title='Models', frameon=False, ncol=2)
ds_space_ym['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

l2 = plt.legend([p1], ["Truth"], loc='lower right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_xlim(xlims)
ax.set_ylim(ylims)

plt.title('Spatial extrapolation to unseen site (ensemble median)', fontsize=15, pad=10)
plt.ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
plt.savefig('../../plots/AGU2022/spatial_extrapolation_'+str(location)+'.pdf')

#%% make space line plot
location = 2
ax = plt.subplot(111)
sb.despine()
ds_time_ym['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=3)
l1 = plt.legend(models, title='Models', frameon=False, ncol=2)
ds_time_ym['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

ax.set_xlim(xlims)
ax.set_ylim(ylims)

l2 = plt.legend([p1], ["Truth"], loc='lower right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

plt.title('Temporal extrapolation to unseen "future" (ensemble median)', fontsize=15, pad=10)
plt.ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
#plt.savefig('../../plots/AGU2022/temporal_extrapolation_'+str(location)+'.pdf')

#%% correct for the seasonal shift
ds_time_corr = ds_time.groupby('time.year').apply(lambda x: x - x.min())
ds_time_ym_corr = ds_time_corr.groupby('time.year').mean()

#%% make space line plot
location = 2
ax = plt.subplot(111)
sb.despine()
ds_time_ym_corr['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=3)
l1 = plt.legend(models, title='Models', frameon=False, ncol=2)
ds_time_ym_corr['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

ax.set_xlim(xlims)
ax.set_ylim(ylims)

l2 = plt.legend([p1], ["Truth"], loc='lower right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

plt.title('Temporal extrapolation to unseen "future" (ensemble median)', fontsize=15, pad=10)
plt.ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
#plt.savefig('../../plots/AGU2022/temporal_extrapolation_'+str(location)+'.pdf')


#%% make space line plot
location = 0
ax = plt.subplot(111)
sb.despine()
ds_time_ym['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2)
l1 = plt.legend(models, title='Models', frameon=False, ncol=2)
ds_time_ym['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

ax.set_xlim(xlims)
ax.set_ylim(ylims)

l2 = plt.legend([p1], ["Truth"], loc='lower right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

plt.gca().set_prop_cycle(None) # reset color cycle
ds_time_ym['GPP_hat'].isel(location=location, quantile=0).plot(hue='model', ax=ax, lw=2, ls='-.', alpha=.7)
plt.gca().set_prop_cycle(None) # reset color cycle
ds_time_ym['GPP_hat'].isel(location=location, quantile=2).plot(hue='model', ax=ax, lw=2, ls='-.', alpha=.7)
plt.legend().remove()

#plt.title('Temporal extrapolation to unseen "future" (median + 0.1 $\&$ 0.9 quantiles)', fontsize=15, pad=10)
plt.title('Temporal extrapolation to unseen "future" (ensemble median)', fontsize=15, pad=10)
plt.ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
plt.savefig('../../plots/AGU2022/temporal_extrapolation_quantiles_'+str(location)+'.pdf')

#%% prepare to look into causality
ds_causal_ym_diff = ds_causal_ym - ds_time_ym.isel(model=slice(6))

#%% make space line plot
location = 0
ax = plt.subplot(111)
sb.despine()
ds_causal_ym_diff['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2)
l1 = plt.legend(models, title='Models', frameon=False, ncol=2)
ds_causal_ym_diff['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

l2 = plt.legend([p1], ["Truth"], loc='lower right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_xlim(xlims)
ax.set_ylim(ylims)

plt.title('Spatial extrapolation to unseen site (ensemble median)', fontsize=15, pad=10)
plt.ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
plt.savefig('../../plots/AGU2022/causal_extrapolation_'+str(location)+'.pdf')

#%% prepare to look into causality
ds_CO2_ym_diff = ds_time_ym.copy(deep=True)
ds_CO2_ym_diff['model'] = models[:-1]
ds_CO2_ym_diff['GPP_hat'] = ds_time_ym.isel(model=slice(6))['GPP_hat'] - ds_CO2_ym.sel(year=slice('2015','2100'))['GPP_hat']
ds_CO2_ym_diff['GPP'] = ds_time_ym.isel(model=slice(6))['GPP'] - ds_CO2_ym.sel(year=slice('2015','2100'))['GPP']
ds_CO2_ym_diff['GPP'] = ds_CO2_ym_diff['GPP'] - ds_CO2_ym_diff['GPP'].isel(year=1)
ds_CO2_ym_diff['GPP_hat'] = ds_CO2_ym_diff['GPP_hat'] - ds_CO2_ym_diff['GPP_hat'].isel(year=1)

#%% make space line plot
location = 0
ax = plt.subplot(111)
sb.despine()
ds_CO2_ym_diff.isel(location=location, quantile=0).plot.scatter(x='co2', y='GPP_hat', hue='model', ax=ax, lw=2)
l1 = plt.legend(models, title='Models', frameon=False, ncol=2)
ds_CO2_ym_diff.isel(location=location).plot.scatter(x='co2', y='GPP', ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

l2 = plt.legend([p1], ["Truth"], loc='lower right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_xlim(xlims)
ax.set_ylim(ylims)

plt.title('Spatial extrapolation to unseen site (ensemble median)', fontsize=15, pad=10)
plt.ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
plt.savefig('../../plots/AGU2022/CO2_sensitivity_'+str(location)+'.pdf')

#%% prepare dataframes
df_models = ds_CO2_ym_diff[['GPP_hat', 'co2']].isel(location=0).to_dataframe()
df_obs = ds_CO2_ym_diff[['GPP', 'co2']].isel(location=0).to_dataframe()
df_models = df_models.reset_index()
df_models = df_models.rename({'GPP_hat': 'GPP'}, axis=1)
df_obs = df_obs.reset_index()
df_obs['model'] = 'truth'
df_obs['quantile'] = 0.5
df = pd.concat([df_obs, df_models])

#%% correct cycle
ccycle = sb.color_palette()
old_color = ccycle[6]
ccycle[6] = 'k'
sb.set_palette(ccycle)

# %%
g = sb.lmplot(
    x='co2', 
    y='GPP',
    hue='model',
    hue_order=models[:-1]+['truth'],
    col='quantile',
    col_order=[0.5,0.1,0.9],
    data=df,
    height=4,
    aspect=0.8)

#plt.legend(models, title='Models', frameon=False)

g.set_axis_labels('Atmospheric CO$_2$, ppm', 'GPP, gC day$^{-1}$ m$^{-2}$')
#g.set_titles(['1', '2', '3'])
#g.despine(left=True)

for i, ax in enumerate(g.axes.flat):
    titles = [
        'Median', 'Quantile: 0.1', 'Quantile: 0.9'
    ]
    ax.set_title(titles[i])
    ax.axhline(0.0, color='k', ls=':', lw=1.2, zorder=-1)
    
plt.savefig('../../plots/AGU2022/CO2_sensitivity_lineplot.pdf')

#%% prepare boxplot
res = {}
for model in df['model'].unique()[:-1]:
    data = df[(df['quantile'] == 0.5) & (df['model'] == model)].dropna()
    x = data['co2']
    y = data['GPP'] * 365 # to annual estimates
    
    res[model] = np.polyfit(x,y, deg=1)[0]

df2 = pd.DataFrame()
df2['model'] = res.keys()
df2['value'] = res.values()

#%%

g = sb.catplot(x='model', y='value', data=df2, order=models[:-1]+['truth'], kind='bar', height=4)
g.set_axis_labels('', 'GPP sensitivity to CO2$_2$, gC yr$^{-1}$ m$^{-2}$ ppm$^{-1}$')
g.set_xticklabels(rotation=45)

plt.savefig('../../plots/AGU2022/CO2_sensitivity_barplot.pdf')

#%% correct cycle back
ccycle[6] = old_color
sb.set_palette(ccycle)

#%% calc mean seasonal cycle
ds_time_msc_start = ds_time.sel(time=slice("2015","2020")).groupby('time.dayofyear').mean()
ds_time_msc_mid = ds_time.sel(time=slice("2055","2060")).groupby('time.dayofyear').mean()
ds_time_msc_end = ds_time.sel(time=slice("2095","2100")).groupby('time.dayofyear').mean()

#%% plot seasonal cycle
location = 0
fig = plt.figure(figsize=(9,7))
ax = plt.subplot(311)
sb.despine()
ds_time_msc_start['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2)
l1 = plt.legend(models, title='Models', frameon=False, ncol=1)
ds_time_msc_start['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

ax.set_xlim(xlims)
ax.set_ylim(ylims)

l2 = plt.legend([p1], ["Truth"], loc='lower center', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_title('Mean seasonal cycle into "future climate"', fontsize=15, pad=10)
ax.set_ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('')
ax.annotate('2015--2020',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)

ax = plt.subplot(312)
sb.despine()
ds_time_msc_mid['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2)
plt.legend('').remove()
ds_time_msc_mid['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

ax.set_title('')
ax.set_ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('')

ax.annotate('2055--2060',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)

ax = plt.subplot(313)
sb.despine()
ds_time_msc_end['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2)
plt.legend('').remove()
ds_time_msc_end['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

#plt.title('Temporal extrapolation to unseen "future" (median + 0.1 $\&$ 0.9 quantiles)', fontsize=15, pad=10)
ax.set_title('')
ax.set_ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('Day of the year')

ax.annotate('2095--2100',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)

plt.savefig('../../plots/AGU2022/temporal_extrapolation_msc_'+str(location)+'.pdf')

#%% calc mean seasonal cycle for causality
#%% prepare to look into causality
ds_causal_diff = ds_time.isel(model=slice(6)).copy(deep=True)
ds_causal_diff['model'] = models[:-1]
ds_causal_diff['GPP_hat'] = ds_time.isel(model=slice(6))['GPP_hat'] - ds_causal.sel(time=slice('2015','2100'))['GPP_hat']
ds_causal_diff['GPP'] = ds_time.isel(model=slice(6))['GPP'] - ds_causal.sel(time=slice('2015','2100'))['GPP']

ds_causal_msc_start = ds_causal_diff.sel(time=slice("2015","2020")).groupby('time.dayofyear').mean()
ds_causal_msc_mid = ds_causal_diff.sel(time=slice("2055","2060")).groupby('time.dayofyear').mean()
ds_causal_msc_end = ds_causal_diff.sel(time=slice("2095","2100")).groupby('time.dayofyear').mean()

#%% plot seasonal cycle
location = 0
fig = plt.figure(figsize=(9,7))
ax = plt.subplot(311)
sb.despine()
ds_causal_msc_start['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2, alpha=0.5)
l1 = plt.legend(models, title='Models', frameon=False, ncol=2, loc='lower left')
#ds_causal_msc_start['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()



p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

ax.set_xlim(xlims)
ax.set_ylim(ymin=-5.5)

#l2 = plt.legend([p1], ["Truth"], loc='lower center', frameon=False) # this removes l1 from the axes.
#plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_title('Effect of setting non-causal variable to a constant value', fontsize=15, pad=10)
ax.set_ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('')
ax.annotate('2015--2020',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)

ax = plt.subplot(312)
sb.despine()
ds_causal_msc_mid['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2, alpha=0.5)
plt.legend('').remove()
#ds_causal_msc_mid['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

ax.set_title('')
ax.set_ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('')

ax.annotate('2055--2060',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)

ax = plt.subplot(313)
sb.despine()
ds_causal_msc_end['GPP_hat'].isel(location=location, quantile=1).plot(hue='model', ax=ax, lw=2, alpha=0.5)
plt.legend('').remove()
#ds_causal_msc_end['GPP'].isel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

#plt.title('Temporal extrapolation to unseen "future" (median + 0.1 $\&$ 0.9 quantiles)', fontsize=15, pad=10)
ax.set_title('')
ax.set_ylabel('GPP, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('Day of the year')

ax.annotate('2095--2100',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)

plt.savefig('../../plots/AGU2022/causal_consistency_msc_'+str(location)+'.pdf')

#%%
ds_time_ym.isel(model=slice(6))

# %%
ds_CO2 = drop_first_year(ds_CO2).load()
ds_causal = drop_first_year(ds_causal).load()
ds_time = drop_first_year(ds_time).load()
ds_space = drop_first_year(ds_space).load()
ds_val = drop_first_year(ds_val).load()

ds_CO2 = add_timescales(ds_CO2)
ds_causal = add_timescales(ds_causal)
ds_time = add_timescales(ds_time)
ds_space = add_timescales(ds_space)
ds_val = add_timescales(ds_val)

#ds_CO2['model'] = models
#ds_causal['model'] = models
#ds_time['model'] = models
#ds_space['model'] = models
#ds_val['model'] = models

met_CO2_global = compute_metrics(ds_CO2)
met_causal_global = compute_metrics(ds_causal)
met_time_global = compute_metrics(ds_time)
met_space_global = compute_metrics(ds_space)
met_val_global = compute_metrics(ds_val)

met_CO2 = compute_metrics(ds_CO2, dim='time')
met_causal = compute_metrics(ds_causal, dim='time')
met_time = compute_metrics(ds_time, dim='time')
met_space = compute_metrics(ds_space, dim='time')
met_val = compute_metrics(ds_val, dim='time')

# %%
met_space_global_df = met_space_global.sel(quantile=0.5).to_dataframe().reset_index()
met_time_global_df = met_time_global.sel(quantile=0.5).to_dataframe().reset_index()
met_val_global_df = met_val_global.sel(quantile=0.5).to_dataframe().reset_index()

met_space_global_df['mode'] = 'space'
met_time_global_df['mode'] = 'time'
met_val_global_df['mode'] = 'val'

met_global_df = pd.concat((met_space_global_df, met_time_global_df, met_val_global_df)).drop('quantile', axis=1)
met_global_df = pd.melt(met_global_df, id_vars=['model', 'mode'], value_vars=['nse', 'nse_msc', 'nse_iav'], var_name='scale')

# %%
met_space_df = met_space.sel(quantile=0.5).to_dataframe().reset_index()
met_time_df = met_time.sel(quantile=0.5).to_dataframe().reset_index()
met_val_df = met_val.sel(quantile=0.5).to_dataframe().reset_index()

met_space_df['mode'] = 'space'
met_time_df['mode'] = 'time'
met_val_df['mode'] = 'val'

met_df = pd.concat((met_space_df, met_time_df, met_val_df)).drop('quantile', axis=1)
met_df = pd.melt(met_df, id_vars=['model', 'mode', 'location'], value_vars=['nse', 'nse_msc', 'nse_iav'], var_name='scale')

# %%
df_plot = met_df[met_df['mode'] != 'time']
df_plot['model'] = models * int((len(df_plot) / len(models)))
df_plot['scale'] = df_plot['scale'].replace('_', '-', regex=True) # because of TeX

g = sb.catplot(
    x='scale', 
    y='value',
    hue='model',
    col='mode',
    col_order=['val','space'],
    kind='box',
    data=df_plot,
    height=4,
    aspect=0.8,
    whis=100000)

#plt.legend(models, title='Models', frameon=False)

g.set_axis_labels('', 'Normalized NSE [-]')
g.set_xticklabels(['Daily', 'Seasonal\n cycle', 'Interannual\n anomalies'])
#g.set_titles(['1', '2', '3'])
#g.despine(left=True)

for i, ax in enumerate(g.axes.flat):
    titles = [
        'Spatial cross-validation (to unseen site)',
        'Spatial (to unseen cluster)'
    ]
    ax.set_title(titles[i])
    ax.axhline(0.5, color='k', ls=':', lw=1.2, zorder=-1)
    
plt.savefig('../../plots/AGU2022/spatial_extrapolation_barplot.pdf')

#%%
df_plot = met_df
df_plot['model'] = models * int((len(df_plot) / len(models)))
df_plot['scale'] = df_plot['scale'].replace('_', '-', regex=True) # because of TeX

g = sb.catplot(
    x='scale', 
    y='value',
    hue='model',
    col='mode',
    col_order=['val','space', 'time'],
    kind='box',
    data=df_plot,
    height=4,
    aspect=0.8,
    whis=100000)

#plt.legend(models, title='Models', frameon=False)

g.set_axis_labels('', 'Normalized NSE [-]')
g.set_xticklabels(['Daily', 'Seasonal\n cycle', 'Interannual\n anomalies'])
#g.set_titles(['1', '2', '3'])
#g.despine(left=True)

for i, ax in enumerate(g.axes.flat):
    titles = [
        'Spatial cross-validation (to unseen site)',
        'Spatial (to unseen cluster)',
        'Temporal (to "future climate")'
    ]
    ax.set_title(titles[i])
    ax.axhline(0.5, color='k', ls=':', lw=1.2, zorder=-1)
    
plt.savefig('../../plots/AGU2022/spatial+temporal_extrapolation_barplot.pdf')

#%%
df_plot = met_df[met_df['mode'] == 'time']
df_plot['model'] = models * int((len(df_plot) / len(models)))
df_plot['scale'] = df_plot['scale'].replace('_', '-', regex=True) # because of TeX

g = sb.catplot(
    x='scale', 
    y='value',
    hue='model',
    col='mode',
    col_order=['time'],
    kind='box',
    data=df_plot,
    height=4,
    aspect=1,
    whis=100000)

#plt.legend(models, title='Models', frameon=False)

g.set_axis_labels('', 'Normalized NSE [-]')
g.set_xticklabels(['Daily', 'Seasonal\n cycle', 'Interannual\n anomalies'])
#g.set_titles(['1', '2', '3'])
#g.despine(left=True)

for i, ax in enumerate(g.axes.flat):
    titles = [
        'Temporal (to "future climate")'
    ]
    ax.set_title(titles[i])
    ax.axhline(0.5, color='k', ls=':', lw=1.2, zorder=-1)
    
plt.savefig('../../plots/AGU2022/temporal_extrapolation_barplot.pdf')