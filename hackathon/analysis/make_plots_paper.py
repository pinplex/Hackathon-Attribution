#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 2023
Title: Analysis of USMILE Hackathon: Make plots for manuscript
@author: awinkler & bkaft
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

# %% plot settings
#pd.options.display.float_format = '{:,.2f}'.format
#sb.set_theme(style='whitegrid', palette='pastel')
#sb.set_context("notebook", rc={"lines.linewidth": 0.6})
#plt.rcParams['figure.dpi'] = 200
#plt.rcParams['savefig.dpi'] = 300

plt.rcParams['font.size'] = 14
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''\usepackage{libertine}
                                            \usepackage[libertine]{newtxmath}'''
                                            
                                          
colors = [
        (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
        "#ffc300",
        (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
        "#f0004e",
]

sb.set_palette(colors)
#%% define paths
save = True #False # if plots should be saved
plt_pth = '../../plots/manuscript/'                                       

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

#%% print models
models = [os.path.basename(d) for d in glob('../logs/*')]
#models = models[-1]
print(models)
#%% exclude gt_model for this part of the analysis
# models.remove('gt_model') #@todo: which other models did we want to exclude
# models.remove('multimodel') #@todo: which other models did we want to exclude
models = ['linear', 'simplemlp', 'lstm', 'attn_nores'] # our choice of models to present
models_labels = ['Linear', 'Lagged MLP', 'LSTM', 'Attention']

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
    data['model'] = models
    
#%% fix: replace GPP for CO2; rerun sensitvitiy script to get rid off this step
GPP_obs = xr.open_dataset('../../simple_gpp_model/data/CMIP6/predictor-variables_historical+ssp585+GPP_no-CO2-change.nc')
ds_CO2['GPP'] = GPP_obs['GPP']
    
#%% calc annual means
ds_CO2_ym = ds_CO2.groupby('time.year').mean()
ds_causal_ym = ds_causal.groupby('time.year').mean()
ds_time_ym = ds_time.groupby('time.year').mean()
ds_space_ym = ds_space.groupby('time.year').mean()

#%% Figure 01: make line plot for extrapolation to unseen  site in unseen cluster
location = 11 #@todo: why do we take location 15?
plotfname = plt_pth+'Fig-01_temporal_spatial_extrapolation_location-'+str(location)

fig = plt.figure(figsize=(13.5,4.5))
ax = plt.subplot(121)
ax.minorticks_on()
sb.despine()
ds_space_ym['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', 
                                                                 ax=ax, lw=2)
l1 = plt.legend(models_labels, title='Models', frameon=False, ncol=2)
ds_space_ym['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

l2 = plt.legend([p1], ["Ground-truth"], loc='lower right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_xlim(xlims)
ax.set_ylim(ylims)

plt.title('Spatial extrapolation to unseen site (ensemble median)', fontsize=14, pad=10)
plt.ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
plt.annotate(r'\textbf{a}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=18, fontweight="bold")
#plt.savefig('../../plots/AGU2022/spatial_extrapolation_'+str(location)+'.pdf')

#% make line plot for extrapolation to future climate
# location = 14
ax = plt.subplot(122)
ax.minorticks_on()
sb.despine()
ds_time_ym['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=2)
l1 = plt.legend(models_labels, title='Models', frameon=False, ncol=2)
ds_time_ym['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

ax.set_xlim(xlims)
ax.set_ylim(ylims)

l2 = plt.legend([p1], ["Ground-truth"], loc='center left', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

plt.title('Temporal extrapolation to unseen site \& "future" (ensemble median)', fontsize=14, pad=10)
plt.ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
plt.annotate(r'\textbf{b}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=18, fontweight="bold")

if save == True:
    plt.savefig(plotfname+'.pdf')

    ## crop figure
    os.system('pdfcrop --margins=5 '+plotfname+'.pdf')
    os.system('mv '+plotfname+'-crop.pdf '+plotfname+'.pdf')

#%% alternative Figure 01: make line plot for extrapolation to unseen site in unseen cluster

location = 11 # 13 #@todo: why do we take location 15?; 11 for supplemenatry figure
plotfname = plt_pth+'Fig-01_temporal_spatial_extrapolation_alternative-'+str(location)

fig = plt.figure(figsize=(13.5,4.5))
ax = plt.subplot(121)
ax.minorticks_on()
sb.despine()
ds_space_ym['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=2)
l1 = plt.legend(models_labels, title='Models', frameon=False, ncol=2)
ds_space_ym['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

l2 = plt.legend([p1], ["Ground-\ntruth"], loc='lower right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_xlim(xlims)
ax.set_ylim(ylims)

plt.title('Spatial extrapolation to unseen site (ensemble median)', fontsize=14, pad=10)
plt.ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')
plt.annotate(r'\textbf{a}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")
#plt.savefig('../../plots/AGU2022/spatial_extrapolation_'+str(location)+'.pdf')

#% make line plot for extrapolation to future climate
# location = 14
ax = plt.subplot(122)
ax.minorticks_on()
sb.despine()
ds_time_ym['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=3)
l1 = plt.legend(models_labels, title='Models', frameon=False, ncol=1)
ds_time_ym['GPP'].sel(location=location).plot(ax=ax, c='k', lw=3, ls='--')

plt.gca().set_prop_cycle(None) # reset color cycle
ds_time_ym['GPP_hat'].sel(location=location, quantile=0.1).plot(hue='model', ax=ax, lw=2, ls='-.')
plt.gca().set_prop_cycle(None) # reset color cycle
ds_time_ym['GPP_hat'].sel(location=location, quantile=0.9).plot(hue='model', ax=ax, lw=2, ls='-.')
plt.legend().remove()

xlims = ax.get_xlim()
ylims = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=3, ls='--')

l2 = plt.legend([p1], ["Ground-\ntruth"], loc='center left', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_xlim(xlims)
ax.set_ylim(ylims)

#plt.title('Temporal extrapolation to unseen "future" (median + 0.1 $\&$ 0.9 quantiles)', fontsize=15, pad=10)
plt.title('Temporal extrapolation to unseen "future"\n (median $\&$ 0.1/0.9 quantiles)', fontsize=14, pad=10)
plt.ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
plt.xlabel('Time, yr')

plt.annotate(r'\textbf{b}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")

if save == True:
    plt.savefig(plotfname+'.pdf')

    ## crop figure
    os.system('pdfcrop --margins=5 '+plotfname+'.pdf')
    os.system('mv '+plotfname+'-crop.pdf '+plotfname+'.pdf')

#%% calculate CO2 sensitivity
ds_CO2_ym_diff = ds_time_ym.sel(year=slice('2016','2100'))[['GPP_hat']] - ds_CO2_ym.sel(year=slice('2016','2100'))[['GPP_hat']]
ds_CO2_ym_diff['GPP_hat'] = ds_CO2_ym_diff['GPP_hat'] - ds_CO2_ym_diff['GPP_hat'].isel(year=0)

tmp = ds_time_ym.sel(year=slice('2016','2100'))[['GPP']] - ds_CO2_ym.sel(year=slice('2016','2100'))[['GPP']]
ds_CO2_ym_diff['GPP'] = tmp['GPP'] - tmp['GPP'].isel(year=2)

ds_CO2_ym_diff['co2'] = ds_time_ym.sel(year=slice('2016','2100'))['co2']
ds_CO2_ym_diff['model'] = models_labels
#%% prepare dataframes to plot CO2 sensitivity
location = 1 # locaiton that has predictions from past to future
df_models = ds_CO2_ym_diff[['GPP_hat', 'co2']].sel(location=location).to_dataframe()
df_obs = ds_CO2_ym_diff[['GPP', 'co2']].sel(location=location).to_dataframe()
df_models = df_models.reset_index()
df_models = df_models.rename({'GPP_hat': 'GPP'}, axis=1)
df_obs = df_obs.reset_index()
df_obs['model'] = 'Truth'
df_obs['quantile'] = 0.5
df = pd.concat([df_obs, df_models])

#%% add black color in color palette
# ccycle = sb.color_palette()
# old_color = ccycle[3]
# ccycle[4] = 'k'
sb.set_palette(colors+['k'])

#%% prepare boxplot (quick and dirty)
res = {}
frames = {}
for quantile in df['quantile'].unique():
    for model in models_labels+['Truth']:
        if model == 'Truth' and quantile != 0.5:
            continue
        else:
            data = df[(df['model'] == model) & (df['quantile'] == quantile)].dropna()
            x = data['co2']
            y = data['GPP'] * 365 # to annual estimates
            
            res[model] = np.polyfit(x,y, deg=1)[0]

    df2 = pd.DataFrame()
    df2['model'] = res.keys()
    df2['value'] = res.values()
    df2['quantile'] = quantile

    frames[quantile] = df2

df2 = pd.concat(frames.values())

#%% make CO2 sensitivity plot
plotfname = plt_pth+'Fig-0X_CO2-sensitivity_part-01'
g = sb.lmplot(
    x='co2', 
    y='GPP',
    hue='model',
    hue_order=models_labels+['Truth'],
    col='quantile',
    col_wrap=2,
    col_order=[0.5,0.9,0.1],
    data=df,
    height=5,
    aspect=0.9)

#plt.legend(models, title='Models', frameon=False)

g.set_axis_labels('Atmospheric CO$_2$, ppm', '$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
#g.set_titles(['1', '2', '3'])
#g.despine(left=True)

titles = [
    'Ensemble Median', 'Quantile: 0.9', 'Quantile: 0.1'
]
identifier = ['a','d','c']
for i, ax in enumerate(g.axes.flat):
    ax.set_title(titles[i], y=1.0, pad=-15)
    ax.annotate(r'\textbf{'+identifier[i]+'}', xy=(-.085, 1.05), xycoords='axes fraction', fontsize=20, fontweight="bold")
    ax.axhline(0.0, color='k', ls=':', lw=1.2, zorder=-1)
    plt.minorticks_on()
    
if save == True:
    plt.savefig(plotfname+'.pdf')

    ## crop figure
    os.system('pdfcrop --margins=5 '+plotfname+'.pdf')
    os.system('mv '+plotfname+'-crop.pdf '+plotfname+'.pdf')

#%% make CO2 sensitivity plot
plotfname = plt_pth+'Fig-0X_CO2-sensitivity_part-02'
g = sb.catplot(y='model', x='value', data=df2, order=models_labels+['Truth'], 
               kind='point', hue='model', height=5, aspect=1.1, capsize=.3, hue_order=models_labels+['Truth'], legend=False)
g.set_axis_labels('GPP sensitivity to CO$_2$, gC yr$^{-1}$ m$^{-2}$ ppm$^{-1}$', '')
g.refline(x=0, zorder=-1)
#g.set_yticklabels(direction='in')

for i, ax in enumerate(g.axes.flat):
    ax.tick_params(axis="y", direction="in", pad=-15)
    plt.setp(ax.get_yticklabels(), ha="left", bbox=dict(boxstyle="round", ec="white", fc="white", alpha=0.95))
    plt.gca().tick_params(axis='y', which='minor', left=False)
    plt.minorticks_on()
    ax.set_title('CO$_2$ Sensitivity')
    ax.set_xlim([-1.5,5.5])
    ax.annotate(r'\textbf{b}', xy=(-.085, 1.05), xycoords='axes fraction', fontsize=20, fontweight="bold")
    
if save == True:
    #plt.subplots_adjust(1)
    plt.savefig(plotfname+'.pdf', bbox_inches="tight")

    ## crop figure
    os.system('pdfcrop --margins=5 '+plotfname+'.pdf')
    os.system('mv '+plotfname+'-crop.pdf '+plotfname+'.pdf')

#%% remove balck color again
# ccycle[6] = old_color
# sb.set_palette(ccycle)

#%% calc mean seasonal cycle
ds_time_msc_start = ds_time.sel(time=slice("2015","2020")).groupby('time.dayofyear').mean()
ds_time_msc_mid = ds_time.sel(time=slice("2055","2060")).groupby('time.dayofyear').mean()
ds_time_msc_end = ds_time.sel(time=slice("2095","2100")).groupby('time.dayofyear').mean()

#%% calc mean seasonal cycle for CO2 constant run
ds_CO2_msc_start = ds_CO2.sel(time=slice("2015","2020")).groupby('time.dayofyear').mean()
ds_CO2_msc_mid = ds_CO2.sel(time=slice("2055","2060")).groupby('time.dayofyear').mean()
ds_CO2_msc_end = ds_CO2.sel(time=slice("2095","2100")).groupby('time.dayofyear').mean()

#%% plot seasonal cycle with and without CO2 change
#@todo: here make nice y-lims
plotfname = plt_pth+'Fig-0X_mean-seasonal-cycle'
location = 1
fig = plt.figure(figsize=(14,9))
ax = plt.subplot(321)
plt.minorticks_on()
sb.despine()
ds_time_msc_start['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=2)
l1 = plt.legend(models_labels, loc='upper left', title='Models', frameon=False, ncol=1)
ds_time_msc_start['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

xlims_ax1 = ax.get_xlim()
ylims_ax1 = ax.get_ylim()

p1, = plt.plot([0,0], [1,1], c='k', lw=2, ls='--')

ax.set_xlim(xlims_ax1)
ax.set_ylim(ylims_ax1)

l2 = plt.legend([p1], ["Ground-\ntruth"], loc='center right', frameon=False) # this removes l1 from the axes.
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

ax.set_title('Future predictions of mean seasonal cycle', fontsize=15, pad=10)
ax.set_ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('')
ax.annotate('2015--2020',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)
ax.annotate(r'\textbf{a}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")

ax = plt.subplot(323)
plt.minorticks_on()
sb.despine()
ds_time_msc_mid['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=2)
plt.legend('').remove()
ds_time_msc_mid['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

ax.set_title('')
ax.set_ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('')

xlims_ax2 = ax.get_xlim()
ylims_ax2 = ax.get_ylim()

ax.annotate('2055--2060',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)
ax.annotate(r'\textbf{b}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")

ax = plt.subplot(325)
plt.minorticks_on()
sb.despine()
ds_time_msc_end['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=2)
plt.legend('').remove()
ds_time_msc_end['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

#plt.title('Temporal extrapolation to unseen "future" (median + 0.1 $\&$ 0.9 quantiles)', fontsize=15, pad=10)
ax.set_title('')
ax.set_ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('Day of the year')

xlims_ax3 = ax.get_xlim()
ylims_ax3 = ax.get_ylim()

ax.annotate('2095--2100',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)
ax.annotate(r'\textbf{c}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")

#% plot seasonal cycle withtout CO2 change
ax = plt.subplot(322)
plt.minorticks_on()
sb.despine()
ds_CO2_msc_start['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=2)
plt.legend('').remove()
ds_CO2_msc_start['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')


ax.set_title('Future predictions of mean seasonal cycle with constant CO${_2}$', fontsize=15, pad=10)
ax.set_ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('')
ax.annotate('2015--2020',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)
ax.annotate(r'\textbf{d}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")

ax.set_xlim(xlims_ax1)
ax.set_ylim(ylims_ax1)

ax = plt.subplot(324)
plt.minorticks_on()
sb.despine()
ds_CO2_msc_mid['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=2)
plt.legend('').remove()
ds_CO2_msc_mid['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

ax.set_title('')
ax.set_ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('')

ax.annotate('2055--2060',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)
ax.annotate(r'\textbf{e}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")

ax.set_xlim(xlims_ax2)
ax.set_ylim(ylims_ax2)

ax = plt.subplot(326)
plt.minorticks_on()
sb.despine()
ds_CO2_msc_end['GPP_hat'].sel(location=location, quantile=0.5).plot(hue='model', ax=ax, lw=2)
plt.legend('').remove()
ds_CO2_msc_end['GPP'].sel(location=location).plot(ax=ax, c='k', lw=2, ls='--')

#plt.title('Temporal extrapolation to unseen "future" (median + 0.1 $\&$ 0.9 quantiles)', fontsize=15, pad=10)
ax.set_title('')
ax.set_ylabel('$F_\mathrm{GPP}$, gC day$^{-1}$ m$^{-2}$')
ax.set_xlabel('Day of the year')

ax.annotate('2095--2100',
            xy=(.975, .85), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=14)
ax.annotate(r'\textbf{f}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")

ax.set_xlim(xlims_ax3)
ax.set_ylim(ylims_ax3)

if save == True:
    #plt.tight_layout()
    plt.savefig(plotfname+'.pdf')

    ## crop figure
    os.system('pdfcrop --margins=5 '+plotfname+'.pdf')
    os.system('mv '+plotfname+'-crop.pdf '+plotfname+'.pdf')

#%% prepare data for space and temporal extrapolation performance
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

#%%
plotfname = plt_pth+'Fig-02_spatial-temporal-extrapolation'
df_plot = met_df
df_plot['model'] = models_labels * int((len(df_plot) / len(models_labels)))
df_plot['scale'] = df_plot['scale'].replace('_', '-', regex=True) # because of TeX

g = sb.catplot(
    x='scale', 
    y='value',
    hue='model',
    col='mode',
    col_order=['val','space', 'time'],
    kind='box',
    data=df_plot,
    height=5,
    aspect=0.8,
    whis=100000)

sb.move_legend(g, "lower left", bbox_to_anchor=(.1, .2), frameon=True, 
               fancybox=False, edgecolor='k', title='Models')
#plt.legend(models, title='Models', frameon=False)

g.set_axis_labels('', 'Normalized NSE [-]')
g.set_xticklabels(['Daily', 'Seasonal\n cycle', 'Interannual\n anomalies'])
#g.set_titles(['1', '2', '3'])
#g.despine(left=True)

titles = [
    'Spatial cross-validation (to unseen site)',
    'Spatial (to unseen cluster)',
    'Temporal (to "future climate")'
]
identifier = ['a','b','c']
for i, ax in enumerate(g.axes.flat):
    ax.annotate(r'\textbf{'+identifier[i]+'}', xy=(-.085, 1.1), xycoords='axes fraction', fontsize=20, fontweight="bold")
    ax.set_title(titles[i])
    ax.axhline(0.5, color='k', ls=':', lw=1.2, zorder=-1)
    
if save == True:
    plt.tight_layout()
    plt.savefig(plotfname+'.pdf')

    ## crop figure
    os.system('pdfcrop --margins=5 '+plotfname+'.pdf')
    os.system('mv '+plotfname+'-crop.pdf '+plotfname+'.pdf')
