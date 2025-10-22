#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:21:23 2024

@author: awinkler
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.path as mpath
from matplotlib.lines import Line2D

save = True
plt_path = '../../plots/manuscript/'
plotfname = plt_path+'Figure-Supp-Map'

## default styling
# plt.rcParams['axes.edgecolor']= 'black'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 80.0
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''\usepackage{libertine}
                                          \usepackage[libertine]{newtxmath}'''

# Create a new figure
fig = plt.figure()#figsize=(10, 10))

# Set the projection
ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())#Orthographic(central_longitude=0.0, central_latitude=90.0))

# Add the map features
#ax.add_feature(cfeature.LAND)
#ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)

# Set the extent of the map
#ax.set_extent([0, 130, 45, 55])
ax.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())

# Define the locations
locations1 = [
     (129.375, 47.563),
     (129.375, 49.438),
     (129.375, 45.688),
     (131.25, 47.563),
     (127.5, 47.563),
     (9.375, 49.429),
     (9.375, 51.304),
     (9.375, 47.554),
     (11.25, 49.429),
     (7.5, 49.429),
]

locations2 = [
    (-75, 49.429),
    (-75, 51.304),
    (-75, 47.554),
    (-73.125, 49.429),
    (-76.875, 49.429)
]

# Plot the locations
ax.scatter([lon for lon, lat in locations1], [lat for lon, lat in locations1], 10, marker='o', color='r', transform=ccrs.PlateCarree())
ax.scatter([lon for lon, lat in locations2], [lat for lon, lat in locations2], 10, marker='o', color='b', transform=ccrs.PlateCarree())

# Create a legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Training and Validation', markerfacecolor='r', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Testing', markerfacecolor='b', markersize=10)
]
ax.legend(handles=legend_elements, loc='center right', fancybox=False, edgecolor='k',
          borderaxespad=0.5, borderpad=0.5)

# Compute a circle in axes coordinates, which we can use as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

ax.set_boundary(circle, transform=ax.transAxes)
ax.stock_img()
ax.gridlines()


# Show the plot
if save == True:
    plt.savefig(plotfname+'.png', dpi=900)

    ## crop figure
    os.system('convert '+plotfname+'.png -trim '+plotfname+'.png')
