#!/bin/python3

# tc_wind_pressure_cycle4.py
# IBTrACS data are converted to 1-minute wind speeds


import os, iris
import numpy as np
import xarray as xr
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from storm_assess import track_tempext_nextgems, track_ibtracs
from utilities import utilities


# OPTIONS
# -------
maxgap = '0'


# SETUP
# -----
members = utilities.declare_nextgems_simulations()
MEMBERS = dict()
for member in members:
    if not 'HAM' in member:
        MEMBERS[member] = members[member]

df = 2

ibtracs_dir = '/work/bb1153/b381900/data/observations/IBTrACS_v4/'
track_dir = '/work/bb1153/b381900/tracking/TempestExtremes/tracks/'

resol_labels = ['','9 km','4 km','2.8 km','28 km','9 km','9 km','4.4 km','9 km','5 km','10 km','10 km','24 km','6 km','12 km']

bins_vmax = np.arange(5,96,5)
bins_mslp = np.arange(850,1021,10)

# Setup figure
figure, ax = plt.subplots(figsize=(8,8))
# create new axes on the right and on the top of the current axes
divider = make_axes_locatable(ax)
# below height and pad are in inches
ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
# make some labels invisible
ax_histx.xaxis.set_tick_params(labelbottom=False)
ax_histy.yaxis.set_tick_params(labelleft=False)


# IBTrACS
# -------
'''
print('loading IBTrACS...')
ibtracs_storms = list()
for year in range(1980,2020):
    for hemisphere in ['NH','SH']:
        ibtracs_fn = '{}_IbTracs_v4.{}.tracks'.format(year,hemisphere)
        ibtracs_ffp = os.path.join(ibtracs_dir,ibtracs_fn)
        ibtracs_storms.extend(list(track_ibtracs.load(ibtracs_ffp)))
ibtracs_mslp_minima = [storm.mslp_min for storm in ibtracs_storms]
ibtracs_vmax_maxima = [storm.vmax for storm in ibtracs_storms]
filter_idx = [i for i in range(len(ibtracs_mslp_minima)) if ibtracs_mslp_minima[i] != 0.]
ibtracs_mslp_minima = [ibtracs_mslp_minima[i] for i in filter_idx]
ibtracs_vmax_maxima = [ibtracs_vmax_maxima[i] for i in filter_idx]

print('loading IBTrACS...')
#ibtracs_fn_nc = 'IBTrACS.since1980.v04r00.nc'
ibtracs_fn_nc = 'IBTrACS.since1980.v04r01.nc'
ibtracs_ffp_nc = os.path.join(ibtracs_dir,ibtracs_fn_nc)
mslp_variable = 'Minimum central preossure from  Official WMO agency' # [sic]
vmax_variable = 'Maximum sustained wind speed from Official WMO agency'
#mslp_variable = 'Minimum central pressure'
#vmax_variable = 'tropical_cyclone_maximum_sustained_wind_speed'
try:    # one variable with name
    ibtracs_mslp = iris.load_cube(ibtracs_ffp_nc,mslp_variable)
    ibtracs_vmax = iris.load_cube(ibtracs_ffp_nc,vmax_variable)
except: # multiple variables with same name
    ibtracs_mslp = iris.load(ibtracs_ffp,mslp_variable)
    ibtracs_vmax = iris.load(ibtracs_ffp,vmax_variable)
    for cube in ibtracs_mslp:
        if cube.var_name == 'usa_pres':
            ibtracs_mslp = cube.copy()
            break
    for cube in ibtracs_vmax:
        if cube.var_name == 'usa_wind':
            ibtracs_vmax = cube.copy()
            break
n_storms = ibtracs_vmax.shape[0]
print(n_storms)
ibtracs_mslp_minima = list()
ibtracs_vmax_maxima = list()
for i in range(n_storms):
    mslp_min_i = np.min(ibtracs_mslp[i].data)
    vmax_max_i = np.max(ibtracs_vmax[i].data)
    if mslp_min_i > 0 and vmax_max_i > 0:
        ibtracs_mslp_minima.append(mslp_min_i)
        ibtracs_vmax_maxima.append(vmax_max_i/1.944)    # convert to m/s
'''
print('loading IBTrACS...')
years = range(1980,2023)
print(' ...year range {}-{}'.format(years[0],years[-1]))
ibtracs_fn_nc = 'IBTrACS.since1980.v04r00.nc'
#ibtracs_fn_nc = 'IBTrACS.since1980.v04r01.nc'
#ibtracs_fn_nc = 'IBTrACS.EP.v04r00.nc'
ibtracs_ffp_nc = os.path.join(ibtracs_dir,ibtracs_fn_nc)

print('processing IBTrACS...')
ibtracs_mslp_minima, ibtracs_vmax_maxima  = utilities.process_ibtracs_pressure_wind_to_1min(ibtracs_ffp_nc,years[0],years[-1])
idx = np.isfinite(ibtracs_vmax_maxima) & np.isfinite(ibtracs_mslp_minima)
ibtracs_mslp_minima = np.array(ibtracs_mslp_minima)[idx]
ibtracs_vmax_maxima = np.array(ibtracs_vmax_maxima)[idx]

coeff = np.polyfit(ibtracs_vmax_maxima,ibtracs_mslp_minima,df)
poly1d = np.poly1d(coeff)
ibtracs_fit_x = np.linspace(min(ibtracs_vmax_maxima),max(ibtracs_vmax_maxima),len(ibtracs_vmax_maxima))
#ibtracs_fit_x = np.linspace(1,90,90)
ibtracs_fit_y = poly1d(ibtracs_fit_x)

ibtracs_vmax_hist,ibtracs_vmax_edges = np.histogram(ibtracs_vmax_maxima,bins = bins_vmax)
ibtracs_mslp_hist,ibtracs_mslp_edges = np.histogram(ibtracs_mslp_minima,bins = bins_mslp)

ax.scatter(ibtracs_vmax_maxima,ibtracs_mslp_minima,color = 'k',marker = ',',s = 1,alpha = 0.05)
ax.plot(ibtracs_fit_x,ibtracs_fit_y,color = 'k',linewidth = 2,label = 'IBTrACS (1-min)')
ax_histx.step(ibtracs_vmax_edges[:-1],ibtracs_vmax_hist,color = 'k',linewidth = 2)
ax_histy.step(ibtracs_mslp_hist,ibtracs_mslp_edges[:-1],color = 'k',linewidth = 2)


# LOOP OVER ICON & IFS MEMBERS
# ----------------------------
for member in MEMBERS:
    print(member)

    cycle = MEMBERS[member]['cycle']
    grid = MEMBERS[member]['grid']
    duration = MEMBERS[member]['duration']
    if member.startswith('ngc3') or member.startswith('ngc4'):
        zoom = member.rsplit('_')[-1]
    startdate = MEMBERS[member]['startdate']
    enddate = MEMBERS[member]['enddate']
    colour = MEMBERS[member]['colour']
    label = MEMBERS[member]['label']
    label_fig = MEMBERS[member]['label_fig']

    if member.startswith('IFS') and 'production' in member:
        track_fn = '{member}_{startdate}-{enddate}_TempExt_{grid}_grid_nside512_maxgap{maxgap}.txt'.format(
            member=label,startdate=startdate,enddate=enddate,grid=grid,maxgap=maxgap)
    elif member.startswith('ngc3') or member.startswith('ngc4'):
        track_fn = '{member}_{startdate}-{enddate}_TempExt_{grid}_grid_maxgap{maxgap}_zoom{zoom}.txt'.format(
            member=label,startdate=startdate,enddate=enddate,grid=grid,maxgap=maxgap,zoom=zoom)
    else:
        track_fn = '{member}_{startdate}-{enddate}_TempExt_{grid}_grid_maxgap{maxgap}.txt'.format(
            member=label,startdate=startdate,enddate=enddate,grid=grid,maxgap=maxgap)
    track_ffp = os.path.join(track_dir,'cycle'+cycle+'/',label,grid+'_grid/',track_fn)

    if member.startswith('ngc2'):
        storms = list(track_tempext_nextgems.load(track_ffp,unstructured_grid = True))
    else:
        storms = list(track_tempext_nextgems.load(track_ffp))

    mslp_minima = [storm.mslp_min for storm in storms]
    vmax_maxima = [storm.vmax for storm in storms]
    idx = np.isfinite(vmax_maxima) & np.isfinite(mslp_minima)
    mslp_minima = np.array(mslp_minima)[idx]
    vmax_maxima = np.array(vmax_maxima)[idx]

    coeff = np.polyfit(vmax_maxima,mslp_minima,df)
    poly1d = np.poly1d(coeff)
    fit_x = np.linspace(min(vmax_maxima),max(vmax_maxima),len(vmax_maxima))
    #fit_x = np.linspace(1,90,90)
    fit_y = poly1d(fit_x)

    vmax_hist,vmax_edges = np.histogram(vmax_maxima,bins = bins_vmax)
    mslp_hist,mslp_edges = np.histogram(mslp_minima,bins = bins_mslp)

    ax.scatter(vmax_maxima,mslp_minima,color = colour,marker = ',',s = 1,alpha = 0.05)
    ax.plot(fit_x,fit_y,color = colour,linewidth = 1.5,label = label_fig)
    ax_histx.step(vmax_edges[:-1],vmax_hist,color = colour,linewidth = 1)
    ax_histy.step(mslp_hist,mslp_edges[:-1],color = colour,linewidth = 1)

categories_mslp = utilities.declare_categories('mslp')
p_maxima = [categories_mslp[cat]['pmax'] for cat in categories_mslp]
for cat,pmax in zip(list(categories_mslp.keys())[1:],p_maxima[1:]):
    ax.axhline(y = round(pmax),color = 'darkgrey',linestyle = '--',linewidth = 0.75)
    ax_histy.axhline(y = round(pmax),color = 'darkgrey',linestyle = '--',linewidth = 0.75)
    ax.text(93,pmax-2,list(categories_mslp.keys())[1:][p_maxima[1:].index(pmax)],
        color = 'darkgrey',backgroundcolor = 'white',fontsize = 8,ha = 'right')
categories_vmax = utilities.declare_categories('vmax')
v_minima = [round(categories_vmax[cat]['vmin']) for cat in categories_vmax]
for cat,vmin in zip(list(categories_vmax.keys())[1:],v_minima[1:]):
    ax.axvline(x = vmin,color = 'darkgrey',linestyle = '--',linewidth = 0.75)
    ax_histx.axvline(x = vmin,color = 'darkgrey',linestyle = '--',linewidth = 0.75)
    ax.text(vmin-1,1019,list(categories_vmax.keys())[1:][v_minima[1:].index(vmin)],
        color = 'darkgrey',backgroundcolor = 'white',fontsize = 8,ha = 'center')
ax_histy.set_xlabel(r'$\it{n}$')
ax_histy.set_xlim(1E0,1E3)
ax_histy.set_xscale('log')
ax_histx.set_ylabel(r'$\it{n}$')
ax_histx.set_ylim(1E0,1E3)
ax_histx.set_yscale('log')
for loc in ['top','right']:
    ax_histy.spines[loc].set_visible(False)
    ax_histx.spines[loc].set_visible(False)


plt.sca(ax)
plt.ylim(850,1025)
plt.xlim(0,95)
plt.yticks(p_maxima)
plt.xticks(v_minima)
ax.set_xlabel(r'$\it{v}_{max}$ [ms$^{-1}$]')
ax.set_ylabel(r'$\it{p}_{min}$ [hPa]')
#ax.set_title('All TCs')
plt.legend(loc = 'best',ncol = 1,prop = dict(size = 9),frameon = False)

plt.savefig('figures/wind_pressure_relationship_cycle4_TempExt.pdf')
plt.show()
print('done.')
