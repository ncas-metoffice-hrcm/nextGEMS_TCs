#!/bin/python3

# fig_1.py
#
# This script reproduces Figure 1 in the following peer-reviewed publication:
# Baker, A. J., Vannière, B., and Vidale, P. L. On the realism of tropical cyclones simulated
# in global storm-resolving climate models. Geophysical Research Letters 51, e2024GL109841.
# https://doi. org/10.1029/2024GL109841.
# (Data sources are documented therein.)


import os
import xarray as xr
import numpy as np
from datetime import datetime
from collections import OrderedDict
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from storm_assess import track_tempext_nextgems
from utilities import utilities


# SETUP
# -----
maxgap = '0'

MEMBERS = utilities.declare_nextgems_simulations()

ibtracs_dir = 'data/IBTrACS_v4/'
track_dir = 'tracks/'

resol_labels = ['','9 km','4.4 km','2.8 km','28 km','9 km','9 km','4.4 km','10km','10 km','5 km','24 km','6 km']
colours = [MEMBERS[member]['colour'] for member in MEMBERS]

figure, axes = plt.subplots(ncols = 3, nrows = 1,figsize = (12,6))

TC_N_all = dict()
TC_DAYS_all = dict()
TC_ACE_all = dict()
TC_N_nh = dict()
TC_DAYS_nh = dict()
TC_ACE_nh = dict()


# IBTrACS
# -------
print('loading IBTrACS...')

years = range(1980,2023)
print(' ...year range {}-{}'.format(years[0],years[-1]))
ibtracs_fn_nc = 'IBTrACS.since1980.v04r00.nc'
ibtracs_ffp_nc = os.path.join(ibtracs_dir,ibtracs_fn_nc)
ibtracs_data = xr.open_dataset(ibtracs_ffp_nc)
ibtracs_mslp = ibtracs_data.wmo_pres
ibtracs_vmax = ibtracs_data.wmo_wind
ibtracs_lat = ibtracs_data.lat
ibtracs_type = ibtracs_data.track_type
ibtracs_numobs = ibtracs_data.numobs

n_storms = len(ibtracs_data.storm)
main_idx = [i for i in range(n_storms) if ibtracs_type[i].data == b'main']  # i.e., omit "spur" storms
n_spurs = n_storms - len(main_idx)
print(' ...removed {} spurs'.format(n_spurs))

storm_lifetimes_all = list()
storm_lifetimes_nh = list()
ace_ibtracs_all = list()
ace_ibtracs_nh = list()
for idx in main_idx:
    d_1 = utilities.numpy_datetime64_to_datetime(ibtracs_lat[idx].time.data[0])                   # 1st timestep
    if d_1.year in years:
        numobs = int(ibtracs_numobs[idx].data)-1
        if numobs > 1:
            d_2 = utilities.numpy_datetime64_to_datetime(ibtracs_lat[idx].time.data[1])           # 2nd timestep
            d_end = utilities.numpy_datetime64_to_datetime(ibtracs_lat[idx].time.data[numobs])    # last timestep
            storm_lifetime = (d_end-d_1).days
            storm_vmax = ibtracs_vmax[idx].data[:numobs]
            if storm_lifetime >= 1. and np.nanmax(storm_vmax) >= (17.*1.94384):                   # i.e., > 1 day and > TS intensity
                storm_lifetimes_all.append(storm_lifetime)
                if (d_2-d_1).total_seconds()/60**2 == 6.:    # check whether 6 hourly
                    ace_storm = 1E-4*np.nansum((storm_vmax/1.94384)**2)
                elif (d_2-d_1).total_seconds()/60**2 == 3.:  # check whether 3 hourly
                    ace_storm = 1E-4*np.nansum((storm_vmax[::2]/1.94384)**2)
                ace_ibtracs_all.append(ace_storm)
                storm_lat = ibtracs_lat[idx].data[:numobs]
                if np.mean(storm_lat) > 0.:                 # check whether NH
                    storm_lifetimes_nh.append(storm_lifetime)
                    ace_ibtracs_nh.append(ace_storm)

TC_N_all['IBTrACS'] = len(storm_lifetimes_all) / len(years)
TC_N_nh['IBTrACS'] = len(storm_lifetimes_nh) / len(years)
TC_DAYS_all['IBTrACS'] = sum(storm_lifetimes_all) / len(years)
TC_DAYS_nh['IBTrACS'] = sum(storm_lifetimes_nh) / len(years)
TC_ACE_all['IBTrACS'] = sum(ace_ibtracs_all) / len(years)
TC_ACE_nh['IBTrACS'] = sum(ace_ibtracs_nh) / len(years)


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
    label = MEMBERS[member]['label']

    if member.startswith('ngc3'):
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

    filtered_storms_all = [storm for storm in storms if storm.vmax > utilities.declare_categories('vmax')['TS']['vmin']]  # remove TD
    filtered_storms_nh = [storm for storm in filtered_storms_all if storm.obs_at_vmax().lat > 0.]

    TC_N_all[member] = len(filtered_storms_all) / duration
    TC_N_nh[member] = len(filtered_storms_nh) / duration
    TC_DAYS_all[member] = sum([storm.nrecords()/4. for storm in filtered_storms_all]) / duration
    TC_DAYS_nh[member] = sum([storm.nrecords()/4. for storm in filtered_storms_nh]) / duration
    ace_member_all = list()
    for storm in filtered_storms_all:
        ace_storm = 1E-4*np.nansum([storm.obs[i].vmax**2 for i in range(storm.nrecords())])
        ace_member_all.append(ace_storm)
    ace_member_nh = list()
    for storm in filtered_storms_nh:
        ace_storm = 1E-4*np.nansum([storm.obs[i].vmax**2 for i in range(storm.nrecords())])
        ace_member_nh.append(ace_storm)

    TC_ACE_all[member] = sum(ace_member_all) / duration
    TC_ACE_nh[member] = sum(ace_member_nh) / duration


# Plot, format, save
bar_ntc = axes[0].bar(TC_N_all.keys(),[TC_N_all[k] for k in TC_N_all.keys()],width = 0.5)
bar_tcdays = axes[1].bar(TC_DAYS_all.keys(),[TC_DAYS_all[k] for k in TC_DAYS_all.keys()],width = 0.5)
bar_ace = axes[2].bar(TC_ACE_all.keys(),[TC_ACE_all[k] for k in TC_ACE_all.keys()],width = 0.5)
bars = [bar_ntc,bar_tcdays,bar_ace]
for bar in bars:
    for b,c,l in zip(bar,['k']+colours,resol_labels):
        b.set_facecolor(c)
        b.set_edgecolor('k')
        axes[bars.index(bar)].annotate(l,(b.get_x()+(b.get_width()/2.),b.get_height()),
            xytext = (0,2),textcoords = 'offset points',
            ha = 'center',va = 'bottom',rotation = '90',fontsize = 10)

bar_ntc = axes[0].bar(TC_N_nh.keys(),[TC_N_nh[k] for k in TC_N_nh.keys()],width = 0.5)
bar_tcdays = axes[1].bar(TC_DAYS_nh.keys(),[TC_DAYS_nh[k] for k in TC_DAYS_nh.keys()],width = 0.5)
bar_ace = axes[2].bar(TC_ACE_nh.keys(),[TC_ACE_nh[k] for k in TC_ACE_nh.keys()],width = 0.5)
bars = [bar_ntc,bar_tcdays,bar_ace]
for bar in bars:
    for b,l in zip(bar,resol_labels):
        b.set_facecolor('none')
        b.set_hatch('///')
        if l == '': # i.e., IBTrACS
            b.set_edgecolor('white')
        else:
            b.set_edgecolor('k')
bar_ntc = axes[0].bar(TC_N_all.keys(),[TC_N_all[k] for k in TC_N_all.keys()],edgecolor = 'k',facecolor = 'none',width = 0.5)
bar_tcdays = axes[1].bar(TC_DAYS_all.keys(),[TC_DAYS_all[k] for k in TC_DAYS_all.keys()],edgecolor = 'k',facecolor = 'none',width = 0.5)
bar_ace = axes[2].bar(TC_ACE_all.keys(),[TC_ACE_all[k] for k in TC_ACE_all.keys()],edgecolor = 'k',facecolor = 'none',width = 0.5)

for ax in axes:
    ax.set_xticklabels(['IBTrACS']+[MEMBERS[k]['label_fig'] for k in list(MEMBERS.keys())],rotation = 90)
axes[0].set_ylim(0,140)
axes[0].set_ylabel(r'$\it{n}_{TC}$ [year$^{-1}$]')
axes[0].set_title('a',fontweight='bold',loc='left')
axes[1].set_ylim(0,1000)
axes[1].set_ylabel(r'$\it{d}_{TC}$ [year$^{-1}$]')
axes[1].set_title('b',fontweight='bold',loc='left')
axes[2].set_ylim(0,350)
axes[2].set_ylabel(r'$\sum\it{u}_{10}$$^{2}$ [m$^{2}s^{-2}year^{-1}$]')
axes[2].set_title('c',fontweight='bold',loc='left')

for loc in ['top','right']:
    for ax in axes:
        ax.spines[loc].set_visible(False)

plt.tight_layout()

plt.savefig('figures/baker_et_al_2024_grl_fig_1.pdf')
plt.show()
print('done.')
