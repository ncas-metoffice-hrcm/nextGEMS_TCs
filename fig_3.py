# tc_ri_and_compisites_cycle3.py

import os, glob
import numpy as np
import xarray as xr
from collections import OrderedDict
#from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from storm_assess import track_tempext_nextgems, track_ibtracs
from utilities import utilities

# Options
maxgap = '0'


# Setup
ri_threshold = 15.4     # 15.4 m/s = 30 kt
major_threshold = 49.4  # m/s
hurr_threshold = 33.  # m/s

#nstorms = 50
percentile = 10.    # top percentile of storms

geostrophic_normalisation = False
latitude_threshold = 0.

working = '/home/b/b381900/nextGEMS/tropical_cyclone_analysis/'
ibt_track_dir = '/home/b/b381900/work/data/observations/IBTrACS_v4/'
track_dir = '/work/bb1153/b381900/tracking/TempestExtremes/tracks/'

members = utilities.declare_nextgems_simulations()
MEMBERS = dict()
for member in members:
    if not 'HAM' in member:
        MEMBERS[member] = members[member]

members = list(MEMBERS.keys())#['tco1279','tco2559','tco3999','IFS_28_NEMO_25_cycle3','IFS_9_NEMO_25_cycle3','IFS_9_FESOM_5_cycle3','IFS_4_FESOM_5_cycle3','IFS_9_FESOM_5_production',
#'ngc2013','ngc2012','ngc2009','ngc3028_8','ngc3028_10','ngc4005_9']
ordered_keys = ['IBTrACS']+members
resol_labels = ['']+[MEMBERS[member]['atmos_resol'] for member in MEMBERS]
legend_labels = [MEMBERS[member]['label_fig'] for member in MEMBERS]


# FIG 1 (intensification rates, RI ratio)
# =====

# Setup figure
print('plotting RI')
figure, axes = plt.subplots(nrows = 2,ncols = 2,figsize = (10,8),
    gridspec_kw = {'height_ratios': [2,1],'width_ratios': [3,2]})

figure_args = dict(
    dv = dict(
        obs_unit = r'ms$^{-1}$ 24h$^{-1}$',
        BINS = np.arange(-30.,36.,3.),
        #xaxis_ri = range(-30,31,5),
        xlabel = '24-h intensification rate',
        #y_max = 0.06,
        leg_loc = 2,
        tit_loc = 1))

PDFs = dict()
for obs_name in figure_args.keys():
    PDFs[obs_name] = dict()
RI_RATIO = dict()
RI_CASES = dict()
MAJOR_CASES = dict()
HURR_CASES = dict()
STRONGEST_CASES = dict()

# Load IBTrACS
# (netcdf data for RI plot and ascii data for composite timeseries -- both give an identical intensification rate distribution)
print('loading IBTrACS...')
years = range(1980,2023)
print(' ...year range {}-{}'.format(years[0],years[-1]))
ibtracs_dir = '/work/bb1153/b381900/data/observations/IBTrACS_v4/'
ibtracs_fn_nc = 'IBTrACS.since1980.v04r00.nc'
#ibtracs_fn_nc = 'IBTrACS.since1980.v04r01.nc'
#ibtracs_fn_nc = 'IBTrACS.EP.v04r00.nc'
ibtracs_ffp_nc = os.path.join(ibtracs_dir,ibtracs_fn_nc)

ibtracs_data = xr.open_dataset(ibtracs_ffp_nc)
ibtracs_numobs = ibtracs_data.numobs
ibtracs_type = ibtracs_data.track_type
ibtracs_vmax = ibtracs_data.wmo_wind

n_storms = len(ibtracs_data.storm)
main_idx = [i for i in range(n_storms) if ibtracs_type[i].data == b'main']                            # i.e., omit "spur" storms
n_spurs = n_storms - len(main_idx)
print(' ...removed {} spurs'.format(n_spurs))

ibtracs_vmax_maxima = list()
ibtracs_dv_maxima = list()

threshold_vmax = np.nanpercentile([np.nanmax(ibtracs_vmax[idx].data) / 1.94384 for idx in main_idx],100.-percentile)

for idx in main_idx:
    d_1 = utilities.numpy_datetime64_to_datetime(ibtracs_vmax[idx].time.data[0])                                # 1st timestep
    if d_1.year in years:
        numobs = int(ibtracs_numobs[idx].data)-1
        if numobs > 1:
            d_end = utilities.numpy_datetime64_to_datetime(ibtracs_vmax[idx].time.data[numobs])                 # last timestep
            storm_lifetime = (d_end-d_1).days
            storm_vmax = ibtracs_vmax[idx][:numobs].data / 1.94384                                              # convert to m/s
            if not np.isnan(storm_vmax).all() and storm_lifetime >= 1.:                                         # omit < 1 day
                vmax_idx = np.where(storm_vmax == np.nanmax(storm_vmax))[0]
                storm_dv = utilities.get_storm_dv_ibtracs(storm_vmax)
                if storm_dv:# and max(storm_vmax) >= threshold_vmax:
                    ibtracs_dv_maxima.extend(storm_dv)

BINS = figure_args['dv']['BINS']
obs = ibtracs_dv_maxima

print('IBTrACS')
hemispheres = ['NH','SH']
years = range(1980,2021)

RI_CASES['IBTrACS'] = list()
MAJOR_CASES['IBTrACS'] = list()
HURR_CASES['IBTrACS'] = list()
STRONGEST_CASES['IBTrACS'] = list()

ibt_storms = []
for year in years:
    for hemisphere in hemispheres:
        ibt_track_fn = '{}_IbTracs_v4.{}.tracks'.format(str(year),hemisphere)
        ibt_ffp = os.path.join(ibt_track_dir,ibt_track_fn)

        ibt_storms.extend(list(track_ibtracs.load(ibt_ffp)))

nstorms = int(len(ibt_storms) / percentile)
print(nstorms)

vmax_maxs = [storm.vmax for storm in ibt_storms if not storm.obs == []]
strongest_vmaxs = sorted(vmax_maxs)[-nstorms:]

# dv
BINS = figure_args['dv']['BINS']
obs = list()
for storm in ibt_storms:
    if not storm.obs == []:
        dv = utilities.get_storm_dv_storm_assess(storm)
        obs.extend(dv)
        if any(v > ri_threshold for v in dv):
            RI_CASES['IBTrACS'].append(storm)
        if storm.vmax > major_threshold:
            MAJOR_CASES['IBTrACS'].append(storm)
        if storm.vmax > hurr_threshold:
            HURR_CASES['IBTrACS'].append(storm)
        if storm.vmax in strongest_vmaxs:
            STRONGEST_CASES['IBTrACS'].append(storm)

#obs = [i for i in obs if i is not None]
#obs = list(filter(lambda i: np.logical_not(np.isnan(i)),np.array(obs)))
#PDFs['dv']['ibtracs'] = gaussian_kde(obs)
PDFs['dv']['ibtracs'] = np.histogram(obs,BINS,density=True)

ri_cases = [i for i in obs if i > ri_threshold]
#ri_ratio = float(len(ri_cases)) / float(len(obs))
obs_positive = [i for i in obs if i > 0.]
ri_ratio = float(len(ri_cases)) / float(len(obs_positive))
RI_RATIO['IBTrACS'] = round(ri_ratio,4)

#axes[0,0].plot(BINS,PDFs['dv']['ibtracs'](BINS),color = 'k',linewidth = 2.,label = 'IBTrACS')
axes[0,0].plot(PDFs['dv']['ibtracs'][1][:-1],PDFs['dv']['ibtracs'][0],color = 'k',linewidth = 2.,label = 'IBTrACS')

# Load nextGEMS
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

    RI_CASES[member] = list()
    MAJOR_CASES[member] = list()
    HURR_CASES[member] = list()
    STRONGEST_CASES[member] = list()

    if member.startswith('ngc2'):
        storms = list(track_tempext_nextgems.load(track_ffp,unstructured_grid = True))
    elif member.startswith('tco') or member.startswith('IFS') or member.startswith('ngc3'):
        storms = list(track_tempext_nextgems.load(track_ffp,unstructured_grid = False))

    vmax_maxs = [storm.vmax for storm in storms]

    nstorms = int(len(storms) / percentile)
    strongest_vmaxs = sorted(vmax_maxs)[-nstorms:]

    # dv
    BINS = figure_args['dv']['BINS']
    obs = list()
    for storm in storms:
        dv = utilities.get_storm_dv_storm_assess(storm)
        obs.extend(dv)
        if any(v > ri_threshold for v in dv):
            RI_CASES[member].append(storm)
        if storm.vmax > major_threshold:
            MAJOR_CASES[member].append(storm)
        if storm.vmax > hurr_threshold:
            HURR_CASES[member].append(storm)
        if storm.vmax in strongest_vmaxs:
            STRONGEST_CASES[member].append(storm)
    #PDFs['dv'][member] = gaussian_kde(obs)
    PDFs['dv'][member] = np.histogram(obs,BINS,density=True)

    ri_cases = [i for i in obs if i > ri_threshold]
    #ri_ratio = float(len(ri_cases)) / float(len(obs))
    obs_positive = [i for i in obs if i > 0.]
    ri_ratio = float(len(ri_cases)) / float(len(obs_positive))
    RI_RATIO[member] = round(ri_ratio,4)

    axes[0,0].plot(PDFs['dv'][member][1][:-1],PDFs['dv'][member][0],color = colour,linewidth = 1.,label = member)

# Plot inset
ax = plt.axes([.38,.75,.20,.18])
plt.plot(PDFs['dv']['ibtracs'][1][:-1],PDFs['dv']['ibtracs'][0],color = 'k',linewidth = 2.,label = 'IBTrACS')
for member in members:
    plt.plot(PDFs['dv'][member][1][:-1],PDFs['dv'][member][0],color = MEMBERS[member]['colour'],linewidth = 1.,label = member)
plt.xlim(15,30)
plt.ylim(0,0.002)
#plt.gca().set_yscale('log')
#plt.ylim(1E-4,2E-3)
plt.yticks([0.,0.0005,0.001,0.0015,0.002],[str(x) for x in [0.0,0.5,1.0,1.5,2.0]])
ax.set_ylabel(r'x10$^{-3}$')
ax.yaxis.set_label_coords(-0.15,1.1)

axes[0,0].set_title('c',fontweight='bold',loc='left')#24-hour intensification rates')
axes[0,0].set_xlim(-30,30)
axes[0,0].set_ylim(0,0.18)
axes[0,1].set_ylim(0,0.05)
axes[0,1].set_ylabel('RI ratio')
axes[0,1].set_title('d',fontweight='bold',loc='left')#r'RI ratio [threshold = {}'.format(ri_threshold)+' ms$^{-1}$]')
axes[0,1].set_xticklabels(ordered_keys,rotation = 90)

# Plot RI ratio
axes[0,0].axvline(x = ri_threshold,color = 'grey',linestyle = '--',linewidth = 1.)
axes[0,0].axvline(x = 0,color = 'grey',linewidth = 1.)
axes[0,0].text(16.,0.03,'{}'.format(ri_threshold)+r' ms$^{-1}$ 24h$^{-1}$',color = 'grey')
bar = axes[0,1].bar(ordered_keys,[RI_RATIO[k] for k in ordered_keys],width = 0.5)
axes[0,1].set_xticklabels(['IBTrACS']+legend_labels,rotation = 90)
for b,c,l in zip(bar,['k']+[MEMBERS[m]['colour'] for m in ordered_keys[1:]],resol_labels):
    b.set_facecolor(c)
    b.set_edgecolor('k')
    axes[0,1].annotate(l,(b.get_x()+(b.get_width()/2.),b.get_height()),
        xytext = (0,2),textcoords = 'offset points',
        ha = 'center',va = 'bottom',rotation = '90',fontsize = 10)

# Format, save plot
for loc in ['top','right']:
    axes[0,0].spines[loc].set_visible(False)
    axes[0,1].spines[loc].set_visible(False)
obs_names = ['dv']
for obs_name in obs_names:
    axes[0,obs_names.index(obs_name)].set_xlabel('{} [{}]'.format(figure_args[obs_name]['xlabel'],figure_args[obs_name]['obs_unit']))
    axes[0,obs_names.index(obs_name)].set_ylabel('Frequency [density]')
handles = [mlines.Line2D([0],[0],color = 'k',linewidth = 2.,label = 'IBTrACS')]+\
    [mlines.Line2D([0],[0],color = c,linewidth = 1.,label = l)\
        for c,l in zip([MEMBERS[m]['colour'] for m in members],legend_labels)]
axes[1,0].legend(handles = handles,loc = 'upper center',ncol = 1,frameon = False,prop = dict(size = 9),bbox_to_anchor=(0.25,2.5))
axes[1,0].set_axis_off()
axes[1,1].set_axis_off()
plt.tight_layout()

plt.savefig('figures/RI_cycle4_IBTrACS_vs_ICON_and_IFS.pdf')
plt.show()


# FIG 2 (composite timeseries)
# =====

# Setup figure
print('plotting composite timeseries')
plt.clf()
figure, axes = plt.subplots(nrows = 6,ncols = 1,figsize = (7,18))

for member in ['IBTrACS']+members:
    TS = []
    MSLP = []
    VMAX = []
    SST = []
    PRECIP = []

    print(member)
    if member in list(MEMBERS.keys()):
        if member.startswith('ngc3'):
            zoom = member.rsplit('_')[-1]
        enddate = MEMBERS[member]['enddate']
        colour = MEMBERS[member]['colour']
        label = MEMBERS[member]['label']
        label_fig = MEMBERS[member]['label_fig']

    for S in STRONGEST_CASES[member]:
    #for S in RI_CASES[member]:
    #for S in MAJOR_CASES[member]:
    #for S in HURR_CASES[member]:
        lifetime_S = S.nrecords()
        vmax_tstep = [S.obs[i].date for i in range(S.nrecords())].index(S.obs_at_vmax().date)
        standardised_t = np.linspace(-(vmax_tstep),lifetime_S-vmax_tstep-1,lifetime_S)

        # get variables
        mslp_s = np.array([S.obs[i].mslp for i in range(S.nrecords())])
        vmax_s = np.array([S.obs[i].vmax for i in range(S.nrecords())])
        if not member == 'IBTrACS':
            sst_s = np.array([S.obs[i].extras['sst'] for i in range(S.nrecords())])
            if member.startswith('ngc'):
                tp_s = np.array([S.obs[i].extras['tp']*86400. for i in range(S.nrecords())])
            elif member.startswith('tco') or member.startswith('IFS'):
                tp_s = np.array([S.obs[i].extras['tp'] for i in range(S.nrecords())])
        # handle missing values
        idx = np.where(vmax_s < 0.)[0]
        vmax_s[idx] = np.nan
        idx = np.where(vmax_s > 100.)[0]
        vmax_s[idx] = np.nan
        idx = np.where(mslp_s == 0.)[0]
        mslp_s[idx] = np.nan
        if not member == 'IBTrACS':
            idx = np.where(sst_s < 275.)[0]
            sst_s[idx] = np.nan
            idx = np.where(sst_s == 9999.)[0]
            sst_s[idx] = np.nan

        TS.append(standardised_t)
        MSLP.append(mslp_s)
        VMAX.append(vmax_s)
        if not member == 'IBTrACS':
            SST.append(sst_s)
            PRECIP.append(tp_s)

    mslp_composite_mean = []
    mslp_composite_std = []
    vmax_composite_mean = []
    vmax_composite_std = []
    sample_size = []
    if not member == 'IBTrACS':
        sst_composite_mean = []
        sst_composite_std = []
        tp_composite_mean = []
        tp_composite_std = []

    min_ts = int(min([min(s) for s in TS]))
    max_ts = int(max([max(s) for s in TS]))
    xaxis = (np.arange(min_ts,max_ts+1)*6.)/24.
    for t in range(min_ts,max_ts+1):
        s_idx = [s for s in range(len(TS)) if np.isin(t,TS[s])]

        mslp_t = [MSLP[s_i][TS[s_i].tolist().index(t)] for s_i in s_idx]
        mslp_composite_mean.append(np.nanmean(mslp_t,axis = 0))
        mslp_composite_std.append(np.nanstd(mslp_t,axis = 0))
        vmax_t = [VMAX[s_i][TS[s_i].tolist().index(t)] for s_i in s_idx]
        vmax_composite_mean.append(np.nanmean(vmax_t,axis = 0))
        vmax_composite_std.append(np.nanstd(vmax_t,axis = 0))
        sample_size.append(len(vmax_t))
        if not member == 'IBTrACS':
            sst_t = [SST[s_i][TS[s_i].tolist().index(t)] for s_i in s_idx]
            sst_composite_mean.append(np.nanmean(sst_t,axis = 0))
            sst_composite_std.append(np.nanstd(sst_t,axis = 0))
            tp_t = [PRECIP[s_i][TS[s_i].tolist().index(t)] for s_i in s_idx]
            tp_composite_mean.append(np.nanmean(tp_t,axis = 0))
            tp_composite_std.append(np.nanstd(tp_t,axis = 0))

    if member == 'IBTrACS':
        axes[0].plot(xaxis,vmax_composite_mean,linewidth=2.,color='k')
        axes[1].plot(xaxis,mslp_composite_mean,linewidth=2.,color='k')
        axes[4].step(xaxis,sample_size,linewidth=2.,color='k')
    else:
        axes[0].plot(xaxis,vmax_composite_mean,linewidth=1.,color=colour)
        axes[1].plot(xaxis,mslp_composite_mean,linewidth=1.,color=colour)
        axes[2].plot(xaxis,sst_composite_mean,linewidth=1.,color=colour)
        axes[3].plot(xaxis,tp_composite_mean,linewidth=1.,color=colour)
        axes[4].step(xaxis,sample_size,linewidth=1.,color=colour)

# Format figure
for ax,l in zip(range(5),['a','b','c','d','e']):
    axes[ax].set_title(l,fontweight='bold',loc='left')
    axes[ax].axvline(x = 0,color = 'grey',linestyle = '--',linewidth = 1)
    axes[ax].set_xlim(-8,8)
    if ax < 4:
        axes[ax].set_xlabel('')
        #axes[ax].set_xticklabels([])
    for loc in ['top','right']:
        axes[ax].spines[loc].set_visible(False)
axes[0].set_ylabel(r'$\it{v}_{max}$ [ms$^{-1}$]')
axes[0].set_ylim(0,70)
axes[1].set_ylabel(r'$\it{p}_{min}$ [hPa]')
axes[1].set_ylim(900,1020)
axes[2].set_ylabel(r'SST [K]')
axes[2].set_ylim(280,310)
axes[3].set_ylabel(r'pr [mm day$^{-1}$]')
axes[3].set_ylim(0,40)
axes[4].set_ylabel(r'$\it{n}$')
axes[4].set_yscale('log')
axes[4].set_ylim(1E0,6E2)
axes[4].set_xlabel(r'composite t [days, centred on $\it{v}_{max}$]')
if percentile == 10.:
    axes[0].set_title('Strongest decile  TCs')
else:
    axes[0].set_title(r'Strongest {}% TCs'.format(percentile))
#axes[0].set_title(r'RI ratio [threshold = {}'.format(ri_threshold)+' ms$^{-1}$]')
#axes[0].set_title(r'Cat3 $-$ Cat5')
#axes[0].set_title(r'Cat1 $-$ Cat5')
labels = ['{} ({})'.format(member,len(STRONGEST_CASES[member])) for member in members]
#labels = ['{} ({})'.format(member,len(RI_CASES[member])) for member in members]
#labels = ['{} ({})'.format(member,len(MAJOR_CASES[member])) for member in members]
#labels = ['{} ({})'.format(member,len(HURR_CASES[member])) for member in members]
#handles = [mlines.Line2D([0],[0],color = 'k',linewidth = 2.,label = 'IBTrACS ({})'.format(len(ibt_storms)))]+\
handels = [mlines.Line2D([0],[0],color = 'k',linewidth = 2.,label = 'IBTrACS ({})'.format(len(STRONGEST_CASES['IBTrACS'])))]+\
    [mlines.Line2D([0],[0],color = c,linewidth = 1.,label = l)\
        #for c,l in zip(['k','orange','goldenrod','firebrick','b','darkblue'],labels)]   # IFS cycles 2 & 3
        for c,l in zip([MEMBERS[member]['colour'] for member in members],legend_labels)]
axes[5].legend(handles = handles,loc = 'center',ncol = 3,frameon = False,prop = dict(size = 8))
axes[5].set_axis_off()
plt.tight_layout()

# Save
plt.savefig('figures/RI_composites_cycle4_IBTrACS_vs_ICON_and_IFS.pdf',bbox_inches = 'tight')
#plt.savefig('figures/MAJOR_composites_IBTrACS_vs_ICON_and_IFS.pdf',bbox_inches = 'tight')
#plt.savefig('figures/ALL_HURRICANE_composites_IBTrACS_vs_ICON_and_IFS.pdf',bbox_inches = 'tight')
plt.show()

print('done')
