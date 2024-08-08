import os
import xarray as xr
import numpy as np
from storm_assess import track_tempext_radprof_nextgems
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from utilities import utilities


# Options
all_tsteps = False        # False = obs_at_vmax (only for radial wind profiles)

r = '8'                   # 8 m/s 
#r = '17'                 # 17 m/s

threshold_vmax = 17.      # vmax in m/s (0. = include all storms)
threshold_mslp = 975.     # mslp in hPa (1020. = include all storms)

density = True            # histograms in (b) and (c)


# Setup
working = '/home/b/b381900/nextGEMS/tropical_cyclone_analysis/'

members = utilities.declare_nextgems_simulations()
MEMBERS = dict()
for member in members:
    if not member.startswith('ngc'):
        MEMBERS[member] = members[member]

print('RADIUS of {} m/s\n'.format(r))

VMAX = dict()
RSIZE = dict()
IKE = dict()
RPROF = dict()
RPROF_MEAN = dict()

figure,axes = plt.subplots(nrows = 2,ncols = 4,figsize = (19,7))
handles = list()
xaxis_radius = np.linspace(0,10,159) # match radial grid defined for TempestExtremes
bins_rsize = np.arange(0.,11.,.5)
centre_rsize = (bins_rsize[:-1] + bins_rsize[1:]) / 2.
bins_ike = np.arange(0.,620.,10.)
centre_ike = (bins_ike[:-1] + bins_ike[1:]) / 2.
df = 1


# IBTrACS
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
ibtracs_time = ibtracs_data.iso_time
ibtracs_lat = ibtracs_data.lat
ibtracs_vmax = ibtracs_data.wmo_wind
ibtracs_mslp = ibtracs_data.wmo_pres
ibtracs_usarmw = ibtracs_data.usa_rmw
ibtracs_reunionrmw = ibtracs_data.reunion_rmw
ibtracs_bomrmw = ibtracs_data.bom_rmw

n_storms = len(ibtracs_data.storm)
main_idx = [i for i in range(n_storms) if ibtracs_type[i].data == b'main']                            # i.e., omit "spur" storms
n_spurs = n_storms - len(main_idx)
print(' ...removed {} spurs'.format(n_spurs))

ibtracs_vmax_maxima = list()
ibtracs_dv_maxima = list()
ibtracs_rmw_minima = list()
ibtracs_rmw_minima_lt_threshold = list()
ibtracs_lmi = list()

for idx in main_idx:
    d_1 = utilities.numpy_datetime64_to_datetime(ibtracs_vmax[idx].time.data[0])                                # 1st timestep
    if d_1.year in years:
        numobs = int(ibtracs_numobs[idx].data)-1
        if numobs > 1:
            d_end = utilities.numpy_datetime64_to_datetime(ibtracs_vmax[idx].time.data[numobs])                 # last timestep
            storm_lifetime = (d_end-d_1).days
            storm_vmax = ibtracs_vmax[idx][:numobs].data / 1.94384                                              # convert to m/s
            storm_mslp = ibtracs_mslp[idx][:numobs].data
            if not np.isnan(storm_vmax).all() and storm_lifetime >= 1.:                                         # omit < 1 day
                storm_usarmw = ibtracs_usarmw[idx][:numobs].data * 1.852                                        # convert nautical miles to km
                storm_reunionrmw = ibtracs_reunionrmw[idx][:numobs].data * 1.852
                storm_bomrmw = ibtracs_bomrmw[idx][:numobs].data * 1.852
                storm_lat = ibtracs_lat[idx][:numobs].data
                if not np.isnan(storm_usarmw).all() or not np.isnan(storm_reunionrmw).all() or not np.isnan(storm_bomrmw).all():
                    vmax_idx = np.where(storm_vmax == np.nanmax(storm_vmax))[0]
                    storm_dv = utilities.get_storm_dv_ibtracs(storm_vmax)
                    rmw = np.concatenate([storm_usarmw[vmax_idx],storm_reunionrmw[vmax_idx],storm_bomrmw[vmax_idx]])
                    mslp = storm_mslp[vmax_idx]
                    lmi = abs(storm_lat[vmax_idx])
                    if storm_dv and not np.isnan(rmw).all():
                        if max(storm_dv) > 0. and idx != 2557:  # omit storm idx 2557 that has very large RMW
                            if min(mslp) <= threshold_mslp:
                                ibtracs_dv_maxima.append(np.nanmax(storm_dv))
                                ibtracs_rmw_minima.append(np.nanmin(rmw) / 111.13)
                                ibtracs_lmi.append(np.nanmean(lmi))
                                ibtracs_rmw_minima_lt_threshold.append(np.nanmean(rmw))

rmw_mean_deg = np.mean(ibtracs_rmw_minima_lt_threshold) / 111.13                                                # convert to degrees
rmw_std_deg = np.std(ibtracs_rmw_minima_lt_threshold) / 111.13
axes[0,0].axvline(x = rmw_mean_deg,color = 'k',linewidth = 1.25)
axes[0,0].axvline(x = rmw_mean_deg-rmw_std_deg,color = 'darkgrey',linewidth = 1.25)
axes[0,0].axvline(x = rmw_mean_deg+rmw_std_deg,color = 'darkgrey',linewidth = 1.25)

coeff = np.polyfit(ibtracs_rmw_minima,ibtracs_dv_maxima,df)
poly1d = np.poly1d(coeff)
fit_x = np.linspace(min(ibtracs_rmw_minima),max(ibtracs_rmw_minima),len(ibtracs_rmw_minima))
fit_y = poly1d(fit_x)
axes[1,0].scatter(ibtracs_rmw_minima,ibtracs_dv_maxima,marker = '.',color = 'k',alpha = 0.1)
axes[1,0].plot(fit_x,fit_y,linestyle='-',linewidth=1.5,color='k')

coeff = np.polyfit(ibtracs_rmw_minima,ibtracs_lmi,df)
poly1d = np.poly1d(coeff)
fit_x = np.linspace(min(ibtracs_rmw_minima),max(ibtracs_rmw_minima),len(ibtracs_rmw_minima))
fit_y = poly1d(fit_x)
axes[1,1].scatter(ibtracs_rmw_minima,ibtracs_lmi,marker = '.',color = 'k',alpha = 0.1)
axes[1,1].plot(fit_x,fit_y,linestyle='-',linewidth=1.5,color='k')

handles.append(mlines.Line2D([0],[0],color = 'k',linewidth = 1.5,label = r'IBTrACS ($\it{n}$='+str(len(ibtracs_rmw_minima))+')'))


# Plot:
#  - radial wind profiles
#  - TC size
#  - IKE
#  - intensification rate / LMI as a function of RMW
#  - IKE per TC vs TC size
for member in MEMBERS:
    print(member)
    member_idx = list(MEMBERS.keys()).index(member)

    if member.startswith('ngc3'):
        zoom = member.rsplit('_')[-1]
    enddate = MEMBERS[member]['enddate']
    colour = MEMBERS[member]['colour']
    label = MEMBERS[member]['label']
    label_fig = MEMBERS[member]['label_fig']

    VMAX[member] = list()
    RSIZE[member] = list()
    IKE[member] = list()
    RPROF[member] = list()
    RPROF_MEAN[member] = list()

    if member.startswith('ngc3'):
        track_fn = 'radial_profiles/output/{member}_{zoom}_nodefileeditor_radprofs_all_points.txt'.format(member=label,zoom=zoom)
    else:
        track_fn = 'radial_profiles/output/{member}_nodefileeditor_radprofs_all_points.txt'.format(member=label)
    track_ffp = os.path.join(working,track_fn)
    track_fn_rgrd = track_fn.replace('.txt','_regridded_025.txt')
    track_ffp_rgrd = os.path.join(working,track_fn_rgrd)
    #storms = list(track_tempext_radprof_nextgems.load(track_ffp))
    try:
        storms_rgrd = list(track_tempext_radprof_nextgems.load(track_ffp_rgrd))
    except:
        storms_rgrd = list(track_tempext_radprof_nextgems.load(track_ffp))

    dv_maxima = list()
    rmw_minima = list()
    lmi = list()
    for storm in storms_rgrd:
        if storm.mslp_min <= threshold_mslp:
            dv = utilities.get_storm_dv_storm_assess(storm)
            dv_maxima.append(max(dv))
            rprof_vmax = storm.obs_at_vmax().extras['rprof']
            rmw = rprof_vmax.index(max(rprof_vmax)) * 0.125     # radial resolution used to compute rprof
            rmw_minima.append(rmw)
            lmi.append(abs(storm.obs_at_vmax().lat))
        for i in range(storm.nrecords()):
            if storm.obs[i].extras['r_'+r] != 0. and storm.obs[i].extras['ike_'+r] != 0.:
                VMAX[member].append(storm.obs[i].vmax)
                RSIZE[member].append(storm.obs[i].extras['r_'+r])
                IKE[member].append(storm.obs[i].extras['ike_'+r])
    #for storm in storms:
    for storm in storms_rgrd:   # temp while I run radprofs for IFS cycle4 at full resolution
        if storm.mslp_min <= threshold_mslp:
            if all_tsteps:
                rprof_storm = [storm.obs[i].extras['rprof'] for i in range(storm.nrecords())]
                RPROF[member].append(np.mean(np.array(rprof_storm),axis = 0))
            else:
                RPROF[member].append(storm.obs_at_vmax().extras['rprof'])

    VMAX[member] = np.array(VMAX[member])
    RSIZE[member] = np.array(RSIZE[member])
    IKE[member] = np.array(IKE[member]) / 1E9   # convert to TJ
    RPROF_MEAN[member] = np.nanmean(RPROF[member],axis = 0)
    hist_rsize,edges_rsize = np.histogram(RSIZE[member],bins_rsize,density = density)
    hist_ike,edges_ike = np.histogram(IKE[member],bins_ike,density = density)
    if not density:       # normalise by TC count
        hist_rsize = hist_rsize / len(storms_rgrd)
        hist_ike = hist_ike / len(storms_rgrd)

    mean_r8 = np.mean(RSIZE[member])
    std_r8 = np.std(RSIZE[member])
    mean_ike8 = np.mean(IKE[member])# / len(storms_rgrd)     # IKE per TC
    std_ike8 = np.std(IKE[member])# / len(storms_rgrd)
    mean_vmax = np.mean(VMAX[member])
    std_vmax = np.std(VMAX[member])

    axes[0,0].plot(xaxis_radius,RPROF_MEAN[member],linewidth = 1.25,color = colour,label = label_fig+' (n={})'.format(len(RPROF[member])))

    axes[0,1].plot(centre_rsize,hist_rsize,linewidth = 1.25,color = colour)
    axes[0,1].errorbar(mean_r8,.04-(member_idx*.01),xerr = std_r8/2,
        fmt = '.',color = colour,ecolor = colour,elinewidth = 0.75)
    axes[0,1].scatter(mean_r8,.04-(member_idx*.01),s = mean_vmax*(mean_vmax/4),
        marker = '.',color = colour,edgecolor = 'k',zorder = 3)

    axes[0,2].plot(centre_ike,hist_ike,linewidth = 1.25,color = colour)
    axes[0,2].errorbar(mean_ike8,4E-4-(member_idx*.4E-4),xerr = std_ike8/2,
        fmt = '.',color = colour,ecolor = colour,elinewidth = 0.75)
    axes[0,2].scatter(mean_ike8,4E-4-(member_idx*.4E-4),s = mean_vmax*(mean_vmax/3),
        marker = '.',color = colour,edgecolor = 'k',zorder = 3)

    coeff = np.polyfit(rmw_minima,dv_maxima,df)
    poly1d = np.poly1d(coeff)
    fit_x = np.linspace(min(rmw_minima),max(rmw_minima),len(rmw_minima))
    fit_y = poly1d(fit_x)
    axes[1,0].scatter(rmw_minima,dv_maxima,marker = '.',color = colour,alpha = 0.1)
    axes[1,0].plot(fit_x,fit_y,linestyle = '-',linewidth = 1.25,color = colour)

    coeff = np.polyfit(rmw_minima,lmi,df)
    poly1d = np.poly1d(coeff)
    fit_x = np.linspace(min(rmw_minima),max(rmw_minima),len(rmw_minima))
    fit_y = poly1d(fit_x)
    axes[1,1].scatter(rmw_minima,lmi,marker = '.',color = colour,alpha = 0.1)
    axes[1,1].plot(fit_x,fit_y,linestyle = '-',linewidth = 1.25,color = colour)

    axes[1,2].errorbar(mean_r8,mean_ike8,xerr = std_r8/2,yerr = std_ike8/2,
        fmt = '.',color = colour,ecolor = colour,elinewidth = 0.75)
    axes[1,2].scatter(mean_r8,mean_ike8,s = mean_vmax*(mean_vmax/4),
        marker = '.',color = colour,edgecolor = 'k',zorder = 3)

    handles.append(mlines.Line2D([0],[0],color = colour,linewidth = 1.25,label = label_fig+r' ($\it{n}$='+str(len(storms_rgrd))+')'))


# Format, save
axes[0,0].set_title('a',fontweight = 'bold',loc = 'left')
axes[0,0].set_xlim(0,3)
axes[0,0].set_xlabel(r'$\it{R}$ [$\degree$ from TC centre]')
axes[0,0].set_ylim(0,30)
axes[0,0].set_ylabel(r'$\it{v}_t$ [m s$^{-1}$]')
axes[0,0].axhline(y = 8,color = 'darkgrey',linewidth = 0.75,linestyle = '--')
axes[0,0].text(rmw_mean_deg+rmw_std_deg+0.1,3,'IBTrACS mean RMW = {}'.format(round(rmw_mean_deg,2))+r'$\degree$',
    color = 'k',fontsize = 8,ha = 'left',va = 'center')
axes[0,0].text(0.6,8,r+r' ms$^{-1}$',
    color = 'darkgrey',backgroundcolor = 'white',fontsize = 8,ha = 'left',va = 'center')
axes[0,0].text(3,2,r'$\it{p_{min}}$ $\leq$ 975 hPa',
    color = 'darkgrey',backgroundcolor = 'white',fontsize = 8,ha = 'right',va = 'center')

axes[0,1].set_title('b',fontweight = 'bold',loc = 'left')
axes[0,1].set_xlim(0,10)
axes[0,1].set_xlabel(r'$\it{R}_{'+r+'}$ [$\degree$ from TC centre]')
if density:
    axes[0,1].set_ylim(-0.04,0.3)
    axes[0,1].set_ylabel('Frequency [density]')
    #axes[0,1].set_yticks(np.arange(0.,0.35,.05))
else:
    axes[0,1].set_ylim(-0.5,4)
    axes[0,1].set_ylabel(r'Frequency [normalised by $\it{n}_{TC}$]')

axes[0,2].set_title('c',fontweight = 'bold',loc = 'left')
axes[0,2].set_xlim(0,bins_ike[-2])
axes[0,2].set_xlabel(r'IKE$_{'+r+'}$ [TJ]')
axes[0,2].set_yscale('log')
if density:
    axes[0,2].set_ylim(7E-5,3E-2)
    axes[0,2].set_ylabel('Frequency [density]')
else:
    axes[0,2].set_ylim(1E-2,1E2)
    axes[0,2].set_ylabel(r'Frequency [normalised by $\it{n}_{TC}$]')

axes[1,0].set_title('d',fontweight = 'bold',loc = 'left')
axes[1,0].set_xlim(0,3)
axes[1,0].set_xlabel(r'RMW [$\degree$ from TC centre]')
axes[1,0].set_ylim(0,40)
axes[1,0].set_ylabel(r'24-h intensification rate [ms$^{-1}$ 24h$^{-1}$]')

axes[1,1].set_title('e',fontweight = 'bold',loc = 'left')
axes[1,1].set_xlim(0,3)
axes[1,1].set_xlabel(r'RMW [$\degree$ from TC centre]')
axes[1,1].set_ylim(0,70)
axes[1,1].set_ylabel(r'LMI [$\degree$N]')

axes[1,2].set_title('f',fontweight = 'bold',loc = 'left')
#axes[1,2].set_xlim(2,6)
axes[1,2].set_xlabel(r'$\it{R}_{'+r+'}$ [$\degree$ from TC centre]')
#axes[1,2].set_ylim(0,1)
axes[1,2].set_ylabel(r'IKE$_{'+r+'}$ [TJ per TC]')

axes[1,3].legend(handles = handles,loc = 'center',prop = dict(size = 9),frameon = False)
for loc in ['top','right']:
    for ax in axes.flatten():
        ax.spines[loc].set_visible(False)
for ax in [0,1]:
    axes[ax,3].set_axis_off()
plt.tight_layout()

fig_fn = 'figures/tc_size_distributions_cycle4'
if all_tsteps:
    fig_fn = fig_fn+'_all_tsteps'
else:
    fig_fn = fig_fn+'_select_tsteps'
fig_ffp = os.path.join(working,fig_fn+'.pdf')
print(fig_ffp)
plt.savefig(fig_ffp)
plt.show()
print('done')
