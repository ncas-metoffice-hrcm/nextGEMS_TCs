# ============
# utilities.py
# ============
#
# Functions for handling model output, TempestExtremes tracks, IBTrACS data etc


import numpy as np
import xarray as xr
from collections import OrderedDict
from datetime import datetime


def declare_nextgems_simulations():
    MEMBERS = dict(
        tco1279 = dict(
            cycle = '2',
            grid = 'regular',
            duration = 1.95,
            atmos_resol = '9 km',
            label = 'tco1279-orca025',
            label_fig = 'cycle2, IFS 9km, NEMO 0.25$\degree$',
            startdate = '20200120',
            enddate = '20211231',
            linestyle = '-',
            colour = 'orange'),
        tco2559 = dict(
            cycle = '2',
            grid = 'regular',
            duration = 0.95,
            atmos_resol = '4.4 km',
            label = 'tco2559-ng5',
            label_fig = 'cycle2, IFS 4.4km, FESOM 5km',
            startdate = '20200120',
            enddate = '20201231',
            linestyle = '-',
            colour = 'goldenrod'),
        tco3999 = dict(
            cycle = '2',
            grid = 'regular',
            duration = 0.62,
            atmos_resol = '2.8 km',
            label = 'tco3999-ng5',
            label_fig = 'cycle2, IFS 2.8km, FESOM 5km',
            startdate = '20200120',
            enddate = '20200831',
            linestyle = '-',
            #colour = 'peru'),
            #colour = 'saddlebrown'),
            colour = 'sienna'),
        IFS_28_NEMO_25_cycle3 = dict(
            cycle = '3',
            grid = 'regular',
            duration = 4.95,
            atmos_resol = '28 km',
            label = 'IFS_28-NEMO_25-cycle3',
            label_fig = 'cycle3, IFS 28km, NEMO 0.25$\degree$',
            startdate = '20200121',
            enddate = '20241231',
            linestyle = '-',
            colour = 'lightcoral'),
        IFS_9_NEMO_25_cycle3 = dict(
            cycle = '3',
            grid = 'regular',
            duration = 4.95,
            atmos_resol = '9 km',
            label = 'IFS_9-NEMO_25-cycle3',
            label_fig = 'cycle3, IFS 9km, NEMO 0.25$\degree$',
            startdate = '20200121',
            enddate = '20241231',
            linestyle = '-',
            colour = 'crimson'),
        IFS_9_FESOM_5_cycle3 = dict(
            cycle = '3',
            grid = 'regular',
            duration = 0.95,
            atmos_resol = '9 km',
            label = 'IFS_9-FESOM_5-cycle3',
            label_fig = 'cycle3, IFS 9km, FESOM 5km',
            startdate = '20200121',
            enddate = '20201231',
            linestyle = '-',
            colour = 'firebrick'),
        IFS_4_FESOM_5_cycle3 = dict(
            cycle = '3',
            grid = 'regular',
            duration = 4.95,
            atmos_resol = '4.4 km',
            label = 'IFS_4.4-FESOM_5-cycle3',
            label_fig = 'cycle3, IFS 4.4km, FESOM 5km',
            startdate = '20200121',
            enddate = '20241231',
            linestyle = '-',
            #colour = 'darkred'),
            colour = '#5E1916'),
        ngc2013 = dict(
            cycle = '2',
            grid = 'native',
            duration = 17.7,
            atmos_resol = '10 km',
            label = 'ngc2013',
            label_fig = 'cycle2, ICON, 10km, TTE',
            startdate = '20200120',
            enddate = '20371001',
            linestyle = '-',
            colour = 'lightskyblue'),
        ngc2012 = dict(
            cycle = '2',
            grid = 'native',
            duration = 8.04,
            atmos_resol = '10 km',
            label = 'ngc2012',
            label_fig = 'cycle2, ICON, 10km, Smag.',
            startdate = '20200120',
            enddate = '20280201',
            linestyle = '-',
            colour = 'dodgerblue'),
        ngc2009 = dict(
            cycle = '2',
            grid = 'native',
            duration = 2.11,
            atmos_resol = '5 km',
            label = 'ngc2009',
            label_fig = 'cycle2, ICON, 5km, Smag.',
            startdate = '20200120',
            enddate = '20220301',
            linestyle = '-',
            colour = 'b'),
        ngc3028_8 = dict(
            cycle = '3',
            grid = 'regular',
            duration = 4.95,
            atmos_resol = '6 km',
            label = 'ngc3028',
            label_fig = 'cycle3, ICON, zoom8, 24km',
            startdate = '20200121',
            enddate = '20241231',
            linestyle = '-',
            colour = 'darkgrey'),
        ngc3028_10 = dict(
            cycle = '3',
            grid = 'regular',
            duration = 4.95,
            atmos_resol = '24 km',
            label = 'ngc3028',
            label_fig = 'cycle3, ICON, zoom10, 6km',
            startdate = '20200121',
            enddate = '20241231',
            linestyle = '-',
            colour = 'slategrey'))

    return MEMBERS


# Tropical cyclone categories (pressure based on Klotzbach et al., 2020; Bourdin et al., 2022 and wind based on NHC)
def declare_categories(based_on):
    if based_on == 'mslp':      # hPa
        categories = OrderedDict(
            TD = dict(pmax = 1020.,pmin = 1005.),
            TS = dict(pmax = 1004.99,pmin = 990.),
            CAT1 = dict(pmax = 989.99,pmin = 975.),
            CAT2 = dict(pmax = 974.99,pmin = 960.),
            CAT3 = dict(pmax = 959.99,pmin = 945.),
            CAT4 = dict(pmax = 944.99,pmin = 925.),
            CAT5 = dict(pmax = 924.99,pmin = 800.))
    elif based_on == 'vmax':    # m/s
        categories = OrderedDict(
            TD = dict(vmin = 0.,vmax = 16.99),
            TS = dict(vmin = 17.,vmax = 31.99),
            CAT1 = dict(vmin = 33.,vmax = 42.99),
            CAT2 = dict(vmin = 43.,vmax = 49.99),
            CAT3 = dict(vmin = 50.,vmax = 57.99),
            CAT4 = dict(vmin = 58.,vmax = 69.99),
            CAT5 = dict(vmin = 70.,vmax = 200.))
    return categories


# Convert numpy datetime64 format (used in IBTrACS NetCDF data) to datetime format
def numpy_datetime64_to_datetime(numpy_datetime):
    timestamp = ((numpy_datetime - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)


# Get 24-hour intensification rates from TRACK / TempestExtremes data (read with storm_assess)
def get_storm_dv_storm_assess(storm, pmin = False):
    n_idx = 3   # 6h data
    storm_dv = []
    if pmin:
        values = [storm.obs[i].mslp for i in range(storm.nrecords())]
    else:
        values = [storm.obs[i].vmax for i in range(storm.nrecords())]
    for i in range(storm.nrecords()-n_idx):
        if values[i+n_idx] > 0. and values[i] > 0.:
            dv = values[i+n_idx]-values[i]
            storm_dv.append(dv)
    return storm_dv


# Get 24-hour intensification rates from IBTrACS data in .nc format (read with xarray)
def get_storm_dv_ibtracs(values, six_hourly = False):
    if six_hourly:
        n_idx = 3
    else:
        n_idx = 7   # data as read in is 3 hourly

    storm_dv = []
    for i in range(len(values)-n_idx):
        if not np.isnan(values[i+n_idx]) and not np.isnan(values[i]):
            dv = values[i+n_idx]-values[i]
            storm_dv.append(dv)
    return storm_dv


def process_ibtracs_pressure_wind_to_10min(ibtracs_ffp_nc,year_start,year_end,basin = None):
    years = range(year_start,year_end+1)

    ibtracs_data = xr.open_dataset(ibtracs_ffp_nc)

    ibtracs_time = ibtracs_data.iso_time
    ibtracs_lat = ibtracs_data.lat
    ibtracs_mslp = ibtracs_data.wmo_pres
    ibtracs_vmax = ibtracs_data.wmo_wind
    ibtracs_type = ibtracs_data.track_type
    ibtracs_numobs = ibtracs_data.numobs
    ibtracs_agency = ibtracs_data.wmo_agency

    n_storms = len(ibtracs_data.storm)
    main_idx = [i for i in range(n_storms) if ibtracs_type[i].data == b'main']                            # i.e., omit "spur" storms
    n_spurs = n_storms - len(main_idx)
    print(' ...removed {} spurs'.format(n_spurs))

    if not basin is None:
        ibtracs_basin = ibtracs_data.basin
        if isinstance(basin,str):
            basin = [basin.upper()]
        elif isinstance(basin,list):
            basin = [b.upper() for b in basin]
        print(' ...filtering {}'.format(', '.join(basin)))

    ibtracs_mslp_minima = list()
    ibtracs_vmax_maxima = list()
    print(' ...converting wind speeds to WMO-standard 10-minute sustained')
    for idx in main_idx:
        numobs = int(ibtracs_numobs[idx].data)-1
        if basin is None:
            domain_include = True                                                                         # None = global, so include all
        else:
            storm_basin = ibtracs_basin[idx][:numobs].data.astype(str).tolist()
            if any([storm_basin[i] in basin for i in range(numobs)]):                                     # storm intersects basin
                if basin == ['EP'] and ('EP' in storm_basin and 'WP' in storm_basin):                     # check whether predominantly WP storms are included in EP
                    if (storm_basin == 'EP').sum() > (storm_basin == 'WP').sum():
                        domain_include = True
                    else:
                        domain_include = False
                else:
                    domain_include = True
            else:
                domain_include = False
        if domain_include:
            d_1 = numpy_datetime64_to_datetime(ibtracs_lat[idx].time.data[0])                                 # 1st timestep
            if d_1.year in years:
                if numobs > 1:
                    d_2 = numpy_datetime64_to_datetime(ibtracs_lat[idx].time.data[1])                         # 2nd timestep
                    d_end = numpy_datetime64_to_datetime(ibtracs_lat[idx].time.data[numobs])                  # last timestep
                    storm_lifetime = (d_end-d_1).days
                    storm_mslp = ibtracs_mslp[idx][:numobs].data
                    storm_vmax = ibtracs_vmax[idx][:numobs].data / 1.94384                                    # convert to m/s
                    storm_agency = ibtracs_agency[idx][:numobs].data                                          # WMO agency for each wind observation
                    if not np.isnan(storm_vmax).all() and not np.isnan(storm_mslp).all():
                        include_storm = True
                    elif not np.isnan(storm_mslp).all():                                                      # get non-WMO winds (NOAA, JTWC or CMA)
                        if not np.isnan(ibtracs_data.usa_wind[idx][:numobs].data).all():
                            storm_vmax = ibtracs_data.usa_wind[idx][:numobs].data / 1.94384
                            include_storm = True
                        elif not np.isnan(ibtracs_data.cma_wind[idx][:numobs].data).all():
                            storm_vmax = ibtracs_data.cma_wind[idx][:numobs].data / 1.94384
                            include_storm = True
                        else:
                            include_storm = False
                    else:
                        include_storm = False
                    if include_storm:
                        if storm_lifetime >= 1.:#and np.nanmax(storm_vmax) >= 17.:                            # omit < 1 day (and TD category)
                            vmax_idx = np.where(storm_vmax == np.nanmax(storm_vmax))[0]
                            for vmax_i in vmax_idx:
                                vmax_t = ibtracs_time[idx][vmax_i].data.astype(str).tolist()
                                if vmax_t.endswith('00:00:00') or vmax_t.endswith('06:00:00')\
                                    or vmax_t.endswith('12:00:00') or vmax_t.endswith('18:00:00'):            # take only timesteps 0/6/12/18 UTC
                                    try:
                                        vmax_agency = ibtracs_agency[idx][vmax_i].data.astype(str)[0]
                                    except:
                                        vmax_agency = ibtracs_agency[idx][vmax_i].data.astype(str).tolist()
                                    if vmax_agency in ['cphc','hurdat_epa','hurdat_atl','jtwc_wp']:           # convert 1-minute to 10-minute
                                        #ratio = 1./1.05
                                        #ratio = 0.88                                                         # conversions assume "at-sea" gust factors (see WMO/TD-No.1555)
                                        ratio = 0.93
                                    elif vmax_agency in ['cma']:                                              # convert 2-minute to 10-minute
                                        #ratio = 1./1.00
                                        ratio = 0.93
                                    elif vmax_agency in ['newdelhi']:                                         # convert 3-minute to 10-minute
                                        #ratio = 1./1.00
                                        ratio = 0.93
                                    else:
                                        ratio = 1./1.00
                                    ibtracs_mslp_minima.append(np.nanmin(storm_mslp))
                                    ibtracs_vmax_maxima.append(np.nanmax(storm_vmax) * ratio)
                                    break                                                                     # just take first vmax instance


if __name__ == '__main__':
    pass

